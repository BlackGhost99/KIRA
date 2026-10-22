import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from research.kira_model import language_phase as phase
from research.kira_model import prepare_language as data
from research.kira_model.bpe_experiment import BPEModel, FORMAT as BASE_FORMAT, fit_codec
from research.kira_model.data import read_corpus
from research.kira_model.model import Config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke.jsonl"


def options(base, corpus, output, **changes):
    values = dict(base=base, corpus=corpus, replay_corpus=corpus, output=output, steps=8,
                  threads=1, batch_size=1, seed=42, warmup=1, eval_every=4, checkpoint_every=2,
                  lr=0.003, replay_ratio=0.25, max_replay_regression=0.35, stop_after=None, resume=False)
    return SimpleNamespace(**(values | changes))


class LanguagePhaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        docs, _ = read_corpus(FIXTURE)
        cls.codec = fit_codec(docs, 384)
        torch.manual_seed(21)
        cls.model = BPEModel(Config(width=16, heads=2, layers=1, context=128), cls.codec.vocab_size)
        cls.base = cls.root / "base.pt"
        torch.save({"format": BASE_FORMAT, "config": vars(cls.model.cfg),
                    "tokenizer_json": cls.codec.serialized, "state_dict": cls.model.state_dict()}, cls.base)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_base_weights_and_codec_are_preserved_before_training(self):
        before = self.base.read_bytes()
        model, codec = phase.load_base(self.base)
        self.assertEqual(codec.serialized, self.codec.serialized)
        self.assertIs(model.head.weight, model.token.weight)
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(model.state_dict()[key], value))
        self.assertEqual(before, self.base.read_bytes())

    def test_profile_and_resume_match_continuous_training_and_selection(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            a = options(self.base, FIXTURE, root / "full")
            expected = phase.train(a)
            a.output, a.stop_after = root / "segmented", 3
            profile = phase.train(a)
            self.assertEqual(profile["best_step"], 0)  # Le profilage ne sélectionne pas à l'étape 3.
            a.resume, a.stop_after, a.base = True, None, None
            actual = phase.train(a)
            self.assertEqual(actual["history"], expected["history"])
            self.assertEqual(actual["new_tokens_seen"], expected["new_tokens_seen"])
            self.assertEqual(actual["replay_tokens_seen"], expected["replay_tokens_seen"])
            for name in ("weights.pt", "last.pt"):
                x = torch.load(root / "full" / name, weights_only=True)
                y = torch.load(root / "segmented" / name, weights_only=True)
                self.assertEqual(x["selected_step"], y["selected_step"])
                for key in x["state_dict"]:
                    self.assertTrue(torch.equal(x["state_dict"][key], y["state_dict"][key]))

    def test_incompatible_resume_does_not_modify_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            a = options(self.base, FIXTURE, Path(temp) / "phase", stop_after=2)
            phase.train(a)
            before = (a.output / "training.pt").read_bytes()
            a.resume, a.stop_after, a.lr = True, None, 0.004
            with self.assertRaisesRegex(ValueError, "incompatible"):
                phase.train(a)
            self.assertEqual(before, (a.output / "training.pt").read_bytes())

    def test_heldout_test_text_never_affects_gradients_or_selection(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            rows = [json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()]
            for row in rows:
                if row["split"] == "test":
                    row["text"] = "Texte réservé changé, qui ne doit jamais être appris."
            altered = root / "altered.jsonl"
            altered.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
            phase.train(options(self.base, FIXTURE, root / "a", replay_corpus=None))
            phase.train(options(self.base, altered, root / "b", replay_corpus=None))
            x = torch.load(root / "a" / "weights.pt", weights_only=True)
            y = torch.load(root / "b" / "weights.pt", weights_only=True)
            for key in x["state_dict"]:
                self.assertTrue(torch.equal(x["state_dict"][key], y["state_dict"][key]))
            self.assertEqual(x["selected_step"], y["selected_step"])

    def test_lower_new_loss_cannot_select_excessive_old_language_regression(self):
        self.assertFalse(phase.select_candidate(1.0, 4.0, 3.0, 2.0, 0.35))
        self.assertTrue(phase.select_candidate(1.0, 3.3, 3.0, 2.0, 0.35))
        self.assertFalse(phase.select_candidate(float("nan"), None, None, 2.0, 0.35))
        self.assertTrue(phase.select_candidate(1.0, None, None, 2.0, 0.35))


class StorySelectionTests(unittest.TestCase):
    def test_complete_texts_and_official_split_origins_are_preserved(self):
        val = "Une histoire de validation distincte. " + "v" * 210
        train = "Une histoire d'entraînement distincte. " + "t" * 210
        raw, _ = data.select_documents([(0, val)], [(0, train)], 1, 1)
        rows = [json.loads(x) for x in raw.splitlines()]
        self.assertEqual({r["text"] for r in rows}, {val, train})
        self.assertEqual({r["split"] for r in rows}, {"train", "validation"})
        self.assertTrue(all(r["synthetic"] and r["training_allowed"] for r in rows))
        self.assertTrue(all(r["license"] == "ODC-By-1.0" for r in rows))

    def test_shared_prefixes_and_normalized_duplicates_cannot_cross_splits(self):
        val = "Histoire A. " + "x" * 200
        other = "Histoire B. " + "y" * 200
        for duplicate in (val.upper(), val[:180] + "z" * 200):
            raw, stats = data.select_documents([(0, val)], [(0, duplicate), (89, other)], 1, 1)
            self.assertEqual(stats["duplicate_text_or_prefix"], 1)
            self.assertIn(other, raw.decode())

    def test_incomplete_and_oversized_samples_fail_before_export(self):
        with self.assertRaisesRegex(ValueError, "incomplet"):
            data.select_documents([], [], 1, 1)
        a, b = "A" * 250, "B" * 250
        with mock.patch.object(data, "MAX_OUTPUT", 100), self.assertRaisesRegex(ValueError, "14 Mio"):
            data.select_documents([(0, a)], [(0, b)], 1, 1)

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(data, "acquire") as acquire:
            with self.assertRaises(FileExistsError):
                data.prepare(Path(temp), Path(temp))
            acquire.assert_not_called()


if __name__ == "__main__":
    unittest.main()
