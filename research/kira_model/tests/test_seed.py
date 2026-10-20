import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from research.kira_model.data import BOS, EOS, decode, encode, read_corpus
from research.kira_model.model import Config, KiraSeed
from research.kira_model.__main__ import batch, evaluate, generate, train

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke.jsonl"


class CorpusTests(unittest.TestCase):
    def test_utf8_roundtrip_and_boundaries(self):
        text = "KIRA : français, 中文, 🧠\n"
        self.assertEqual(decode(encode(text)), text)
        self.assertEqual(encode(text)[0], BOS)
        self.assertEqual(encode(text)[-1], EOS)
        self.assertNotEqual(encode(text, end=False)[-1], EOS)

    def check_bad_row(self, mutate):
        rows = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
        mutate(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaises(ValueError):
                read_corpus(path)

    def test_explicit_authorization_required(self):
        self.check_bad_row(lambda rows: rows[0].update(training_allowed=False))

    def test_provenance_required(self):
        self.check_bad_row(lambda rows: rows[0].update(source=""))

    def test_normalized_duplicates_across_splits_rejected(self):
        self.check_bad_row(lambda rows: rows[-1].update(text="  " + rows[0]["text"].upper() + "\n"))

    def test_no_validation_is_rejected(self):
        self.check_bad_row(lambda rows: [row.update(split="train") for row in rows])

    def test_corpus_read_is_bounded(self):
        with mock.patch("research.kira_model.data.MAX_CORPUS_BYTES", 64):
            with self.assertRaisesRegex(ValueError, "Corpus supérieur"):
                read_corpus(FIXTURE)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_no_attention_to_future_tokens(self):
        torch.manual_seed(8)
        model = KiraSeed(Config(width=16, heads=2, layers=1, context=16)).eval()
        a = torch.tensor([[BOS, 4, 5, 6, 7]])
        b = torch.tensor([[BOS, 4, 5, 90, 91]])
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(a)[:, :3], model(b)[:, :3], atol=1e-6))
            self.assertFalse(torch.allclose(model(a)[:, 3:], model(b)[:, 3:]))

    def test_padding_does_not_become_training_targets(self):
        x, y = batch([[BOS, 65, EOS]], 8)
        self.assertEqual(x[0, :2].tolist(), [BOS, 65])
        self.assertEqual(y[0].tolist(), [65, EOS, -100, -100, -100, -100, -100, -100])

    def test_weights_learn_and_reload_exactly(self):
        torch.manual_seed(9)
        cfg = Config(width=16, heads=2, layers=1, context=16)
        model = KiraSeed(cfg)
        x, y = batch([encode("abcabcabcabc")], cfg.context)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
        initial = F.cross_entropy(model(x).flatten(0, 1), y.flatten()).item()
        for _ in range(30):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
            loss.backward()
            optimizer.step()
        self.assertLess(loss.item(), initial / 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.pt"
            torch.save(model.state_dict(), path)
            copy = KiraSeed(cfg)
            copy.load_state_dict(torch.load(path, weights_only=True))
            with torch.no_grad():
                self.assertTrue(torch.equal(model(x), copy(x)))

    def test_cpu_training_manifest_and_generation(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp) / "run"
            args = SimpleNamespace(corpus=FIXTURE, output=output, steps=3, batch_size=2,
                                   threads=2, lr=0.001, width=16, heads=2, layers=1, context=32, seed=42)
            train(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["initialization"], "random")
            self.assertEqual(manifest["steps"], 3)
            self.assertEqual(len(manifest["documents"]["validation"]), 2)
            checkpoint = torch.load(output / "weights.pt", weights_only=True)
            model = KiraSeed(Config(**checkpoint["config"]))
            model.load_state_dict(checkpoint["state_dict"])
            docs, _ = read_corpus(FIXTURE)
            self.assertAlmostEqual(evaluate(model, docs["validation"], 32), manifest["validation_loss_final"])
            generate(SimpleNamespace(checkpoint=output / "weights.pt", prompt="Bonjour", threads=2,
                                     seed=42, max_new_tokens=5, temperature=0))
            with self.assertRaises(FileExistsError):
                train(args)

    def test_interruption_resumes_same_weights_as_uninterrupted_run(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(corpus=FIXTURE, output=Path(tmp) / "full", steps=8, batch_size=1,
                                   threads=1, lr=0.001, width=16, heads=2, layers=1, context=32,
                                   seed=42, resume=False, checkpoint_every=2)
            train(args)
            full = torch.load(args.output / "weights.pt", weights_only=True)
            args.output = Path(tmp) / "interrupted"
            calls = 0
            clip = torch.nn.utils.clip_grad_norm_

            def interrupt_on_fifth(*a, **kw):
                nonlocal calls
                calls += 1
                if calls == 5:
                    raise KeyboardInterrupt()
                return clip(*a, **kw)

            with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=interrupt_on_fifth):
                with self.assertRaises(KeyboardInterrupt):
                    train(args)
            saved = torch.load(args.output / "training.pt", weights_only=True)
            self.assertEqual(saved["step"], 4)
            args.resume = True
            train(args)
            resumed = torch.load(args.output / "weights.pt", weights_only=True)
            for name, weight in full["state_dict"].items():
                self.assertTrue(torch.equal(weight, resumed["state_dict"][name]), name)
            manifest = json.loads((args.output / "manifest.json").read_text())
            self.assertEqual(manifest["resumed_from_step"], 4)
            self.assertEqual(manifest["steps"], 8)

    def test_incompatible_resume_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = SimpleNamespace(corpus=FIXTURE, output=Path(tmp) / "run", steps=2, batch_size=1,
                                   threads=1, lr=0.001, width=16, heads=2, layers=1, context=32,
                                   seed=42, resume=False, checkpoint_every=1)
            train(args)
            before = (args.output / "training.pt").read_bytes()
            args.resume, args.steps, args.batch_size = True, 4, 2
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                train(args)
            self.assertEqual((args.output / "training.pt").read_bytes(), before)
            args.batch_size, args.steps = 1, 2
            with self.assertRaisesRegex(ValueError, "steps doit dépasser"):
                train(args)
            args.steps = 4
            args.corpus = Path(tmp) / "changed.jsonl"
            args.corpus.write_bytes(FIXTURE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                train(args)
            self.assertEqual((args.output / "training.pt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
