import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import torch
from research.kira_model.bpe_experiment import BPEModel, TextCodec, fit_codec, generate, recode, report_validation, train
from research.kira_model.data import encode, read_corpus
from research.kira_model.model import Config

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke.jsonl"

def options(corpus, output, **changes):
    values = dict(corpus=corpus, output=output, steps=6, batch_size=1, threads=1, lr=0.001,
                  width=16, heads=2, layers=1, context=32, seed=42, vocab_size=384,
                  checkpoint_every=2, resume=False)
    return SimpleNamespace(**(values | changes))

class BPETests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.documents, _ = read_corpus(FIXTURE)

    def test_roundtrip_and_local_vocabulary_compression(self):
        with mock.patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
            codec = fit_codec(self.documents, 384)
        for text in ("français 中文 🧠\n", "  des espaces  ", "<|kira_bos|> et <EOS>", ""):
            self.assertEqual(codec.decode(codec.encode(text)), text)
        text = "Une chercheuse prépare un carnet."
        self.assertLess(len(codec.encode(text)), len(encode(text)))
        self.assertEqual(codec.bos + 1, codec.eos)
        self.assertEqual(codec.eos + 1, codec.vocab_size)

    def test_validation_and_test_never_train_tokenizer(self):
        changed = copy.deepcopy(self.documents)
        changed["validation"] = [{"tokens": encode("UNSEEN_VALIDATION_" * 100)}]
        changed["test"] = [{"tokens": encode("UNSEEN_TEST_" * 100)}]
        a, b = fit_codec(self.documents, 384), fit_codec(changed, 384)
        self.assertEqual(json.loads(a.serialized), json.loads(b.serialized))

    def test_embedded_tokenizer_and_model_shape(self):
        codec = fit_codec(self.documents, 384)
        reconstructed = TextCodec(codec.serialized)
        self.assertEqual(codec.encode("Un récit français."), reconstructed.encode("Un récit français."))
        model = BPEModel(Config(width=16, heads=2, layers=1, context=32), codec.vocab_size)
        self.assertIs(model.head.weight, model.token.weight)
        x = torch.tensor([codec.encode("Un récit")])
        self.assertEqual(tuple(model(x).shape), (1, x.size(1), codec.vocab_size))
        measurement = report_validation(model, recode(self.documents, codec)["validation"], 32)
        self.assertGreater(measurement["bits_per_utf8_byte"], 0)
        self.assertGreater(measurement["predicted_tokens"], 0)
        self.assertGreater(measurement["utf8_bytes"], measurement["predicted_tokens"])

    def test_interrupted_resume_matches_uninterrupted_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(FIXTURE, Path(tmp) / "full")
            train(args)
            full = torch.load(args.output / "weights.pt", weights_only=True)
            args.output = Path(tmp) / "interrupted"
            count = 0
            original = torch.nn.utils.clip_grad_norm_
            def interrupt(*a, **kw):
                nonlocal count
                count += 1
                if count == 5:
                    raise KeyboardInterrupt()
                return original(*a, **kw)
            with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    train(args)
            state = torch.load(args.output / "training.pt", weights_only=True)
            self.assertEqual(state["step"], 4)
            args.resume = True
            train(args)
            resumed = torch.load(args.output / "weights.pt", weights_only=True)
            self.assertEqual(full["tokenizer_json"], resumed["tokenizer_json"])
            for name, value in full["state_dict"].items():
                self.assertTrue(torch.equal(value, resumed["state_dict"][name]), name)

    def test_incompatible_resume_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(FIXTURE, Path(tmp) / "run", steps=2)
            train(args)
            before = (args.output / "training.pt").read_bytes()
            args.steps, args.resume, args.vocab_size = 4, True, 512
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                train(args)
            self.assertEqual(before, (args.output / "training.pt").read_bytes())
            args.vocab_size = 384
            args.corpus = Path(tmp) / "changed.jsonl"
            args.corpus.write_bytes(FIXTURE.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                train(args)
            self.assertEqual(before, (args.output / "training.pt").read_bytes())

    def test_self_contained_checkpoint_and_existing_run_protected(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(FIXTURE, Path(tmp) / "run", steps=2)
            train(args)
            metadata = json.loads((args.output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["initialization"], "random")
            self.assertFalse(metadata["test_split_evaluated"])
            self.assertEqual(metadata["steps"], 2)
            (args.output / "tokenizer.json").unlink()
            generate(SimpleNamespace(checkpoint=args.output / "weights.pt", prompt="Une chercheuse",
                                     threads=1, seed=42, max_new_tokens=5, temperature=0.5, top_k=10))
            with self.assertRaises(FileExistsError):
                train(args)

    def test_legacy_checkpoint_refused_without_conversion(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.pt"
            torch.save({"config": {"width": 16}, "tokenizer": "utf8-bytes-v1"}, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, "Checkpoint incompatible"):
                generate(SimpleNamespace(checkpoint=path, prompt="Une chercheuse", threads=1,
                                         seed=42, max_new_tokens=5, temperature=0.5, top_k=10))
            self.assertEqual(before, path.read_bytes())

    def test_heldout_text_does_not_change_learned_weights(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            states = []
            for i in range(2):
                corpus = Path(tmp) / f"data-{i}.jsonl"
                row = {"id": "test-extra", "source": "test-fixture", "license": "project-authored",
                       "training_allowed": True, "split": "test", "text": f"Texte réservé {i}."}
                corpus.write_text(FIXTURE.read_text(encoding="utf-8") + json.dumps(row) + "\n",
                                  encoding="utf-8")
                args = options(corpus, Path(tmp) / f"run-{i}", steps=2)
                train(args)
                states.append(torch.load(args.output / "weights.pt", weights_only=True))
            self.assertEqual(states[0]["tokenizer_json"], states[1]["tokenizer_json"])
            for name, value in states[0]["state_dict"].items():
                self.assertTrue(torch.equal(value, states[1]["state_dict"][name]), name)

    def test_validation_matches_uniform_model_across_windows(self):
        codec = fit_codec(self.documents, 384)
        rows = recode(self.documents, codec)["validation"]
        class UniformModel(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(*x.shape, codec.vocab_size)
        measured = report_validation(UniformModel(), rows, 7)
        expected_tokens = sum(len(row["tokens"]) - 1 for row in rows)
        expected_bytes = sum(row["utf8_bytes"] for row in rows)
        self.assertEqual(measured["predicted_tokens"], expected_tokens)
        self.assertAlmostEqual(measured["loss_per_token"], math.log(codec.vocab_size), places=5)
        self.assertAlmostEqual(measured["bits_per_utf8_byte"],
                               math.log2(codec.vocab_size) * expected_tokens / expected_bytes, places=5)
