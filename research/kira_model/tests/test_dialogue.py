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

from research.kira_model import dialogue as d
from research.kira_model import dialogue_data as data
from research.kira_model.bpe_experiment import BPEModel, fit_codec, FORMAT as BASE_FORMAT
from research.kira_model.data import read_corpus
from research.kira_model.model import Config


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke.jsonl"


def options(base, path, output, **changes):
    values = dict(base=base, data=path, output=output, steps=6, batch_size=1, threads=1,
                  seed=42, warmup=1, eval_every=3, checkpoint_every=2, lr=0.005,
                  resume=False, stop_after=None)
    return SimpleNamespace(**(values | changes))


class CurriculumTests(unittest.TestCase):
    def test_original_curriculum_is_deterministic_and_closed_answers_are_correct(self):
        rows = data.curriculum()
        groups = data.validate(rows)
        self.assertEqual(rows, data.curriculum())
        self.assertGreater(len(groups["train"]), 700)
        self.assertGreater(len(groups["validation"]), 90)
        self.assertGreater(len(groups["test"]), 90)
        for item in rows:
            prompt = item["messages"][0]["content"]
            answer = item["messages"][-1]["content"]
            if item["skill"] in ("addition", "soustraction"):
                import re
                a, b = map(int, re.findall(r"\d+", prompt))
                self.assertEqual(answer, str(a + b if item["skill"] == "addition" else a - b))
            self.assertTrue(item["synthetic"])

    def test_commuted_additions_share_a_split(self):
        facts = {}
        for item in data.curriculum():
            if item["skill"] == "addition":
                import re
                numbers = re.findall(r"\d+", item["messages"][0]["content"])
                operands = tuple(sorted(map(int, numbers)))
                self.assertEqual(facts.setdefault(operands, item["split"]), item["split"])

    def test_groups_and_duplicate_inputs_cannot_leak_between_splits(self):
        rows = [data.row("a", "Question unique A", "A", "train"),
                data.row("b", "Question unique B", "B", "validation")]
        duplicate = copy.deepcopy(rows[0])
        duplicate.update(id="other", group="other", split="validation")
        duplicate["messages"][-1]["content"] = "Autre réponse."
        with self.assertRaisesRegex(ValueError, "dupliqué"):
            data.validate(rows + [duplicate])
        rows[1]["group"] = rows[0]["group"]
        with self.assertRaisesRegex(ValueError, "plusieurs splits"):
            data.validate(rows)

    def test_authorization_and_role_order_are_required(self):
        rows = data.curriculum()[:2]
        rows[1]["split"] = "validation"
        rows[0]["training_allowed"] = False
        with self.assertRaisesRegex(ValueError, "autorisation"):
            data.validate(rows)
        rows[0]["training_allowed"] = True
        rows[0]["messages"][0]["role"] = "assistant"
        with self.assertRaisesRegex(ValueError, "rôles"):
            data.validate(rows)

    def test_prepare_reuses_identical_data_and_protects_other_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curriculum.jsonl"
            a = data.prepare(path)
            self.assertEqual(a, data.prepare(path))
            groups, digest = data.load(path)
            self.assertEqual(a["sha256"], digest)
            self.assertEqual(a["examples"]["test"], len(groups["test"]))
            path.write_text("autre contenu", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                data.prepare(path)
            self.assertEqual(path.read_text(), "autre contenu")

    def test_reader_bounds_memory(self):
        path = mock.Mock()
        path.open.return_value = contextlib.closing(io.BytesIO(b" " * (data.MAX_BYTES + 1)))
        with self.assertRaisesRegex(ValueError, "8 Mio"):
            data.load(path)


class DialogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        corpus, _ = read_corpus(FIXTURE)
        cls.codec = fit_codec(corpus, 384)
        torch.manual_seed(21)
        cls.base_model = BPEModel(Config(width=16, heads=2, layers=1, context=128), cls.codec.vocab_size)
        cls.base = cls.root / "base.pt"
        cls.base_value = {"format": BASE_FORMAT, "config": vars(cls.base_model.cfg),
                          "tokenizer_json": cls.codec.serialized, "state_dict": cls.base_model.state_dict()}
        torch.save(cls.base_value, cls.base)
        cls.rows = [data.row("greet", "Bonjour.", "Salut !", "train", "reference_only"),
                    data.row("greet", "Salut.", "Salut !", "train", "reference_only"),
                    data.row("greet", "Bonsoir.", "Salut !", "validation", "reference_only"),
                    data.row("copy", "Copie le nombre 9.", "9", "test")]
        cls.path = cls.root / "data.jsonl"
        cls.path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in cls.rows) + "\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_extending_roles_preserves_every_original_weight_and_token(self):
        before = self.base.read_bytes()
        model, codec = d.load_model(self.base)
        self.assertEqual(codec.serialized, self.codec.serialized)
        self.assertIs(model.token.weight, model.head.weight)
        for key, value in self.base_model.state_dict().items():
            actual = model.state_dict()[key]
            if key in ("token.weight", "head.weight"):
                self.assertTrue(torch.equal(value, actual[:codec.vocab_size]))
                self.assertEqual(actual.size(0), value.size(0) + 2)
            else:
                self.assertTrue(torch.equal(value, actual))
        self.assertEqual(before, self.base.read_bytes())

    def test_only_assistant_tokens_and_eos_are_learning_targets(self):
        turns = [{"role": "user", "content": "Utilisateur : <assistant> 🧠"},
                 {"role": "assistant", "content": "Bonjour."}]
        ids, mask = d.encode_messages(turns, self.codec)
        item = {"ids": ids, "mask": mask}
        x, y = d.make_batch([item])
        expected = self.codec.backend.encode("Bonjour.", add_special_tokens=False).ids + [self.codec.eos]
        self.assertEqual(y[y != -100].tolist(), expected)
        self.assertEqual(ids.count(self.codec.vocab_size), 1)
        self.assertEqual(ids.count(self.codec.vocab_size + 1), 1)
        self.assertFalse(mask[ids.index(self.codec.vocab_size + 1)])
        self.assertEqual(x.shape, y.shape)
        self.assertLess(x.size(1), 128)

    def test_dynamic_padding_does_not_add_targets(self):
        examples = d.encode_rows(self.rows[:2], self.codec, 128)
        examples[1]["ids"] = examples[1]["ids"][:-2]
        examples[1]["mask"] = examples[1]["mask"][:-2]
        x, y = d.make_batch(examples)
        self.assertEqual(x.size(1), max(len(e["ids"]) - 1 for e in examples))
        self.assertEqual(int((y != -100).sum()), sum(sum(e["mask"][1:]) for e in examples))
        self.assertEqual(y[1, -1].item(), -100)

    def test_overlong_examples_are_rejected_before_run_creation(self):
        rows = [data.row("long", "a" * 500, "b", "train")]
        with self.assertRaisesRegex(ValueError, "contexte"):
            d.encode_rows(rows, self.codec, 8)

    def test_interrupted_resume_matches_full_weights_and_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "full")
            d.train(args)
            expected = torch.load(args.output / "last.pt", weights_only=True)
            best_expected = torch.load(args.output / "best.pt", weights_only=True)
            args.output = Path(tmp) / "interrupted"
            original = torch.nn.utils.clip_grad_norm_
            count = 0
            def interrupt(*a, **kw):
                nonlocal count
                count += 1
                if count == 5:
                    raise KeyboardInterrupt()
                return original(*a, **kw)
            with mock.patch("torch.nn.utils.clip_grad_norm_", side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    d.train(args)
            saved = torch.load(args.output / "training.pt", weights_only=True)
            self.assertEqual(saved["step"], 4)
            args.resume, args.base = True, None
            d.train(args)
            actual = torch.load(args.output / "last.pt", weights_only=True)
            best_actual = torch.load(args.output / "best.pt", weights_only=True)
            for key, value in expected["state_dict"].items():
                self.assertTrue(torch.equal(value, actual["state_dict"][key]), key)
                self.assertTrue(torch.equal(best_expected["state_dict"][key], best_actual["state_dict"][key]), key)
            self.assertEqual(best_expected["selected_step"], best_actual["selected_step"])

    def test_profile_segment_does_not_change_schedule_or_best_selection(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "full")
            d.train(args)
            full = torch.load(args.output / "best.pt", weights_only=True)
            args.output, args.stop_after = Path(tmp) / "split", 4
            d.train(args)
            args.stop_after, args.resume, args.base = None, True, None
            d.train(args)
            split = torch.load(args.output / "best.pt", weights_only=True)
            self.assertEqual(full["selected_step"], split["selected_step"])
            for key in full["state_dict"]:
                self.assertTrue(torch.equal(full["state_dict"][key], split["state_dict"][key]), key)

    def test_resume_rejects_changed_schedule_data_and_base_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "run", stop_after=2)
            d.train(args)
            before = (args.output / "training.pt").read_bytes()
            args.resume, args.stop_after, args.steps = True, None, 8
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                d.train(args)
            args.steps = 6
            changed = Path(tmp) / "other.jsonl"
            changed.write_bytes(self.path.read_bytes() + b"\n")
            args.data = changed
            with self.assertRaisesRegex(ValueError, "Reprise incompatible"):
                d.train(args)
            args.data, args.base = self.path, Path(tmp) / "other.pt"
            torch.save({**self.base_value, "extra": 1}, args.base)
            with self.assertRaisesRegex(ValueError, "départ différents"):
                d.train(args)
            self.assertEqual(before, (args.output / "training.pt").read_bytes())

    def test_best_checkpoint_survives_later_worse_validation_and_missing_export(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "run", stop_after=3)
            measure = lambda loss: {"assistant_loss_per_token": loss, "assistant_target_tokens": 10}
            with mock.patch.object(d, "validation", side_effect=[measure(10), measure(8)]):
                d.train(args)
            expected = torch.load(args.output / "best.pt", weights_only=True)
            self.assertEqual(expected["selected_step"], 3)
            (args.output / "best.pt").unlink()
            args.resume, args.base, args.stop_after = True, None, None
            with mock.patch.object(d, "validation", return_value=measure(12)):
                d.train(args)
            actual = torch.load(args.output / "best.pt", weights_only=True)
            self.assertEqual(actual["selected_step"], 3)
            for key in expected["state_dict"]:
                self.assertTrue(torch.equal(expected["state_dict"][key], actual["state_dict"][key]), key)

    def test_training_learns_and_does_not_use_test_text(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            states = []
            for i in range(2):
                rows = copy.deepcopy(self.rows)
                rows[-1]["messages"][0]["content"] = "Texte réservé " + "x" * (1000 + i)
                path = Path(tmp) / f"data-{i}.jsonl"
                path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
                args = options(self.base, path, Path(tmp) / f"run-{i}", steps=24, eval_every=6)
                with mock.patch("urllib.request.urlopen", side_effect=AssertionError("No network")):
                    d.train(args)
                report = json.loads((args.output / "manifest.json").read_text())
                self.assertLess(report["best_validation_loss"], report["validation_initial"]["assistant_loss_per_token"])
                self.assertFalse(report["test_split_evaluated"])
                states.append(torch.load(args.output / "last.pt", weights_only=True))
            for key in states[0]["state_dict"]:
                self.assertTrue(torch.equal(states[0]["state_dict"][key], states[1]["state_dict"][key]), key)

    def test_replay_resume_preserves_weights_and_counters(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "full", replay_corpus=FIXTURE, replay_ratio=0.6)
            d.train(args)
            expected = torch.load(args.output / "last.pt", weights_only=True)
            args.output, args.stop_after = Path(tmp) / "split", 4
            d.train(args)
            args.resume, args.stop_after, args.base = True, None, None
            d.train(args)
            actual = torch.load(args.output / "last.pt", weights_only=True)
            state = torch.load(args.output / "training.pt", weights_only=True)
            self.assertGreater(state["language_tokens_seen"], 0)
            for key in expected["state_dict"]:
                self.assertTrue(torch.equal(expected["state_dict"][key], actual["state_dict"][key]), key)

    def test_language_guard_rejects_better_dialogue_with_excessive_language_regression(self):
        measure = lambda n: {"assistant_loss_per_token": n, "assistant_target_tokens": 10}
        language = lambda n: {"loss_per_token": n, "bits_per_utf8_byte": n, "predicted_tokens": 10, "utf8_bytes": 20}
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            args = options(self.base, self.path, Path(tmp) / "run", replay_corpus=FIXTURE)
            with mock.patch.object(d, "validation", side_effect=[measure(10), measure(8), measure(7)]), \
                 mock.patch.object(d, "report_validation", side_effect=[language(5), language(5.1), language(6)]):
                d.train(args)
            best = torch.load(args.output / "best.pt", weights_only=True)
            self.assertEqual(best["selected_step"], 3)
            self.assertEqual(best["language_validation"]["loss_per_token"], 5.1)
            metadata = json.loads((args.output / "manifest.json").read_text())
            self.assertFalse(metadata["history"][-1]["language_guard_passed"])

    def test_context_drops_whole_old_turns_but_refuses_overlong_current_message(self):
        turns = [{"role": "user", "content": "Ancien " * 100},
                 {"role": "assistant", "content": "Ancienne réponse."},
                 {"role": "user", "content": "Bonjour."}]
        _, kept = d.prompt_ids(turns, self.codec, 64, 8)
        self.assertEqual(kept, turns[-1:])
        with self.assertRaisesRegex(ValueError, "trop long"):
            d.prompt_ids(turns[:1], self.codec, 64, 8)

    def test_generation_masks_virtual_roles_and_stops_on_eos(self):
        codec = self.codec
        class Model(torch.nn.Module):
            cfg = Config(width=16, heads=2, layers=1, context=128)
            def forward(self, x):
                logits = torch.zeros(1, x.size(1), codec.vocab_size + 2)
                logits[:, :, [codec.bos, codec.vocab_size, codec.vocab_size + 1]] = 1e9
                logits[:, :, codec.eos] = 10
                return logits
        answer, _ = d.reply(Model(), codec, [{"role": "user", "content": "Bonjour."}], 4)
        self.assertEqual(answer, "")
        with self.assertRaisesRegex(ValueError, "suppression silencieuse"):
            d.reply(Model(), codec, [{"role": "user", "content": "X " * 1000}], 4, preserve_history=True)

    def test_evaluation_reports_exact_closed_tasks_and_open_answers_separately(self):
        rows = [data.row("greet", "A", "B", "train", "reference_only"),
                data.row("greet", "C", "D", "validation", "reference_only"),
                data.row("copy", "Copie 7", "7", "validation")]
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp) / "data.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            args = SimpleNamespace(threads=1, seed=42, limit=10, split="validation", checkpoint=self.base,
                                   data=path, max_new_tokens=8, output=Path(tmp) / "evaluation.json")
            def answer(model, codec, turns, *a, **kw):
                return ("7" if turns[-1]["content"] == "Copie 7" else "Autre formulation"), turns
            with mock.patch.object(d, "reply", side_effect=answer):
                d.evaluate(args)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["closed_tasks"], {"correct": 1, "total": 1})
            self.assertIn("human-review", report["open_answers"])
            self.assertEqual(len(report["cases"]), 2)
            with self.assertRaises(FileExistsError):
                d.evaluate(args)

    def test_learning_rate_has_fixed_warmup_and_cosine_end(self):
        self.assertAlmostEqual(d.learning_rate(1, 100, 10, 0.01), 0.001)
        self.assertAlmostEqual(d.learning_rate(10, 100, 10, 0.01), 0.01)
        self.assertAlmostEqual(d.learning_rate(100, 100, 10, 0.01), 0.001)
        self.assertGreater(d.learning_rate(20, 100, 10, 0.01), d.learning_rate(90, 100, 10, 0.01))


class CalculatorTests(unittest.TestCase):
    def test_exact_integer_fraction_and_decimal_results(self):
        self.assertEqual(d.calculate("17 * 23"), "391")
        self.assertEqual(d.calculate("0.1 + 0.2"), "3/10")
        self.assertEqual(d.calculate("1 / 3 + 1 / 6"), "1/2")
        self.assertEqual(d.calculate("-(2 + 3) ** 2"), "-25")
        self.assertEqual(d.calculate("2 ** -3"), "1/8")

    def test_python_code_and_excessive_computation_are_refused(self):
        for expression in ("__import__('os').system('echo nope')", "open('fichier')", "x + 1",
                           "True + 1", "[1,2]", "2**999999", "1e999", "1" * 257):
            with self.subTest(expression=expression), self.assertRaises(ValueError):
                d.calculate(expression)
        with self.assertRaisesRegex(ValueError, "zéro"):
            d.calculate("1 / 0")
