import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest import mock

import torch

from research.kira_model import corpus, gpu_train, lm_chat
from research.kira_model.tests.test_kira_corpus import write_source

TINY = ["--preset", "test", "--micro-batch", "4", "--lr", "0.01", "--warmup", "3", "--eval-every", "6",
        "--eval-tokens", "1024", "--device", "cpu", "--threads", "1", "--log-every", "3", "--checkpoint-minutes", "0"]


def pretrain_args(data, output, *extra):
    return gpu_train.build_parser().parse_args(["pretrain", "--data", str(data), "--output", str(output),
                                                "--batch-tokens", "512", *TINY, *extra])


def sft_args(base, conversations, output, *extra):
    return gpu_train.build_parser().parse_args(
        ["sft", "--base", str(base), "--conversations", str(conversations), "--output", str(output),
         "--steps", "8", "--batch-size", "8", "--micro-batch", "4", "--warmup", "2", "--eval-every", "4",
         "--eval-tokens", "1024", "--device", "cpu", "--threads", "1", "--log-every", "2",
         "--checkpoint-minutes", "0", "--lr", "0.003", *extra])


def quiet(function, *args):
    with mock.patch("builtins.print"):
        return function(*args)


def conversation_rows(extra=()):
    pairs = [("Bonjour KIRA, tu vas bien ?", "Bonjour ! Je suis disponible. Que veux-tu apprendre ?"),
             ("Qui es-tu ?", "Je suis KIRA, un modèle expérimental."),
             ("Explique la force.", "Une force change le mouvement d'un objet."),
             ("Peux-tu m'aider en physique ?", "Oui. Quel sujet veux-tu voir ?")]
    rows = []
    for i in range(24):
        question, answer = pairs[i % len(pairs)]
        rows.append({"id": f"c{i}", "group": f"g{i}", "split": ("validation", "test", "train", "train")[i % 4],
                     "source": "unit-test", "license": "test-only", "training_allowed": True,
                     "messages": [{"role": "user", "content": f"{question} ({i})"},
                                  {"role": "assistant", "content": answer}]})
    return rows + list(extra)


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return path


class GPUTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        (cls.root / "src").mkdir()
        sources = [write_source(cls.root / "src", "alpha", 500, 1), write_source(cls.root / "src", "beta", 300, 2)]
        with mock.patch("builtins.print"):
            corpus.prepare(cls.root / "data", cls.root / "raw", sources, vocab_size=600, tokenizer_chars=150_000,
                           threads=1, heldout_tokens=3000, shard_tokens=20_000)
        quiet(gpu_train.train, pretrain_args(cls.root / "data", cls.root / "base", "--max-steps", "12"), "pretrain")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pretraining_learns_records_provenance_and_exports_loadable_weights(self):
        manifest = json.loads((self.root / "base" / "manifest.json").read_text(encoding="utf-8"))
        first = manifest["initial_eval"]["language"]["alpha"]["loss_per_token"]
        last = manifest["last_eval"]["language"]["alpha"]["loss_per_token"]
        self.assertLess(last, first)
        self.assertEqual(manifest["status"], "experimental-not-a-general-assistant")
        self.assertFalse(manifest["test_split_evaluated"])
        self.assertEqual(manifest["provenance"]["pretraining"]["alpha"]["license"], "test-only")
        self.assertEqual(manifest["sessions"][0]["stop_reason"], "planned_end")
        self.assertTrue(manifest["samples"][-1]["samples"])
        model, codec, saved = lm_chat.load(self.root / "base" / "weights.pt")
        self.assertEqual(saved["step"], 12)
        answer, history = lm_chat.reply(model, codec, [{"role": "user", "content": "Bonjour"}], 8)
        self.assertIsInstance(answer, str)

    def test_resume_from_a_read_only_previous_session_is_exact(self):
        straight, first, second = (self.root / name for name in ("straight", "first", "second"))
        quiet(gpu_train.train, pretrain_args(self.root / "data", straight, "--max-steps", "10"), "pretrain")
        quiet(gpu_train.train, pretrain_args(self.root / "data", first, "--max-steps", "10", "--session-steps", "4"), "pretrain")
        quiet(gpu_train.train, pretrain_args(self.root / "data", second, "--max-steps", "10", "--resume-from", str(first)), "pretrain")
        a = torch.load(straight / "weights.pt", weights_only=True)["state_dict"]
        b = torch.load(second / "weights.pt", weights_only=True)["state_dict"]
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))
        sessions = torch.load(second / "training.pt", weights_only=True)["sessions"]
        self.assertEqual([(s["start_step"], s["end_step"]) for s in sessions], [(0, 4), (4, 10)])

    def test_resume_refuses_other_options_shapes_and_unannounced_code_changes(self):
        first = self.root / "refusals"
        quiet(gpu_train.train, pretrain_args(self.root / "data", first, "--max-steps", "6", "--session-steps", "2"), "pretrain")
        for extra in (["--lr", "0.02"], ["--width", "64"], ["--mix", "alpha=1"], ["--batch-tokens", "1024"]):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                quiet(gpu_train.train, pretrain_args(self.root / "data", self.root / "other", "--max-steps", "6",
                                                     "--resume-from", str(first), *extra), "pretrain")
        with mock.patch.object(gpu_train, "code_sha", return_value="0" * 64):
            with self.assertRaises(ValueError):
                quiet(gpu_train.train, pretrain_args(self.root / "data", self.root / "other", "--max-steps", "6",
                                                     "--resume-from", str(first)), "pretrain")
            quiet(gpu_train.train, pretrain_args(self.root / "data", self.root / "accepted", "--max-steps", "6",
                                                 "--resume-from", str(first), "--allow-code-change"), "pretrain")
        sessions = torch.load(self.root / "accepted" / "training.pt", weights_only=True)["sessions"]
        self.assertTrue(sessions[-1]["code_change_accepted"])

    def test_decay_plan_can_be_set_late_but_never_changed_once_started(self):
        run = self.root / "decay"
        quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--session-steps", "4"), "pretrain")
        with self.assertRaises(ValueError):
            quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--decay-start", "3", "--decay-steps", "4"), "pretrain")
        quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--decay-start", "4", "--decay-steps", "4",
                                             "--session-steps", "2"), "pretrain")
        with self.assertRaises(ValueError):
            quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--decay-start", "5", "--decay-steps", "4"), "pretrain")
        result = quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--decay-start", "4", "--decay-steps", "4"), "pretrain")
        self.assertEqual((result["step"], result["stop_reason"]), (8, "planned_end"))
        history = torch.load(run / "training.pt", weights_only=True)["history"]
        rates = {e["step"]: e["lr"] for e in history if "lr" in e}
        self.assertAlmostEqual(rates[3], 0.01)
        self.assertAlmostEqual(rates[6], 0.01 * (1 - (2 / 4) ** 0.5))

    def test_schedule_shapes(self):
        plan = {"start": 10, "steps": 10, "shape": "linear", "min_ratio": 0.1}
        self.assertEqual(gpu_train.learning_rate(5, 1.0, 10, plan), 0.5)
        self.assertEqual(gpu_train.learning_rate(10, 1.0, 10, plan), 1.0)
        self.assertAlmostEqual(gpu_train.learning_rate(15, 1.0, 10, plan), 0.55)
        self.assertAlmostEqual(gpu_train.learning_rate(20, 1.0, 10, plan), 0.1)
        self.assertEqual(gpu_train.learning_rate(500, 1.0, 10, None), 1.0)

    def test_time_limit_stops_cleanly_with_a_checkpoint(self):
        run = self.root / "time"
        result = quiet(gpu_train.train, pretrain_args(self.root / "data", run, "--time-limit", "1", "--final-reserve", "0"), "pretrain")
        self.assertEqual(result["stop_reason"], "time_limit")
        self.assertTrue((run / "training.pt").exists() and (run / "weights.pt").exists())

    def test_pretraining_needs_an_end_and_never_reuses_a_foreign_folder(self):
        with self.assertRaises(ValueError):
            quiet(gpu_train.train, pretrain_args(self.root / "data", self.root / "endless"), "pretrain")
        foreign = self.root / "foreign"
        foreign.mkdir()
        (foreign / "notes.txt").write_text("autre chose", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            quiet(gpu_train.train, pretrain_args(self.root / "data", foreign, "--max-steps", "2"), "pretrain")

    @unittest.skipUnless(sys.platform.startswith("linux") and torch.distributed.is_available(),
                         "Gloo multi-processus vérifié sous Linux seulement")
    def test_two_processes_give_the_weights_of_one_process(self):
        one, two = self.root / "one", self.root / "two"
        extra = ("--max-steps", "6", "--z-loss", "0")
        quiet(gpu_train.train, pretrain_args(self.root / "data", one, *extra), "pretrain")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        torch.multiprocessing.spawn(_worker, args=(str(self.root / "data"), str(two), port, list(extra)), nprocs=2, join=True)
        a = torch.load(one / "weights.pt", weights_only=True)["state_dict"]
        b = torch.load(two / "weights.pt", weights_only=True)["state_dict"]
        self.assertLess(max((a[k] - b[k]).abs().max().item() for k in a), 1e-6)
        self.assertEqual(torch.load(two / "training.pt", weights_only=True)["sessions"][-1]["world_size"], 2)

    def test_dialogue_selects_by_validation_under_the_language_guard(self):
        conversations = write_rows(self.root / "conversations.jsonl", conversation_rows())
        base = self.root / "base" / "weights.pt"
        strict, loose = self.root / "sft-strict", self.root / "sft-loose"
        quiet(gpu_train.train, sft_args(base, conversations, loose, "--replay-data", str(self.root / "data"),
                                        "--max-language-regression", "100"), "sft")
        quiet(gpu_train.train, sft_args(base, conversations, strict, "--replay-data", str(self.root / "data"),
                                        "--max-language-regression", "0", "--replay-ratio", "0"), "sft")
        loose_weights = torch.load(loose / "weights.pt", weights_only=True)
        self.assertGreater(loose_weights["selected_step"], 0)
        self.assertTrue((loose / "last.pt").exists())
        manifest = json.loads((loose / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["provenance"]["conversations"]["examples"]["test"], 6)
        self.assertTrue(manifest["samples"][-1]["samples"][0]["turns"])
        strict_manifest = json.loads((strict / "manifest.json").read_text(encoding="utf-8"))
        guarded = [e for e in strict_manifest["history"] if "selected" in e]
        self.assertTrue(guarded)
        self.assertTrue(all(e["language_guard_passed"] or not e["selected"] for e in guarded))
        strict_weights = torch.load(strict / "weights.pt", weights_only=True)
        if strict_weights["selected_step"] == 0:
            base_state = torch.load(base, weights_only=True)["state_dict"]
            self.assertTrue(all(torch.equal(base_state[k], strict_weights["state_dict"][k]) for k in base_state))

    def test_conversation_files_are_checked_before_training(self):
        from research.kira_model.codec import Codec
        codec = Codec((self.root / "data" / "tokenizer.json").read_text(encoding="utf-8"))
        base = conversation_rows()
        bad = {
            "réservée": base + [dict(base[2], id="x", group="x", messages=[{"role": "user", "content": "bsr"},
                                                                          {"role": "assistant", "content": "Bonsoir !"}])],
            "groupe": base + [dict(base[2], id="y", group=base[0]["group"])],
            "fin": base + [dict(base[2], id="z", group="z", messages=[{"role": "user", "content": "Salut"}])],
            "long": base + [dict(base[2], id="w", group="w", messages=[{"role": "user", "content": "mot " * 200},
                                                                      {"role": "assistant", "content": "ok"}])],
        }
        for name, rows in bad.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                gpu_train.load_conversations(write_rows(self.root / f"bad-{name}.jsonl", rows), codec, 64)
        encoded, _, counts, _ = gpu_train.load_conversations(write_rows(self.root / "good.jsonl", base), codec, 64)
        self.assertNotIn("test", encoded)
        self.assertEqual(counts["test"], 6)
        x, y = gpu_train.conversation_batch(encoded["train"][:2], codec.pad)
        learned = [t for row in y.tolist() for t in row if t != -100]
        self.assertIn(codec.eos, learned)
        self.assertNotIn(codec.special["user"], learned)
        self.assertNotIn(codec.special["assistant"], learned)


def _worker(rank, data, output, port, extra):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), WORLD_SIZE="2", RANK=str(rank), LOCAL_RANK=str(rank))
    torch.set_num_threads(1)
    quiet(gpu_train.train, pretrain_args(data, output, *extra), "pretrain")


if __name__ == "__main__":
    unittest.main()
