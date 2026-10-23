import ast
from pathlib import Path
import re
import tempfile
import unittest

from research.kira_model import build_notebooks, sft_data
from research.kira_model.codec import Codec, train_tokenizer
from research.kira_model.gpu_train import load_conversations


class ConversationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = [t for row in sft_data.authored() for m in row["messages"] for t in [m["content"]]]
        cls.codec = Codec(train_tokenizer(texts * 3, 700))

    def test_authored_examples_avoid_every_probe_even_without_punctuation(self):
        reserved = sft_data.reserved_keys()
        self.assertIn(sft_data.probe_key("Bsr !"), reserved)
        self.assertEqual(sft_data.probe_key("Comment t'appelles-tu ?"), sft_data.probe_key("Comment t'appelles tu ?"))
        kept, stats = sft_data.build(sft_data.authored() + sft_data.lab_references(), self.codec, 1024)
        for row in kept:
            for turn in row["messages"]:
                if turn["role"] == "user":
                    self.assertNotIn(sft_data.probe_key(turn["content"]), reserved)
        self.assertGreaterEqual(stats["probe_question_excluded"], 1)
        self.assertGreater(len(kept), 200)

    def test_splits_follow_the_first_question_and_case_variants_are_kept(self):
        kept, _ = sft_data.build(sft_data.authored(), self.codec, 1024)
        firsts = [r["messages"][0]["content"] for r in kept]
        self.assertIn("Bonjour KIRA", firsts)
        self.assertIn("bonjour kira", firsts)
        by_group = {}
        for row in kept:
            self.assertEqual(by_group.setdefault(row["group"], row["split"]), row["split"])
        self.assertEqual({r["split"] for r in kept}, {"train", "validation", "test"})

    def test_too_long_conversations_are_counted_not_cut(self):
        kept, stats = sft_data.build(sft_data.authored(), self.codec, 24)
        self.assertGreater(stats["over_context"], 0)
        self.assertTrue(all(len(self.codec.conversation(r["messages"])[0]) <= 25 for r in kept))

    def test_prepared_file_is_accepted_by_the_trainer(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "tokenizer.json").write_text(self.codec.serialized, encoding="utf-8")
            manifest = sft_data.prepare(folder / "tokenizer.json", 512, folder / "raw", folder / "conv.jsonl",
                                        include_external=False)
            encoded, digest, counts, sources = load_conversations(folder / "conv.jsonl", self.codec, 512)
            self.assertEqual(digest, manifest["sha256"])
            self.assertEqual(counts, manifest["examples"])
            self.assertTrue(all(s["license"] for s in sources))
            with self.assertRaises(FileExistsError):
                sft_data.prepare(folder / "tokenizer.json", 512, folder / "raw", folder / "conv.jsonl", include_external=False)


class NotebookTests(unittest.TestCase):
    def test_committed_notebooks_match_the_generator(self):
        for name, value in build_notebooks.NOTEBOOKS.items():
            path = build_notebooks.ROOT / name
            self.assertTrue(path.exists(), name)
            self.assertEqual(path.read_text(encoding="utf-8"), build_notebooks.render(value),
                             f"{name} : lancer python -m research.kira_model.build_notebooks")

    def test_notebooks_are_valid_python_without_secrets(self):
        for name, value in build_notebooks.NOTEBOOKS.items():
            source = "\n".join("".join(cell["source"]) for cell in value["cells"] if cell["cell_type"] == "code")
            ast.parse(source)
            self.assertNotRegex(source, re.compile(r"(api[_-]?key|token\s*=\s*['\"]\w|password|ghp_|hf_[A-Za-z0-9]{8})", re.I))
            self.assertIn(build_notebooks.REPOSITORY, source)
            self.assertEqual(value["nbformat"], 4)


if __name__ == "__main__":
    unittest.main()
