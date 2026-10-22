import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research.kira_model import prepare_aya_fr as aya
from research.kira_model import prepare_language as stories
from research.kira_model.conversation_eval import reserved_prompts
from research.kira_model.dialogue_data import row, normalize, validate


def encode(messages):
    return [0] + [1 for m in messages for _ in m["content"]]


def source(prompt="Une question unique de source.", answer="Une réponse complète.", **extra):
    return {"inputs": prompt, "targets": answer, "language_code": "fra", "annotation_type": "original-annotations"} | extra


class AyaImportTests(unittest.TestCase):
    def test_complete_answers_preserve_source_and_drop_annotator_identifier(self):
        record = source(user_id="private-annotator-column")
        rows, raw, _ = aya.build([(17, record)], encode, 5000)
        real = [r for r in rows if r["id"].startswith("aya-fr:")]
        self.assertEqual(len(real), 1)
        self.assertEqual(real[0]["messages"][-1]["content"], record["targets"])
        self.assertEqual(real[0]["license"], "Apache-2.0")
        self.assertIn("#row=17", real[0]["source"])
        self.assertFalse(real[0]["synthetic"])
        self.assertNotIn("private-annotator-column", raw.decode())

    def test_wrong_language_identity_claim_and_overlong_answer_are_excluded(self):
        records = [(0, source(language_code="eng")),
                   (1, source(answer="Je suis ChatGPT.")),
                   (2, source(answer="Longue réponse. " * 300))]
        rows, _, stats = aya.build(records, encode, 256)
        self.assertFalse(any(r["id"].startswith("aya-fr:") for r in rows))
        self.assertEqual(stats["external_identity_mention"], 1)

    def test_duplicate_questions_do_not_create_different_targets_in_other_splits(self):
        records = [(0, source()), (1, source(prompt="UNE QUESTION UNIQUE DE SOURCE.", answer="Réponse divergente."))]
        rows, _, stats = aya.build(records, encode, 5000)
        self.assertEqual(sum(r["id"].startswith("aya-fr:") for r in rows), 1)
        self.assertEqual(stats["duplicate_input"], 1)

    def test_conflicting_split_from_other_source_is_never_moved_into_training(self):
        first, _, _ = aya.build([(0, source())], encode, 5000)
        original = next(r for r in first if r["id"].startswith("aya-fr:"))
        existing = row("dialogue-fr", source()["inputs"], "Réponse source précédente.",
                       "test" if original["split"] != "test" else "validation", "reference_only")
        existing.update(id="oasst1-fake", synthetic=False, source="test-source", license="Apache-2.0")
        rows, _, stats = aya.build([(0, source())], encode, 5000, [existing])
        self.assertFalse(any(r["id"].startswith("aya-fr:") for r in rows))
        self.assertEqual(next(r for r in rows if r["id"] == "oasst1-fake")["split"], existing["split"])
        self.assertEqual(stats["split_conflict"], 1)

    def test_free_conversation_probes_never_become_training_examples(self):
        rows, _, _ = aya.build([(0, source(prompt="bjr"))], encode, 5000)
        self.assertFalse(any(r["id"].startswith("aya-fr:") for r in rows))
        for r in rows:
            for message in r["messages"]:
                if message["role"] == "user":
                    self.assertNotIn(normalize(message["content"]), reserved_prompts())
        self.assertTrue(any(len(r["messages"]) > 2 for r in rows if r["skill"] == "interaction"))
        validate(rows)

    def test_parquet_dependency_is_lazy_and_existing_output_is_protected(self):
        with mock.patch.dict("sys.modules", {"pyarrow": None, "pyarrow.parquet": None}):
            with self.assertRaisesRegex(ValueError, "PyArrow"):
                list(stories.parquet_records(Path("unused"), ["text"]))
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(aya, "acquire") as acquire:
            with self.assertRaises(FileExistsError):
                aya.prepare(None, Path(temp), Path(temp))
            acquire.assert_not_called()

    def test_generic_source_download_requires_both_size_and_digest(self):
        for payload in (b"too long", b"evil"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                with mock.patch.object(stories.urllib.request, "urlopen", return_value=io.BytesIO(payload)):
                    with self.assertRaises(ValueError):
                        stories.acquire(root, {"source": (4, "not-the-digest")}, "https://example.invalid/")
                self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
