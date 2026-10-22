from copy import deepcopy
import gzip
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from research.kira_model import prepare_oasst_fr as p


def node(identifier, text, assistant=False, children=None):
    labels = {key: {"count": 3, "value": 0.0}
              for key in ("spam", "lang_mismatch", "pii", "not_appropriate", "fails_task")}
    labels["quality"] = {"count": 3, "value": 0.9}
    return {"message_id": identifier, "text": text, "role": "assistant" if assistant else "prompter",
            "lang": "fr", "deleted": False, "synthetic": False, "review_result": True,
            "review_count": 3, "rank": 0 if assistant else None, "labels": labels,
            "replies": children or []}


def tree(root, identifier="t"):
    return {"message_tree_id": identifier, "tree_state": "ready_for_export", "prompt": root}


def encode(messages):
    return [0] + [1 for t in messages for _ in t["content"]]


class OasstImportTests(unittest.TestCase):
    def test_reviewed_complete_paths_keep_provenance_and_root_group(self):
        leaf = node("a2", "Réponse finale.", True)
        q = node("u2", "Autre question unique.", children=[leaf])
        a = node("a1", "Première réponse.", True, [q])
        root = node("u1", "Question de départ unique.", children=[a])
        rows, stats = p.extract([tree(root)], encode, 10000)
        real = [r for r in rows if not r["synthetic"]]
        self.assertEqual(len(real), 2)
        self.assertEqual(real[0]["group"], real[1]["group"])
        self.assertEqual(real[0]["split"], real[1]["split"])
        self.assertEqual(real[1]["source_message_ids"], ["u1", "a1", "u2", "a2"])
        self.assertEqual(real[1]["messages"][-1]["content"], "Réponse finale.")
        self.assertEqual(real[1]["license"], "Apache-2.0")

    def test_duplicate_roots_across_trees_share_split_even_when_replies_differ(self):
        a = node("a1", "Première réponse.", True, [node("u2", "Suite A.", children=[node("a2", "Fin A.", True)])])
        b = node("b1", "Autre première réponse.", True, [node("v2", "Suite B.", children=[node("b2", "Fin B.", True)])])
        rows, _ = p.extract([tree(node("u1", "Racine commune.", children=[a]), "t1"),
                             tree(node("v1", "  RACINE   COMMUNE. ", children=[b]), "t2")], encode, 10000)
        real = [r for r in rows if not r["synthetic"]]
        self.assertEqual(len(real), 3)  # La première entrée dupliquée est supprimée.
        self.assertEqual(len({r["group"] for r in real}), 1)
        self.assertEqual(len({r["split"] for r in real}), 1)

    def test_rejected_ancestor_cannot_reenter_through_descendant(self):
        bad = node("a", "Mauvaise branche.", True, [node("u2", "Suite.", children=[node("a2", "Fin.", True)])])
        bad["labels"]["pii"]["value"] = 0.3
        rows, _ = p.extract([tree(node("u", "Question isolée.", children=[bad]))], encode, 10000)
        self.assertFalse(any(not r["synthetic"] for r in rows))

    def test_language_review_rank_quality_and_missing_labels_fail_closed(self):
        good = node("a", "Réponse.", True)
        variants = [{"lang": "en"}, {"review_result": False}, {"synthetic": True}, {"deleted": True},
                    {"rank": 1}, {"review_count": 0}, {"labels": {}}]
        for variant in variants:
            with self.subTest(variant=variant):
                self.assertFalse(p.acceptable(good | variant, "assistant"))
        for value in (0.49, float("nan"), None):
            bad = deepcopy(good)
            bad["labels"]["quality"]["value"] = value
            self.assertFalse(p.acceptable(bad, "assistant"))

    def test_overlong_answer_is_excluded_in_full_and_probe_questions_are_reserved(self):
        for question, answer in (("Question unique.", "Longue réponse " * 100), ("bjr", "Bonjour.")):
            with self.subTest(question=question):
                rows, _ = p.extract([tree(node("u", question, children=[node("a", answer, True)]))], encode, 256)
                self.assertFalse(any(not r["synthetic"] for r in rows))

    def test_duplicate_inputs_are_removed_and_exact_authored_questions_do_not_leak(self):
        a = tree(node("u", "Question répétée unique.", children=[node("a", "Réponse A.", True)]))
        b = deepcopy(a)
        b["prompt"]["replies"][0]["text"] = "Autre réponse."
        authored = tree(node("u3", "Qui es-tu ?", children=[node("a3", "Un autre modèle.", True)]))
        rows, stats = p.extract([a, b, authored], encode, 10000)
        self.assertEqual(sum(not r["synthetic"] for r in rows), 1)
        self.assertEqual(stats["duplicate_paths"], 1)

    def test_decompression_and_download_bounds_and_existing_files_are_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "small.gz"
            with gzip.open(archive, "wb") as f:
                f.write(b"x" * 101)
            with mock.patch.object(p, "MAX_LINE", 100), self.assertRaisesRegex(ValueError, "limites"):
                list(p.read_trees(archive))
            with mock.patch.object(p, "MAX_UNCOMPRESSED", 100), self.assertRaisesRegex(ValueError, "limites"):
                list(p.read_trees(archive))
            output = root / "corpus.jsonl"
            output.with_suffix(".manifest.json").write_text("conserver", encoding="utf-8")
            with mock.patch.object(p, "acquire") as acquire, self.assertRaises(FileExistsError):
                p.prepare(None, root, output)
            acquire.assert_not_called()
            self.assertFalse(output.exists())
            wrong = root / "source"
            wrong.write_bytes(b"bad")
            with mock.patch.object(p, "FILES", {"source": (4, hashlib.sha256(b"good").hexdigest())}), self.assertRaisesRegex(ValueError, "Cache"):
                p.acquire(root)
            self.assertEqual(wrong.read_bytes(), b"bad")

    def test_oversized_download_is_removed_and_never_promoted_to_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with mock.patch.object(p, "FILES", {"source": (4, hashlib.sha256(b"good").hexdigest())}), \
                 mock.patch.object(p.urllib.request, "urlopen", return_value=io.BytesIO(b"oversize")), \
                 self.assertRaisesRegex(ValueError, "taille"):
                p.acquire(root)
            self.assertEqual(list(root.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
