import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from research.kira_model.__main__ import train
from research.kira_model.data import read_corpus
from research.kira_model.prepare_fr import build_rows, chunks, clean_body, prepare, source_bytes


def book(text):
    return ("Metadata and terms\n*** START OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\n"
            + text + "\n*** END OF THE PROJECT GUTENBERG EBOOK EXAMPLE ***\nLicense footer").encode("utf-8")


class FrenchCorpusTests(unittest.TestCase):
    def test_strips_boilerplate_and_rejects_missing_boundaries(self):
        self.assertEqual(clean_body(book("Un récit français.")), "Un récit français.")
        with self.assertRaises(ValueError):
            clean_body(b"missing header")

    def test_chunks_preserve_words_without_overlap(self):
        text = "\n\n".join(" ".join(f"mot{i}-{j}" for j in range(25)) for i in range(15))
        pieces = chunks(text, size=200)
        self.assertGreater(len(pieces), 1)
        self.assertEqual(" ".join(" ".join(pieces).split()), " ".join(text.split()))
        self.assertTrue(all(len(piece) <= 200 for piece in pieces))

    def test_source_split_is_by_book_with_eight_sampled_chunks(self):
        specs = [{"id": n, "author": str(n), "title": str(n), "split": split, "sha256": "fixture"}
                 for n, split in ((1, "train"), (2, "validation"), (3, "test"))]
        raw = {s["id"]: book("\n\n".join(f"livre{s['id']} section{i} " + "texte français " * 100
                                             for i in range(25))) for s in specs}
        rows, metadata = build_rows(raw, sources=specs)
        self.assertEqual(sum(r["split"] == "validation" for r in rows), 8)
        self.assertEqual(sum(r["split"] == "test" for r in rows), 8)
        for row in rows:
            number = int(row["id"].split("-")[1])
            self.assertEqual(row["split"], specs[number - 1]["split"])
        self.assertEqual(metadata[1]["selected_chunks"][0], 0)
        self.assertEqual(metadata[1]["selected_chunks"][-1], metadata[1]["available_chunks"] - 1)

    def test_cached_source_is_verified_without_network(self):
        raw = book("Texte vérifié.")
        spec = {"id": 1, "sha256": hashlib.sha256(raw).hexdigest()}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "1.txt").write_bytes(raw)
            with mock.patch("urllib.request.urlopen", side_effect=AssertionError("Network not allowed")):
                self.assertEqual(source_bytes(spec, path), raw)
            spec["sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "différente"):
                source_bytes(spec, path)
            self.assertEqual((path / "1.txt").read_bytes(), raw)

    def test_download_hash_failure_never_creates_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"unexpected content")):
                with self.assertRaises(ValueError):
                    source_bytes({"id": 1, "sha256": "bad"}, Path(tmp))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_preparation_works_offline_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            def local_source(spec, folder):
                return book(f"Livre {spec['id']} : ceci est un texte original de test, distinct des autres.")
            with mock.patch("research.kira_model.prepare_fr.source_bytes", side_effect=local_source):
                prepare(Path(tmp))
                docs, digest = read_corpus(Path(tmp) / "corpus.jsonl")
                metadata = json.loads((Path(tmp) / "sources.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["corpus_sha256"], digest)
                self.assertEqual({k: len(v) for k, v in docs.items()}, {"train": 2, "validation": 1, "test": 1})
                with self.assertRaises(FileExistsError):
                    prepare(Path(tmp))

    def test_changing_test_text_does_not_change_training_weights(self):
        fixture = Path(__file__).parent.parent / "fixtures" / "smoke.jsonl"
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            outputs = []
            for i in range(2):
                corpus = Path(tmp) / f"corpus-{i}.jsonl"
                extra = {"id": "test-extra", "split": "test", "source": "test-fixture",
                         "license": "project-authored-test", "training_allowed": True,
                         "text": f"Texte de test tenu à l'écart, variation {i}."}
                corpus.write_text(fixture.read_text(encoding="utf-8") + json.dumps(extra) + "\n", encoding="utf-8")
                output = Path(tmp) / f"run-{i}"
                args = SimpleNamespace(corpus=corpus, output=output, steps=3, batch_size=1, threads=1,
                                       lr=0.001, width=16, heads=2, layers=1, context=32, seed=42)
                train(args)
                manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
                self.assertGreater(manifest["seconds_per_step"], 0)
                outputs.append(torch.load(output / "weights.pt", weights_only=True)["state_dict"])
            for key, value in outputs[0].items():
                self.assertTrue(torch.equal(value, outputs[1][key]), key)
