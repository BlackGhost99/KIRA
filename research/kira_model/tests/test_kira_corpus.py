import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

from research.kira_model import corpus
from research.kira_model.codec import Codec

WORDS = ("le la les un une des chat maison soleil physique énergie force masse vitesse apprendre comprendre "
         "étudier bonjour merci forêt rivière école élève professeur livre carnet mesure").split()


def write_source(directory, name, count, seed, extra=()):
    rng = random.Random(seed)
    rows = [{"id": f"{name}-{i}", "text": " ".join(rng.choice(WORDS) for _ in range(rng.randint(40, 90))).capitalize() + "."}
            for i in range(count)]
    rows += [{"id": f"{name}-extra-{i}", "text": text} for i, text in enumerate(extra)]
    path = Path(directory) / f"{name}.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    data = path.read_bytes()
    return {"name": name, "base_url": Path(directory).as_uri(), "format": "jsonl", "id_field": "id",
            "files": [{"path": path.name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
            "license": "test-only", "training_allowed": True, "default_weight": 0.5, "min_chars": 50,
            "heldout_permille": 60, "origin": "synthetic unit-test text"}


class RegistryTests(unittest.TestCase):
    def test_every_pinned_source_is_complete_and_attributed(self):
        names = set()
        for source in corpus.SOURCES:
            self.assertNotIn(source["name"], names)
            names.add(source["name"])
            self.assertRegex(source["revision"], r"^[0-9a-f]{40}$")
            self.assertIs(source["training_allowed"], True)
            for key in ("repo", "license", "license_url", "attribution", "origin", "language"):
                self.assertTrue(source.get(key), (source["name"], key))
            self.assertTrue(source["files"])
            for entry in source["files"]:
                self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
                self.assertGreater(entry["size"], 0)
                self.assertTrue(entry["path"].endswith(".parquet"))
            self.assertTrue(corpus.url_for(source, source["files"][0]["path"]).startswith(
                f"https://huggingface.co/datasets/{source['repo']}/resolve/{source['revision']}/"))
        self.assertAlmostEqual(sum(s["default_weight"] for s in corpus.SOURCES), 1.0)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        duplicate = "Ce paragraphe identique apparaît dans les deux sources et ne doit être gardé qu'une fois."
        self.sources = [write_source(self.root / "src", "alpha", 400, 1, [duplicate, "court"]),
                        write_source(self.root / "src", "beta", 300, 2, [duplicate.upper()])]

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, name="data", **changes):
        options = dict(vocab_size=600, tokenizer_chars=150_000, threads=1, heldout_tokens=4000, shard_tokens=20_000)
        with mock.patch("builtins.print"):
            return corpus.prepare(self.root / name, self.root / "raw", self.sources, **(options | changes))

    def test_preparation_is_deterministic_and_verifiable(self):
        first, second = self.prepare("a"), self.prepare("b")
        for name in ("alpha", "beta"):
            self.assertEqual(first["sources"][name]["splits"], second["sources"][name]["splits"])
        self.assertEqual(first["tokenizer"]["sha256"], second["tokenizer"]["sha256"])
        codec = Codec((self.root / "a" / "tokenizer.json").read_text(encoding="utf-8"))
        for name, source in first["sources"].items():
            for split, part in source["splits"].items():
                for shard in part["shards"]:
                    raw = (self.root / "a" / shard["file"]).read_bytes()
                    self.assertEqual(hashlib.sha256(raw).hexdigest(), shard["sha256"])
                    self.assertEqual(len(raw), 2 * shard["tokens"])
                    first_ids = [int.from_bytes(raw[i:i + 2], "little") for i in range(0, 8, 2)]
                    self.assertEqual(first_ids[0], codec.bos)

    def test_heldout_documents_never_train_the_tokenizer(self):
        used = {}
        texts = list(corpus.tokenizer_sample(self.sources, self.root / "raw", 10**9, used))
        heldout = set()
        for source in self.sources:
            for line in (self.root / "src" / f"{source['name']}.jsonl").read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if corpus.split_of(source["name"], row["id"], 60) != "train":
                    heldout.add(row["text"])
        self.assertTrue(heldout)
        self.assertFalse(heldout & set(texts))
        self.assertEqual(sum(v["documents"] for v in used.values()), len(texts))

    def test_duplicates_short_texts_and_split_caps_are_counted(self):
        manifest = self.prepare(heldout_tokens=300)
        alpha, beta = manifest["sources"]["alpha"]["counts"], manifest["sources"]["beta"]["counts"]
        self.assertEqual(alpha["too_short"], 1)
        self.assertEqual(beta["duplicate"], 1)  # même texte à la casse près, déjà vu dans alpha
        discarded = sum(c.get("validation_beyond_cap_discarded", 0) + c.get("test_beyond_cap_discarded", 0)
                        for c in (alpha, beta))
        self.assertGreater(discarded, 0)

    def test_train_budget_and_file_integrity(self):
        manifest = self.prepare(budgets={"alpha": 2000})
        self.assertLess(manifest["sources"]["alpha"]["splits"]["train"]["tokens"], 2000 + 200)
        self.sources[0]["files"][0]["sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            self.prepare("altered")

    def test_cached_file_is_checked_and_output_never_overwritten(self):
        self.prepare()
        cached = next((self.root / "raw" / "alpha").iterdir())
        cached.write_bytes(cached.read_bytes() + b"x")
        with self.assertRaises(ValueError):
            self.prepare("other")
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_sources_require_license_and_explicit_permission(self):
        self.sources[1]["training_allowed"] = False
        with self.assertRaises(ValueError):
            self.prepare()

    def test_no_network_is_used_for_local_sources(self):
        real = corpus.urllib.request.urlopen

        def local_only(url, *args, **kwargs):
            if not str(url).startswith("file:"):
                raise AssertionError("Accès réseau inattendu")
            return real(url, *args, **kwargs)
        with mock.patch.object(corpus.urllib.request, "urlopen", side_effect=local_only):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
