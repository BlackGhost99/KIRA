"""Corpus de préentraînement de KIRA LM : sources épinglées, filtres, splits, tokenizer et jetons.

    python -m research.kira_model.corpus list
    python -m research.kira_model.corpus prepare --raw-dir BRUT --output DONNEES --threads 4

Prévu pour une session Kaggle sur CPU (le quota GPU n'est pas consommé). Chaque fichier source est
téléchargé à une révision fixe et comparé à sa taille et à son SHA-256 publiés. PyArrow ne sert qu'à
lire les fichiers Parquet ; l'entraînement n'utilise que les jetons produits ici.
"""
from __future__ import annotations

import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

from .codec import Codec, normalize_text, train_tokenizer

DATA_FORMAT = "kira-token-shards-1"
SPLIT_RULE = ("sha256(source:document_id) mod 1000 : bucket < p => test, < 2p => validation, sinon train "
              "(p = heldout_permille de la source, 1 par défaut) ; "
              "au-delà du plafond, les documents des buckets validation/test sont écartés, jamais déplacés vers train")
HF = "https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}"

# Registre épinglé, vérifié le 2026-10-23 via l'API Hugging Face (révision, taille, SHA-256 de chaque fichier).
# Fichiers répartis sur chaque jeu (un sur N) pour la diversité ; jetons estimés depuis les métadonnées Parquet.
HQ_PLUS_FILES = [
    ("fra_Latn/000_00000.parquet", 685771316, "d09a4a6c1a9cf7c55f828416c851dafaba159eaaff048e5180f66d569667043a"),
    ("fra_Latn/000_00015.parquet", 654061038, "cd5e0a2b94635c3ed6033977958a07372aa295da6d4e96e20e4795b8686a47e0"),
    ("fra_Latn/000_00030.parquet", 581061732, "8979e8afe802207932a23755115fef176248d801623851cfe9bfdf0bcfac4e9b"),
    ("fra_Latn/000_00045.parquet", 588857011, "5b2d47a5c057403663cceeff0a39a8ac37d4c816340cdc46a1651353c7377013"),
    ("fra_Latn/000_00060.parquet", 571159749, "6ff9a4579ce8f085cc66f34ff7606abd19e8db35ceb4a6998a4c34fcacf922b8"),
    ("fra_Latn/000_00075.parquet", 571391848, "950bf0b73bc0f51beb29d37c6237c9bf543566f8e16991853710d39128b6c325"),
    ("fra_Latn/000_00090.parquet", 548458754, "269a99f225d0dd353dd3e3fb126b1f28e13d22d950493ba52142eb5f7198e789"),
    ("fra_Latn/000_00105.parquet", 546758598, "d94ca51c818aeb939fb3a9ebe9e4e2c2b090f8a73df22db28fcfa84a3a637c34"),
    ("fra_Latn/000_00120.parquet", 112268362, "d63d976e67d02b6e11243a9cb4f23fff960fc26b1ff1a8f518fad32f99d598c3"),
    ("fra_Latn/000_00135.parquet", 388479769, "e507e90b0838e2bd1a0b494d87c29b5baa4f281c884661842096134429de1105"),
    ("fra_Latn/000_00150.parquet", 392894772, "a18d2c498543942e58c8ae0d870872ee0759f5cd1a6ac0283ad6da5806bd1730"),
    ("fra_Latn/000_00165.parquet", 378783764, "79060019832c39d10d6a83e9ebb41925d01b722121262c4083a48ec9ea481bed"),
    ("fra_Latn/000_00180.parquet", 368210018, "dea8a0af37890e90df1c75d11a799eb9d7e301f8d27c772079d3250d249efa9d"),
    ("fra_Latn/000_00195.parquet", 354970795, "a4ded2ab10ffc52d6e43655b391ce1ba2f4a019a4b445456ab51c2684d80dc1f"),
    ("fra_Latn/000_00210.parquet", 342124150, "9ad1b04f4185a268d8b9220adbe021cde41146f3c4edb29f170c3ee4025ba9ec"),
    ("fra_Latn/000_00225.parquet", 327261142, "974a80462d4b92ab55d3f7250b8b0b76cfffa807db0176c9efebccbcafaf9d7e"),
]
PDFS_EDU_FILES = [
    ("data/fra_Latn/train/000_00000.parquet", 875951091, "240c4fbd3151cac02efd0b90b06653bc72fc8aa6253cb8eb1cce2057997681f1"),
    ("data/fra_Latn/train/000_00010.parquet", 847092146, "8f15ef59d86ae2369b41dcb74633900acd9577c38f0b0394e3d6f5bdbb083941"),
    ("data/fra_Latn/train/000_00020.parquet", 833515381, "d52461451850b2517ad2e17a82444510d1c4d761361ab7e4489c3483d4df7688"),
    ("data/fra_Latn/train/000_00030.parquet", 830243024, "50fe89c8f3b4d8eb5d4bdec7749cfa6103d3bfe5d1dc40934cdaecaa257f012c"),
]
WIKI_FILES = [
    ("data/frwiki/000_00000.parquet", 2186642528, "f6aa8b5cbe1bade6ae943e9c060dda580ed6a42d5dc5dbb4fa70bd83b801a513"),
    ("data/frwiki/000_00003.parquet", 2166699854, "f5d53084f8a7cd01c59398d683a2ab88bbb06d7055e75efc0fadbe458f289c5c"),
]
LECARNET_FILES = [
    ("data/train-00000-of-00004.parquet", 213040725, "b3ff2ab7966383ac6f83687ccaeb894033e57048ee6680baf27db234fbd89e6e"),
    ("data/train-00001-of-00004.parquet", 212762348, "82f945f788c81bbc87305026c55428e8bdbf13d0a0f7b3770f767df916c210c6"),
    ("data/train-00002-of-00004.parquet", 213200762, "008e12dc661ca5fc6484884ea907faa48b24bff37eb8f8ed86ec1f83e03b976f"),
    ("data/train-00003-of-00004.parquet", 212992524, "8769227ccd0eaa2bd084117dcb8f22a9ff5804df86b5122e666633daa82ae1dd"),
]


def _files(entries):
    return [{"path": path, "size": size, "sha256": sha} for path, size, sha in entries]


SOURCES = [
    {"name": "fineweb2-hq-plus-fr", "repo": "epfml/FineWeb2-HQ-Plus", "revision": "146690ef88e86e6d728a03ea6f6d85037d8cf9fa",
     "files": _files(HQ_PLUS_FILES), "id_field": "id", "language": "fr", "default_weight": 0.5,
     "license": "ODC-By-1.0", "license_url": "https://opendatacommons.org/licenses/by/1-0/",
     "attribution": "FineWeb2-HQ-Plus (EPFL, epfml), sous-ensemble de FineWeb2 (Hugging Face) ; soumis aux conditions "
                    "d'utilisation de Common Crawl",
     "origin": "pages web Common Crawl 2013-2024, 10 % jugés de meilleure qualité par un classifieur XLM-RoBERTa",
     "training_allowed": True},
    {"name": "finepdfs-edu-fr", "repo": "HuggingFaceFW/finepdfs-edu", "revision": "9cfabe2127faca99b3d5c4dc6d1fcb397399ebde",
     "files": _files(PDFS_EDU_FILES), "id_field": "id", "language": "fr", "default_weight": 0.22,
     "license": "ODC-By-1.0", "license_url": "https://opendatacommons.org/licenses/by/1-0/",
     "attribution": "FinePDFs-Edu (Hugging Face, Kydlíček et al. 2025) ; soumis aux conditions d'utilisation de Common Crawl",
     "origin": "documents PDF de Common Crawl filtrés par un classifieur éducatif", "training_allowed": True},
    {"name": "finewiki-fr", "repo": "HuggingFaceFW/finewiki", "revision": "8bd13e72e6a002407649b3e898535f42ceb1aeb9",
     "files": _files(WIKI_FILES), "id_field": "id", "language": "fr", "default_weight": 0.18,
     "license": "CC-BY-SA-4.0", "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
     "attribution": "Texte de Wikipédia en français, contributeurs de Wikipédia ; adaptation FineWiki (Hugging Face, "
                    "G. Penedo 2025) sous CC BY-SA 4.0",
     "origin": "articles de Wikipédia nettoyés (sans références ni liens externes)", "training_allowed": True},
    {"name": "lecarnet", "repo": "MaxLSB/LeCarnet", "revision": "d0eaae9f8f7f777fde86f16c0b607da1335d2700",
     "files": _files(LECARNET_FILES), "language": "fr", "default_weight": 0.10,
     "license": "ODC-By-1.0", "license_url": "https://opendatacommons.org/licenses/by/1-0/",
     "attribution": "LeCarnet — MaxLSB et jddqd ; histoires générées avec Mistral-Large-Instruct-2411",
     "origin": "histoires courtes synthétiques à vocabulaire simple", "training_allowed": True},
]


def source_index(sources=None):
    return {s["name"]: s for s in (SOURCES if sources is None else sources)}


def digest_file(path):
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def url_for(source, path):
    if source.get("base_url"):
        return source["base_url"].rstrip("/") + "/" + path
    return HF.format(repo=source["repo"], revision=source["revision"], path=path)


def acquire(source, entry, raw_dir, attempts=3):
    """Télécharge un fichier une seule fois ; un écart de taille ou d'empreinte est refusé."""
    path = raw_dir / source["name"] / entry["path"].replace("/", "__")
    expected = (entry["size"], entry["sha256"])
    if path.exists():
        if digest_file(path) != expected:
            raise ValueError(f"Fichier en cache différent de la source épinglée : {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url_for(source, entry["path"]), timeout=60) as response, open(temporary, "wb") as out:
                written = 0
                while chunk := response.read(4 * 1024 * 1024):
                    written += len(chunk)
                    if written > entry["size"]:
                        raise ValueError(f"Source plus grande que sa taille publiée : {entry['path']}")
                    out.write(chunk)
            if digest_file(temporary) != expected:
                raise ValueError(f"Empreinte différente de la source épinglée : {entry['path']}")
            temporary.replace(path)
            return path
        except (OSError, TimeoutError) as exc:
            temporary.unlink(missing_ok=True)
            if attempt == attempts:
                raise OSError(f"Téléchargement impossible après {attempts} essais : {entry['path']} ({exc})") from exc
            time.sleep(5 * attempt)
        except ValueError:
            temporary.unlink(missing_ok=True)
            raise


def read_rows(source, path):
    """(identifiant, texte) dans l'ordre du fichier."""
    text_field, id_field = source.get("text_field", "text"), source.get("id_field")
    if source.get("format", "parquet") == "jsonl":
        with open(path, encoding="utf-8") as stream:
            for index, line in enumerate(stream):
                if line.strip():
                    row = json.loads(line)
                    yield str(row[id_field]) if id_field else str(index), row.get(text_field)
        return
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("La lecture Parquet demande PyArrow (préinstallé sur Kaggle)") from exc
    columns = [text_field] + ([id_field] if id_field else [])
    index = 0
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=1024, columns=columns, use_threads=False):
            texts = batch.column(text_field).to_pylist()
            ids = batch.column(id_field).to_pylist() if id_field else [None] * len(texts)
            for doc_id, text in zip(ids, texts):
                # Sans identifiant publié : fichier + ligne, unique d'un fichier à l'autre.
                yield str(doc_id) if doc_id is not None else f"{Path(path).name}#{index}", text
                index += 1


def split_of(source_name, doc_id, permille=1):
    bucket = int(hashlib.sha256(f"{source_name}:{doc_id}".encode()).hexdigest()[:8], 16) % 1000
    return "test" if bucket < permille else "validation" if bucket < 2 * permille else "train"


def clean(text, source, stats):
    """Filtres légers et déclarés : les corpus retenus sont déjà filtrés par leurs éditeurs."""
    if not isinstance(text, str):
        stats["invalid_text"] += 1
        return None
    text = normalize_text(text).strip()
    if "�" in text:
        stats["replacement_character"] += 1
        return None
    if len(text) < source.get("min_chars", 200):
        stats["too_short"] += 1
        return None
    return text


def documents(source, raw_dir, seen, stats, deadline=None, delete=False):
    """Documents nettoyés et dédoublonnés (texte normalisé exact, toutes sources confondues)."""
    for entry in source["files"]:
        if deadline and time.monotonic() > deadline:
            stats["stopped_by_time_limit"] += 1
            return
        path = acquire(source, entry, raw_dir)
        for doc_id, text in read_rows(source, path):
            stats["read"] += 1
            text = clean(text, source, stats)
            if text is None:
                continue
            key = int.from_bytes(hashlib.sha256(" ".join(text.casefold().split()).encode()).digest()[:8], "little")
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            yield doc_id, text
        if delete:
            path.unlink(missing_ok=True)


class ShardWriter:
    def __init__(self, root, name, split, shard_tokens):
        self.root, self.name, self.split, self.limit = root, name, split, shard_tokens
        self.buffer, self.shards, self.documents, self.utf8_bytes = array("H"), [], 0, 0
        (root / name).mkdir(parents=True, exist_ok=True)

    @property
    def tokens(self):
        return sum(s["tokens"] for s in self.shards) + len(self.buffer)

    def add(self, ids, text_bytes):
        self.buffer.extend(ids)
        self.documents += 1
        self.utf8_bytes += text_bytes
        if len(self.buffer) >= self.limit:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        relative = f"{self.name}/{self.split}-{len(self.shards):05d}.bin"
        data = self.buffer
        if sys.byteorder != "little":
            data = array("H", data)
            data.byteswap()
        raw = data.tobytes()
        temporary = self.root / (relative + ".part")
        temporary.write_bytes(raw)
        temporary.replace(self.root / relative)
        self.shards.append({"file": relative, "tokens": len(self.buffer), "sha256": hashlib.sha256(raw).hexdigest()})
        self.buffer = array("H")

    def summary(self):
        self.flush()
        return {"documents": self.documents, "tokens": self.tokens, "utf8_bytes": self.utf8_bytes, "shards": self.shards}


def tokenizer_sample(sources, raw_dir, budget_chars, used):
    """Textes du split train seulement, au prorata des poids par défaut, pour apprendre le vocabulaire."""
    total_weight = sum(s["default_weight"] for s in sources) or 1
    for source in sources:
        quota = int(budget_chars * source["default_weight"] / total_weight)
        taken, count, stats = 0, 0, Counter()
        for doc_id, text in documents(source, raw_dir, set(), stats):
            if taken >= quota:
                break
            if split_of(source["name"], doc_id, source.get("heldout_permille", 1)) != "train":
                continue
            taken += len(text)
            count += 1
            yield text
        used[source["name"]] = {"documents": count, "characters": taken}


def prepare(output, raw_dir, sources=None, vocab_size=32000, tokenizer_chars=400_000_000, threads=4,
            heldout_tokens=2_000_000, shard_tokens=200_000_000, time_limit=0.0, budgets=None):
    started = time.monotonic()
    deadline = started + time_limit if time_limit else None
    sources = [dict(s) for s in (SOURCES if sources is None else sources)]
    if not sources:
        raise ValueError("Aucune source de corpus sélectionnée")
    for source in sources:
        if budgets and source["name"] in budgets:
            source["max_train_tokens"] = budgets[source["name"]]
        if not source.get("files") or not source.get("license") or source.get("training_allowed") is not True:
            raise ValueError(f"Source {source['name']} : fichiers, licence et autorisation explicite requis")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Dossier de données non vide : choisir un nouveau dossier")
    output.mkdir(parents=True, exist_ok=True)
    raw_dir = Path(raw_dir)
    os.environ["RAYON_NUM_THREADS"] = str(max(1, threads))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    tokenizer_used = {}
    serialized = train_tokenizer(tokenizer_sample(sources, raw_dir, tokenizer_chars, tokenizer_used), vocab_size)
    codec = Codec(serialized)
    (output / "tokenizer.json").write_text(codec.serialized, encoding="utf-8")
    print(json.dumps({"tokenizer": codec.sha256, "text_vocab": codec.text_size, "model_vocab": codec.vocab_size,
                      "trained_on": tokenizer_used, "seconds": round(time.monotonic() - started)}), flush=True)

    seen, manifest_sources = set(), {}
    for source in sources:
        stats = Counter()
        writers = {split: ShardWriter(output, source["name"], split, shard_tokens)
                   for split in ("train", "validation", "test")}
        limits = {"train": source.get("max_train_tokens"), "validation": heldout_tokens, "test": heldout_tokens}
        # Textes encodés par lots ; « upper » borne leurs jetons (au plus un par octet, plus bos/eos).
        pending, upper = [], Counter()

        def drain():
            for (split, text), ids in zip(pending, codec.encode_batch([t for _, t in pending])):
                writers[split].add([codec.bos, *ids, codec.eos], len(text.encode("utf-8")))
            pending.clear()
            upper.clear()

        for doc_id, text in documents(source, raw_dir, seen, stats, deadline, source.get("delete_raw", False)):
            split = split_of(source["name"], doc_id, source.get("heldout_permille", 1))
            limit = limits[split]
            if limit and writers[split].tokens + upper[split] >= limit:
                drain()  # compte exact avant de décider : un plafond n'est jamais dépassé de plus d'un document
            if limit and writers[split].tokens >= limit:
                if split == "train":
                    stats["train_budget_reached"] += 1
                    break
                stats[f"{split}_beyond_cap_discarded"] += 1
                continue
            size = len(text.encode("utf-8"))
            pending.append((split, text))
            upper[split] += size + 2
            if sum(upper.values()) >= 32_000_000:
                drain()
            if deadline and time.monotonic() > deadline:
                stats["stopped_by_time_limit"] += 1
                break
        drain()
        splits = {split: writer.summary() for split, writer in writers.items()}
        if not splits["train"]["tokens"] or not splits["validation"]["tokens"]:
            raise ValueError(f"Source {source['name']} : pas assez de documents pour train et validation")
        keys = ("repo", "revision", "license", "license_url", "attribution", "origin", "default_weight",
                "language", "min_chars", "max_train_tokens", "heldout_permille")
        manifest_sources[source["name"]] = {
            **{k: source[k] for k in keys if k in source},
            "files": [{"path": f["path"], "size": f["size"], "sha256": f["sha256"]} for f in source["files"]],
            "counts": dict(stats), "splits": splits}
        print(json.dumps({"source": source["name"], "tokens": {s: v["tokens"] for s, v in splits.items()},
                          "counts": dict(stats), "seconds": round(time.monotonic() - started)}), flush=True)

    from .gpu_train import code_sha  # empreinte du code qui a produit les jetons
    import tokenizers
    manifest = {"format": DATA_FORMAT, "dtype": "uint16", "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "tokenizer": {"file": "tokenizer.json", "sha256": codec.sha256, "text_vocab": codec.text_size,
                              "model_vocab": codec.vocab_size, "trained_on": tokenizer_used,
                              "trained_on_splits": ["train"]},
                "split_rule": SPLIT_RULE, "heldout_tokens_cap": heldout_tokens,
                "document_format": "bos + texte NFC + eos, documents concaténés",
                "filters": "texte non vide, sans caractère de remplacement, au moins min_chars caractères ; "
                           "doublons exacts (casse et espaces normalisés) retirés toutes sources confondues",
                "time_limit_reached": bool(deadline and time.monotonic() > deadline),
                "tool": {"code_sha256": code_sha(), "corpus_sha256": hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
                         "tokenizers": tokenizers.__version__, "python": sys.version.split()[0]},
                "sources": manifest_sources,
                "scope": "pretraining-text-not-dialogue-not-verified-facts"}
    temporary = output / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest


def parse_budgets(text):
    if not text:
        return None
    result = {}
    for part in text.split(","):
        name, _, value = part.partition("=")
        result[name.strip()] = int(float(value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list")
    p = commands.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--raw-dir", type=Path, required=True)
    p.add_argument("--sources", help="Noms séparés par des virgules (défaut : toutes les sources du registre)")
    p.add_argument("--budget", help="source=jetons_train,source=jetons_train")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--tokenizer-chars", type=int, default=400_000_000)
    p.add_argument("--heldout-tokens", type=int, default=2_000_000)
    p.add_argument("--shard-tokens", type=int, default=200_000_000)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--time-limit", type=float, default=0.0)
    p.add_argument("--delete-raw", action="store_true", help="Effacer chaque fichier source une fois converti")
    for command in (listing, p):
        command.add_argument("--registry", type=Path, help="Fichier JSON de sources remplaçant le registre intégré")
    args = parser.parse_args()
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8")) if args.registry else SOURCES
        if args.command == "list":
            for source in registry:
                print(json.dumps({k: source.get(k) for k in ("name", "repo", "revision", "license", "origin",
                                                              "default_weight")}, ensure_ascii=False))
            return
        index = source_index(registry)
        names = args.sources.split(",") if args.sources else list(index)
        unknown = [n for n in names if n not in index]
        if unknown:
            raise ValueError("Sources inconnues : " + ", ".join(unknown))
        chosen = [dict(index[n], delete_raw=args.delete_raw) for n in names]
        manifest = prepare(args.output, args.raw_dir, chosen, args.vocab_size, args.tokenizer_chars, args.threads,
                           args.heldout_tokens, args.shard_tokens, args.time_limit, parse_budgets(args.budget))
        print(json.dumps({"output": str(args.output), "sources": {n: {s: v["tokens"] for s, v in m["splits"].items()}
                                                                   for n, m in manifest["sources"].items()}}))
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + "\n")


if __name__ == "__main__":
    main()
