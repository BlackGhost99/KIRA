"""Génère les notebooks Kaggle de KIRA LM dans research/kira_model/notebooks/.

    python -m research.kira_model.build_notebooks           # réécrit les notebooks
    python -m research.kira_model.build_notebooks --check   # échoue s'ils ne sont pas à jour

Les notebooks récupèrent le code public du dépôt à un commit consigné : ils ne contiennent ni code
recopié, ni secret, ni clé. Ils se lancent sur Kaggle avec « Save Version → Save & Run All ».
Un notebook Kaggle ne peut pas relire sa propre sortie : le préentraînement alterne donc entre deux
notebooks identiques (2A et 2B), chacun reprenant la sortie de l'autre.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / "notebooks"
REPOSITORY = "https://github.com/BlackGhost99/KIRA"

FIND = '''# Retrouve les entrées ajoutées au notebook (« Add Input »), en lecture seule sous /kaggle/input.
import json, os, shutil, subprocess, sys
from pathlib import Path

def find_data():
    found = []
    for manifest in Path("/kaggle/input").rglob("manifest.json"):
        try:
            if json.loads(manifest.read_text(encoding="utf-8")).get("format") == "kira-token-shards-1":
                found.append(manifest.parent)
        except (OSError, ValueError):
            pass
    if len(found) != 1:
        raise SystemExit("Il faut exactement une entrée de données KIRA (sortie du notebook kira-1-donnees). "
                         f"Trouvé : {[str(f) for f in found]}")
    return found[0]

def find_run(name):
    """Session la plus avancée d'un run parmi les entrées : (dossier, étape), ou (None, None)."""
    best, best_step = None, None
    for state in Path("/kaggle/input").rglob(f"runs/{name}/training.pt"):
        manifest = state.parent / "manifest.json"
        step = json.loads(manifest.read_text(encoding="utf-8")).get("steps", 0) if manifest.exists() else 0
        if best_step is None or step > best_step:
            best, best_step = state.parent, step
    return best, best_step
'''

BOOTSTRAP = '''# Récupère le code public de KIRA au commit choisi et installe le tokenizer épinglé.
def run(*command, **options):
    print("$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(c) for c in command], check=True, **options)

CODE = Path("/tmp/KIRA")
if not (CODE / ".git").exists():
    run("git", "clone", "--quiet", "{repository}", CODE)
run("git", "-C", CODE, "fetch", "--quiet", "origin")
run("git", "-C", CODE, "checkout", "--quiet", "--detach", KIRA_REF if len(KIRA_REF) == 40 else "origin/" + KIRA_REF)
COMMIT = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
os.environ.update(KIRA_CODE_COMMIT=COMMIT, PYTHONPATH=str(CODE), PYTHONUNBUFFERED="1")
run(sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "tokenizers==0.23.3")
import torch
image = Path("/etc/git_commit")
print("Code KIRA :", COMMIT, "| PyTorch", torch.__version__, "| GPU :", torch.cuda.device_count(),
      "| image Kaggle :", image.read_text().strip() if image.exists() else "?")
'''.replace("{repository}", REPOSITORY)

STREAM = '''# Lance une commande et garde aussi sa sortie dans un fichier journal.
def stream(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("$", " ".join(map(str, command)), flush=True)
    with open(log_path, "a", encoding="utf-8") as log:
        process = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code:
        raise SystemExit(f"Échec (code {code}) : voir {log_path}. Les sauvegardes déjà écrites restent dans /kaggle/working.")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


def notebook(cells, gpu):
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "kaggle": {"accelerator": "nvidiaTeslaT4" if gpu else "none", "isGpuEnabled": gpu,
                                    "isInternetEnabled": True, "language": "python", "sourceType": "notebook"}}}


DATA = notebook([
    md('''
# KIRA — 1. Préparer les données (CPU, sans GPU)

Ce notebook télécharge des textes français **aux révisions épinglées**, vérifie leur taille et leur empreinte
SHA-256, apprend le vocabulaire de KIRA sur le split d'entraînement seulement, puis écrit les fichiers de jetons.
Il ne consomme **pas** le quota GPU.

**Réglages Kaggle** (panneau de droite) : *Accelerator* = **None** ; *Internet* = **On** ;
*Environment* = **Pin to original environment**.
Puis **Save Version → Save & Run All (Commit)**. Tu peux fermer la page : le travail continue.

Sortie : le dossier `donnees/` (jetons, tokenizer, `manifest.json` avec licences et attributions).
Les sources et leurs licences sont décrites dans `docs/kira-kaggle.md` du dépôt.
'''),
    code('''
# Paramètres
KIRA_REF = "main"          # branche ou commit (40 caractères) du dépôt public
VOCAB_SIZE = 32000         # vocabulaire texte appris (les rôles s'ajoutent hors du texte)
SOURCES = ""               # vide = toutes les sources du registre ; sinon "nom1,nom2"
BUDGET = ""                # facultatif : "source=jetons,source=jetons" pour plafonner l'entraînement
TIME_LIMIT_HOURS = 10.5    # arrêt propre avant la limite de 12 h de Kaggle
'''),
    code(FIND),
    code(BOOTSTRAP),
    code(STREAM),
    code('''
# Les fichiers bruts vont dans /kaggle/working (Kaggle peut arrêter une session qui écrit trop ailleurs)
# et sont effacés dès qu'ils sont convertis ; seul donnees/ reste dans la sortie.
OUTPUT, RAW = Path("/kaggle/working/donnees"), Path("/kaggle/working/brut")
command = [sys.executable, "-m", "research.kira_model.corpus", "prepare", "--output", OUTPUT, "--raw-dir", RAW,
           "--vocab-size", VOCAB_SIZE, "--threads", os.cpu_count() or 4,
           "--time-limit", int(TIME_LIMIT_HOURS * 3600), "--delete-raw"]
if SOURCES:
    command += ["--sources", SOURCES]
if BUDGET:
    command += ["--budget", BUDGET]
try:
    stream(command, Path("/kaggle/working/journal-donnees.txt"))
finally:
    shutil.rmtree(RAW, ignore_errors=True)
'''),
    code('''
# Résumé : jetons par source et par split, licences.
manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
total = 0
for name, source in manifest["sources"].items():
    tokens = {split: part["tokens"] for split, part in source["splits"].items()}
    total += tokens["train"]
    print(f"{name:26} train {tokens['train']/1e9:6.2f} G  validation {tokens['validation']/1e6:5.2f} M  "
          f"test {tokens['test']/1e6:5.2f} M  | {source.get('license')}")
print(f"Total entraînement : {total/1e9:.2f} milliard(s) de jetons ; arrêt par limite de temps : {manifest['time_limit_reached']}")
run("du", "-sh", OUTPUT)
'''),
], gpu=False)


def pretrain(letter, other):
    return notebook([
        md(f'''
# KIRA — 2{letter}. Préentraînement (GPU T4 ×2)

KIRA LM apprend le français à partir de **poids initialisés de zéro**. Une session dure au plus ~11 h et s'arrête
d'elle-même ; la suivante reprend exactement à la même étape.

Kaggle ne laisse pas un notebook relire sa propre sortie : on **alterne** entre deux notebooks identiques,
**2A** et **2B**. Chacun prend en entrée la sortie de l'autre : A → B → A → B…

**Réglages Kaggle** : *Accelerator* = **GPU T4 x2** ; *Internet* = **On** ; *Environment* = **Pin to original environment**.
**Add Input** : la sortie de **kira-1-donnees**, et la sortie de **kira-2{other}** (sauf pour la toute première session).
Avant de lancer, vérifie dans *Input* que l'entrée kira-2{other} est à sa **dernière version**.
Puis **Save Version → Save & Run All (Commit)**.

Première session du run : `NEW_RUN = True` (puis remets `False`). Pour une simple mesure de vitesse, utilise
un autre `RUN_NAME` avec `MEASURE_STEPS = 30`.
Fin du parcours : fixe `DECAY_START` (l'étape actuelle) et `DECAY_STEPS` (par exemple 15 % des étapes faites) ;
le taux d'apprentissage descend alors jusqu'à zéro et le run se termine.
'''),
        code('''
# Paramètres
RUN_NAME = "kira-100m-a"   # même nom pour toutes les sessions d'un même run
NEW_RUN = False            # True seulement pour la toute première session de ce run
RESUME_STEP = None         # facultatif : étape attendue (affichée à la fin de la session précédente)
PRESET = "kira-100m"       # kira-30m, kira-100m ou kira-125m-deep
BATCH_TOKENS = 524288      # jetons par mise à jour (identique d'une session à l'autre)
MICRO_BATCH = 8            # séquences par GPU et par passe ; 4 si la mémoire manque, 16 si elle suffit
LR = 0.0015               # fixé pour tout le run (float16 sur T4 : prudence)
WARMUP = 500
EVAL_EVERY = 500           # mesure de validation et textes générés toutes les N étapes
MIX = ""                   # vide = poids du manifeste ; sinon "source=poids,source=poids"
TIME_LIMIT_HOURS = 11.0    # durée d'entraînement de cette session
MEASURE_STEPS = None       # ex. 30 pour une simple mesure de vitesse
DECAY_START = None         # ex. 9000 : début de la décroissance finale
DECAY_STEPS = None         # ex. 1500
KIRA_REF = None            # None = même commit que la session précédente (ou "main" pour un nouveau run)
ALLOW_CODE_CHANGE = False  # True seulement après avoir lu ce qui a changé dans le code
NCCL_P2P_DISABLE = False   # True seulement si l'entraînement à deux GPU reste bloqué au démarrage
'''),
        code(FIND),
        code('''
DATA = find_data()
PREVIOUS, PREVIOUS_STEP = find_run(RUN_NAME)
if PREVIOUS is None and not NEW_RUN:
    raise SystemExit(f"Aucune session précédente de {RUN_NAME} dans les entrées. Première session : NEW_RUN = True. "
                     "Sinon, ajouter en entrée la sortie de l'autre notebook (2A ↔ 2B) à sa dernière version.")
if PREVIOUS is not None and NEW_RUN:
    raise SystemExit(f"NEW_RUN = True alors que {RUN_NAME} existe déjà (étape {PREVIOUS_STEP}) : remettre NEW_RUN = False "
                     "pour continuer, ou choisir un autre RUN_NAME.")
if RESUME_STEP is not None and PREVIOUS_STEP != RESUME_STEP:
    raise SystemExit(f"Étape {PREVIOUS_STEP} trouvée au lieu de {RESUME_STEP} : mettre l'entrée à sa dernière version.")
if KIRA_REF is None:
    KIRA_REF = "main"
    if PREVIOUS is not None:
        KIRA_REF = json.loads((PREVIOUS / "manifest.json").read_text(encoding="utf-8"))["sessions"][-1].get("code_commit") or "main"
print("Données :", DATA)
print("Reprise :", f"{PREVIOUS} (étape {PREVIOUS_STEP})" if PREVIOUS else "aucune — nouveau run, poids aléatoires")
'''),
        code(BOOTSTRAP),
        code(STREAM),
        code('''
GPUS = torch.cuda.device_count()
if not GPUS:
    raise SystemExit("Aucun GPU visible : choisir Accelerator = GPU T4 x2 dans les réglages du notebook.")
run("nvidia-smi")
OUTPUT = Path("/kaggle/working/runs") / RUN_NAME
arguments = ["pretrain", "--data", DATA, "--output", OUTPUT, "--preset", PRESET, "--batch-tokens", BATCH_TOKENS,
             "--micro-batch", MICRO_BATCH, "--lr", LR, "--warmup", WARMUP, "--eval-every", EVAL_EVERY,
             "--time-limit", int(TIME_LIMIT_HOURS * 3600)]
if PREVIOUS is not None:
    arguments += ["--resume-from", PREVIOUS]
if MIX:
    arguments += ["--mix", MIX]
if MEASURE_STEPS:
    arguments += ["--session-steps", MEASURE_STEPS]
if DECAY_START is not None:
    arguments += ["--decay-start", DECAY_START, "--decay-steps", DECAY_STEPS]
if ALLOW_CODE_CHANGE:
    arguments += ["--allow-code-change"]
if NCCL_P2P_DISABLE:
    os.environ["NCCL_P2P_DISABLE"] = "1"
launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", GPUS]
            if GPUS > 1 else [sys.executable])
# Journal à part : le dossier du run ne doit contenir que ce que l'entraîneur écrit.
stream([*launcher, "-m", "research.kira_model.gpu_train", *arguments], Path("/kaggle/working/journaux") / f"{RUN_NAME}.txt")
'''),
        code('''
# Résumé de la session : vitesse, pertes de validation et textes générés (à relire : ce n'est pas une note de qualité).
manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
session = manifest["sessions"][-1]
speeds = [e["tokens_per_second"] for e in manifest["history"] if "tokens_per_second" in e][-20:]
print(f"Étape {manifest['steps']} | {manifest['tokens_seen']/1e9:.3f} G jetons vus | arrêt : {session['stop_reason']}")
print(f"➜ Prochaine session : l'autre notebook, avec RESUME_STEP = {manifest['steps']}")
if speeds:
    rate = sum(speeds) / len(speeds)
    print(f"Vitesse : {rate:,.0f} jetons/s, soit ~{rate * 3600 / 1e9:.2f} G jetons par heure de GPU")
for name, value in manifest["last_eval"]["language"].items():
    print(f"Validation {name:24} perte {value['loss_per_token']:.4f}")
for sample in manifest["samples"][-1]["samples"] or []:
    print("\\n«", sample["prompt"], "» →", sample["sampled"][:300])
'''),
    ], gpu=True)


DIALOGUE = notebook([
    md('''
# KIRA — 3. Dialogue (GPU)

Adapte les poids préentraînés de KIRA au dialogue, puis rejoue les **30 tours de sondes** de conversation
(dont « bsr ») et prépare un fichier allégé à télécharger pour le PC.

**Réglages Kaggle** : *Accelerator* = **GPU T4 x2** ; *Internet* = **On** (sources de dialogue épinglées) ;
*Environment* = **Pin to original environment**.
**Add Input** : la sortie de **kira-1-donnees**, et celles de **kira-2A** et **kira-2B** (la plus avancée est choisie).
Puis **Save Version → Save & Run All (Commit)**.

Les réponses des sondes sont à **relire** : une bonne perte ne prouve pas un bon dialogue.
'''),
    code('''
# Paramètres
BASE_RUN = "kira-100m-a"          # run de préentraînement à adapter
RUN_NAME = "kira-100m-a-dialogue"
STEPS = 3000
BATCH_SIZE = 64                   # conversations par mise à jour
MICRO_BATCH = 16
LR = 0.0003
WARMUP = 100                      # étapes de montée du taux d'apprentissage (inférieur à STEPS)
REPLAY_RATIO = 0.2                # part des étapes qui répètent le préentraînement
MAX_LANGUAGE_REGRESSION = 0.1     # hausse tolérée de la perte de langue pour retenir des poids
TIME_LIMIT_HOURS = 10.0
KIRA_REF = "main"
'''),
    code(FIND),
    code(BOOTSTRAP),
    code(STREAM),
    code('''
DATA = find_data()
BASE, BASE_STEP = find_run(BASE_RUN)
if BASE is None or not (BASE / "weights.pt").exists():
    raise SystemExit(f"Poids {BASE_RUN}/weights.pt introuvables : ajouter les sorties de kira-2A et kira-2B en entrée.")
print("Poids de départ :", BASE, "étape", BASE_STEP)
CONTEXT = json.loads((BASE / "manifest.json").read_text(encoding="utf-8"))["config"]["context"]
CONVERSATIONS = Path("/kaggle/working/conversations/conversations.jsonl")
if not CONVERSATIONS.exists():
    stream([sys.executable, "-m", "research.kira_model.sft_data", "prepare", "--tokenizer", DATA / "tokenizer.json",
            "--context", CONTEXT, "--raw-dir", Path("/kaggle/working/brut-dialogue"), "--output", CONVERSATIONS],
           Path("/kaggle/working/conversations/journal.txt"))
    shutil.rmtree("/kaggle/working/brut-dialogue", ignore_errors=True)
print(json.loads(CONVERSATIONS.with_suffix(".manifest.json").read_text(encoding="utf-8"))["examples"])
'''),
    code('''
GPUS = torch.cuda.device_count()
if not GPUS:
    raise SystemExit("Aucun GPU visible : choisir un accélérateur GPU dans les réglages du notebook.")
if BATCH_SIZE % (MICRO_BATCH * GPUS):
    GPUS = 1  # le lot doit se répartir exactement entre les GPU
OUTPUT = Path("/kaggle/working/runs") / RUN_NAME
arguments = ["sft", "--base", BASE / "weights.pt", "--conversations", CONVERSATIONS, "--replay-data", DATA,
             "--output", OUTPUT, "--steps", STEPS, "--batch-size", BATCH_SIZE, "--micro-batch", MICRO_BATCH,
             "--lr", LR, "--warmup", WARMUP, "--replay-ratio", REPLAY_RATIO, "--max-language-regression", MAX_LANGUAGE_REGRESSION,
             "--time-limit", int(TIME_LIMIT_HOURS * 3600)]
launcher = ([sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", GPUS]
            if GPUS > 1 else [sys.executable])
# Journal à part : le dossier du run ne doit contenir que ce que l'entraîneur écrit.
stream([*launcher, "-m", "research.kira_model.gpu_train", *arguments], Path("/kaggle/working/journaux") / f"{RUN_NAME}.txt")
'''),
    code('''
# Sondes de conversation (décodage glouton, comme au laboratoire) et fichier allégé à télécharger.
REPORTS = Path("/kaggle/working/rapports")
report = REPORTS / f"{RUN_NAME}-sondes.json"
if not report.exists():
    stream([sys.executable, "-m", "research.kira_model.lm_chat", "probe", "--weights", OUTPUT / "weights.pt",
            "--output", report, "--max-new-tokens", 64], REPORTS / "journal.txt")
light = Path("/kaggle/working") / f"{RUN_NAME}-inference.pt"
if not light.exists():
    stream([sys.executable, "-m", "research.kira_model.lm_chat", "export", "--weights", OUTPUT / "weights.pt",
            "--output", light], REPORTS / "journal.txt")
for case in json.loads(report.read_text(encoding="utf-8"))["cases"]:
    for turn in case["turns"]:
        print(f"Toi > {turn['user']}\\nKIRA > {turn['response']}\\n")
'''),
], gpu=True)

NOTEBOOKS = {"kira-1-donnees.ipynb": DATA, "kira-2A-preentrainement.ipynb": pretrain("A", "B"),
             "kira-2B-preentrainement.ipynb": pretrain("B", "A"), "kira-3-dialogue.ipynb": DIALOGUE}


def render(value):
    return json.dumps(value, ensure_ascii=False, indent=1) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    stale = []
    for name, value in NOTEBOOKS.items():
        path = ROOT / name
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != render(value):
                stale.append(name)
        else:
            ROOT.mkdir(parents=True, exist_ok=True)
            path.write_text(render(value), encoding="utf-8", newline="\n")
    extra = sorted(p.name for p in ROOT.glob("*.ipynb") if p.name not in NOTEBOOKS) if ROOT.exists() else []
    if args.check and extra:
        stale += [f"{name} (inconnu du générateur)" for name in extra]
    if stale:
        sys.exit("Notebooks à régénérer : " + ", ".join(stale))


if __name__ == "__main__":
    main()
