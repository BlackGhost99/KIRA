"""KIRA Seed Dialogue : adapter nos poids BPE, mesurer, puis converser sur CPU."""
from __future__ import annotations

import argparse
import ast
import copy
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
import tokenizers

from .bpe_experiment import BPEModel, TextCodec, FORMAT as BASE_FORMAT, atomic_save, recode, report_validation
from .__main__ import batch as language_batch
from .data import read_corpus
from .dialogue_data import load, normalize, prepare
from .model import Config

FORMAT = "kira-seed-dialogue-0.3"
ROLES = "virtual-user-assistant-v1"


def calculate(expression):
    """Outil exact et borné, distinct du modèle. Aucun eval, appel ou accès fichier."""
    if not expression.strip() or len(expression) > 256:
        raise ValueError("Expression de calcul vide ou trop longue")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError("Expression arithmétique invalide") from exc
    if sum(1 for _ in ast.walk(tree)) > 64:
        raise ValueError("Expression de calcul trop complexe")

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            if not math.isfinite(node.value) or abs(node.value) > 10**12:
                raise ValueError("Nombre hors des limites de cet outil")
            value = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand) * (-1 if isinstance(node.op, ast.USub) else 1)
        elif isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                value = a + b
            elif isinstance(node.op, ast.Sub):
                value = a - b
            elif isinstance(node.op, ast.Mult):
                value = a * b
            elif isinstance(node.op, ast.Div):
                value = a / b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and abs(b) <= 12:
                value = a ** int(b)
            else:
                raise ValueError("Opération non prise en charge ou puissance trop grande")
        else:
            raise ValueError("Seuls les nombres et les opérations + - * / ** sont autorisés")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096:
            raise ValueError("Résultat de calcul trop grand")
        return value
    try:
        return str(visit(tree.body))
    except ZeroDivisionError as exc:
        raise ValueError("Division par zéro") from exc


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_sha():
    digest = hashlib.sha256()
    root = Path(__file__).parent
    for name in ("dialogue.py", "dialogue_data.py", "bpe_experiment.py", "model.py", "data.py", "__main__.py"):
        digest.update(name.encode())
        digest.update((root / name).read_bytes())
    return digest.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def load_model(path):
    """Conserve toutes les lignes anciennes ; seuls deux rôles commencent de zéro."""
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") not in (BASE_FORMAT, FORMAT):
        raise ValueError("Poids incompatibles : checkpoint KIRA BPE ou Dialogue requis")
    codec = TextCodec(saved["tokenizer_json"])
    model = BPEModel(Config(**saved["config"]), codec.vocab_size + 2)
    state = saved["state_dict"]
    if saved["format"] == BASE_FORMAT:
        initialized = model.state_dict()
        state = {key: torch.cat((value, initialized[key][codec.vocab_size:]), dim=0)
                 if key in ("token.weight", "head.weight") else value
                 for key, value in state.items()}
    elif saved.get("roles") != ROLES:
        raise ValueError("Convention de rôles incompatible")
    model.load_state_dict(state, strict=True)
    return model, codec


def encode_messages(messages, codec):
    """Les rôles sont hors du vocabulaire texte : aucune chaîne ne peut les injecter."""
    ids, loss_mask = [codec.bos], [False]
    for i, turn in enumerate(messages):
        if turn["role"] != ("user" if i % 2 == 0 else "assistant"):
            raise ValueError("Alternance utilisateur/assistant invalide")
        assistant = turn["role"] == "assistant"
        ids.append(codec.vocab_size + int(assistant))
        loss_mask.append(False)
        text = codec.backend.encode(turn["content"], add_special_tokens=False).ids
        ids.extend(text)
        loss_mask.extend([assistant] * len(text))
        if assistant:
            ids.append(codec.eos)
            loss_mask.append(True)
    return ids, loss_mask


def encode_rows(rows, codec, context):
    encoded = []
    for item in rows:
        ids, mask = encode_messages(item["messages"], codec)
        if len(ids) > context + 1:
            raise ValueError(f"Exemple {item['id']} : {len(ids) - 1} tokens, contexte {context}. Raccourcir cet exemple.")
        encoded.append({"id": item["id"], "skill": item["skill"], "ids": ids, "mask": mask})
    return encoded


def make_batch(examples):
    # Padding à la longueur réellement nécessaire, crucial pour le petit CPU.
    length = max(len(item["ids"]) - 1 for item in examples)
    xs, ys = [], []
    for item in examples:
        ids, mask = item["ids"], item["mask"]
        targets = [token if learn else -100 for token, learn in zip(ids[1:], mask[1:])]
        padding = length - len(targets)
        xs.append(ids[:-1] + [0] * padding)
        ys.append(targets + [-100] * padding)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


@torch.no_grad()
def validation(model, examples):
    model.eval()
    total, count = 0.0, 0
    for offset in range(0, len(examples), 8):
        x, y = make_batch(examples[offset:offset + 8])
        total += F.cross_entropy(model(x).flatten(0, 1), y.flatten(), reduction="sum").item()
        count += int((y != -100).sum())
    return {"assistant_loss_per_token": total / count, "assistant_target_tokens": count}


def learning_rate(step, steps, warmup, peak):
    if step <= warmup:
        return peak * step / max(1, warmup)
    phase = (step - warmup) / max(1, steps - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * phase)))


def export(model, codec, state, signature, step, validation_loss, language_validation=None):
    return {"format": FORMAT, "kind": "weights", "roles": ROLES,
            "config": asdict(model.cfg), "tokenizer_json": codec.serialized,
            "state_dict": state, "training_signature": signature,
            "selected_step": step, "assistant_validation_loss": validation_loss,
            "language_validation": language_validation}


def train(args):
    if min(args.steps, args.batch_size, args.threads, args.eval_every, args.checkpoint_every) < 1:
        raise ValueError("Les nombres d'étapes, lots, threads et intervalles doivent être positifs")
    if not 0 <= args.warmup < args.steps or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("Warmup ou taux d'apprentissage invalide")
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        raise ValueError("stop-after doit être compris entre 1 et steps")
    replay_path = getattr(args, "replay_corpus", None)
    replay_ratio = getattr(args, "replay_ratio", 0.25) if replay_path else 0.0
    max_regression = getattr(args, "max_language_regression", 0.35)
    if not math.isfinite(replay_ratio) or not 0 <= replay_ratio < 1:
        raise ValueError("replay-ratio doit être compris entre 0 inclus et 1 exclu")
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max-language-regression doit être positif ou nul")
    groups, data_hash = load(args.data)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    saved = None
    if args.resume:
        saved = torch.load(args.output / "training.pt", map_location="cpu", weights_only=True)
        if saved.get("format") != FORMAT or saved.get("kind") != "training":
            raise ValueError("Reprise Dialogue incompatible")
        model, codec = load_model(args.output / "training.pt")
        base_hash = saved["signature"]["base_sha256"]
        if args.base is not None and sha_file(args.base) != base_hash:
            raise ValueError("Reprise incompatible : poids de départ différents")
    else:
        if args.output.exists():
            raise FileExistsError("Dossier déjà présent : utiliser --resume ou un nouveau dossier")
        if args.base is None:
            raise ValueError("--base doit désigner tes poids KIRA BPE")
        base_hash = sha_file(args.base)
        model, codec = load_model(args.base)
    replay, replay_hash = None, None
    if replay_path:
        original, replay_hash = read_corpus(replay_path)
        replay = recode({key: original[key] for key in ("train", "validation")}, codec)
    signature = {"base_sha256": base_hash, "data_sha256": data_hash,
                 "tokenizer_sha256": hashlib.sha256(codec.serialized.encode()).hexdigest(),
                 "config": asdict(model.cfg), "roles": ROLES, "code_sha256": code_sha(),
                 "steps": args.steps, "batch_size": args.batch_size, "threads": args.threads,
                 "seed": args.seed, "lr": args.lr, "warmup": args.warmup, "eval_every": args.eval_every,
                 "replay_sha256": replay_hash, "replay_ratio": replay_ratio,
                 "max_language_regression": max_regression,
                 "torch": str(torch.__version__), "tokenizers": str(tokenizers.__version__)}
    if saved and saved["signature"] != signature:
        raise ValueError("Reprise incompatible : données, code, bibliothèques ou options différents")
    # Les textes de test ne sont pas encodés et ne choisissent jamais les poids.
    examples = encode_rows(groups["train"], codec, model.cfg.context)
    heldout = encode_rows(groups["validation"], codec, model.cfg.context)
    by_skill = {}
    for item in examples:
        by_skill.setdefault(item["skill"], []).append(item)
    skills = sorted(by_skill)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if saved:
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        start, seen = saved["step"], saved["assistant_tokens_seen"]
        language_seen = saved["language_tokens_seen"]
        initial, best_loss, best_step = saved["initial"], saved["best_loss"], saved["best_step"]
        best_state, history = saved["best_state_dict"], saved["history"]
        language_initial, language_final = saved["language_initial"], saved["language_final"]
        best_language = saved["best_language_validation"]
    else:
        initial = validation(model, heldout)
        language_initial = report_validation(model, replay["validation"], model.cfg.context) if replay else None
        language_final = language_initial
        best_language = language_initial
        best_loss, best_step = initial["assistant_loss_per_token"], 0
        best_state = copy.deepcopy(model.state_dict())
        start, seen, language_seen, history = 0, 0, 0, []
    finish = args.stop_after or args.steps
    if finish <= start:
        raise ValueError("Le segment doit dépasser l'étape enregistrée ; conserver steps et retirer stop-after")
    if not saved:
        args.output.mkdir(parents=True, exist_ok=False)

    def save_training(step):
        checkpoint = {**export(model, codec, model.state_dict(), signature, step, None),
                      "kind": "training", "signature": signature, "step": step,
                      "optimizer": optimizer.state_dict(), "python_rng": rng.getstate(),
                      "torch_rng": torch.get_rng_state(), "assistant_tokens_seen": seen,
                      "language_tokens_seen": language_seen,
                      "initial": initial, "best_loss": best_loss, "best_step": best_step,
                      "language_initial": language_initial, "language_final": language_final,
                      "best_language_validation": best_language,
                      "best_state_dict": best_state, "history": history}
        atomic_save(checkpoint, args.output / "training.pt")

    # Reconstruire aussi best.pt après reprise : jamais dépendant d'un export externe.
    atomic_save(export(model, codec, best_state, signature, best_step, best_loss, best_language), args.output / "best.pt")
    if not saved:
        save_training(0)
    print(json.dumps({"model": FORMAT, "parameters": sum(p.numel() for p in model.parameters()),
                      "base_sha256": base_hash, "resume_step": start, "planned_steps": args.steps,
                      "examples": {s: len(v) for s, v in groups.items()},
                      "assistant_only_dialogue_loss": True, "language_replay_ratio": replay_ratio}), flush=True)
    began = time.perf_counter()
    final = None
    for step in range(start + 1, finish + 1):
        replay_step = bool(replay and rng.random() < replay_ratio)
        if replay_step:
            windows = []
            for _ in range(args.batch_size):
                ids = rng.choice(replay["train"])["tokens"]
                offset = rng.randrange(max(1, len(ids) - model.cfg.context))
                windows.append(ids[offset:offset + model.cfg.context + 1])
            x, y = language_batch(windows, model.cfg.context)
        else:
            sample = [rng.choice(by_skill[rng.choice(skills)]) for _ in range(args.batch_size)]
            x, y = make_batch(sample)
        lr = learning_rate(step, args.steps, args.warmup, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        if not torch.isfinite(loss):
            raise RuntimeError("Perte non finie : arrêt de l'entraînement")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if not replay_step:
            seen += int((y != -100).sum())
        else:
            language_seen += int((y != -100).sum())
        if step == 1 or step % 50 == 0 or step == finish:
            entry = {"step": step, "train_loss": loss.item(), "lr": lr,
                     "objective": "language_replay" if replay_step else "assistant_response"}
            history.append(entry)
            print(json.dumps(entry), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            final = validation(model, heldout)
            current_loss = final["assistant_loss_per_token"]
            if not math.isfinite(current_loss):
                raise RuntimeError("Validation non finie : arrêt de l'entraînement")
            language_final = report_validation(model, replay["validation"], model.cfg.context) if replay else None
            if language_final and not math.isfinite(language_final["loss_per_token"]):
                raise RuntimeError("Validation de langue non finie : arrêt de l'entraînement")
            language_ok = not replay or language_final["loss_per_token"] <= language_initial["loss_per_token"] + max_regression
            if current_loss < best_loss and language_ok:
                best_loss, best_step = current_loss, step
                best_state = copy.deepcopy(model.state_dict())
                best_language = language_final
            entry = {"step": step, "validation": final, "best_step": best_step,
                     "language_validation": language_final, "language_guard_passed": language_ok}
            history.append(entry)
            print(json.dumps(entry), flush=True)
            save_training(step)
            atomic_save(export(model, codec, best_state, signature, best_step, best_loss, best_language), args.output / "best.pt")
        elif step % args.checkpoint_every == 0 or step == finish:
            save_training(step)
    if finish % args.eval_every and finish != args.steps:
        # Une mesure de profilage ne modifie pas le calendrier de sélection des poids.
        final = validation(model, heldout)
        language_final = report_validation(model, replay["validation"], model.cfg.context) if replay else None
    seconds = time.perf_counter() - began
    atomic_save(export(model, codec, model.state_dict(), signature, finish, final["assistant_loss_per_token"], language_final), args.output / "last.pt")
    manifest = {"format": FORMAT, "status": "experimental-not-a-general-assistant",
                "initialization": "continued-from-owner-kira-weights", "signature": signature,
                "parameters": sum(p.numel() for p in model.parameters()),
                "planned_steps": args.steps, "completed_steps": finish, "resumed_from_step": start,
                "segment_seconds": seconds, "seconds_per_step": seconds / (finish - start),
                "assistant_tokens_seen": seen, "validation_initial": initial,
                "language_tokens_seen": language_seen,
                "validation_final": final, "best_step": best_step, "best_validation_loss": best_loss,
                "language_validation_initial": language_initial, "language_validation_final": language_final,
                "language_validation_at_best": best_language,
                "language_guard_applied": replay is not None,
                "promotion_status": "not-eligible-general-capabilities-not-demonstrated",
                "assistant_only_dialogue_loss": True, "skill_balanced_sampling": True,
                "test_split_evaluated": False, "examples": {s: len(v) for s, v in groups.items()},
                "history": history}
    write_json(args.output / "manifest.json", manifest)
    print(json.dumps({"steps": finish, "best_step": best_step, "best_validation_loss": best_loss,
                      "seconds_per_step": manifest["seconds_per_step"], "output": str(args.output)}), flush=True)


def prompt_ids(messages, codec, context, reserve):
    history = list(messages)
    if not history or history[-1]["role"] != "user":
        raise ValueError("La conversation doit finir par une question utilisateur")
    while True:
        ids, _ = encode_messages(history, codec)
        ids.append(codec.vocab_size + 1)
        if len(ids) <= context - reserve:
            return ids, history
        if len(history) <= 1:
            raise ValueError("Message trop long pour ce petit contexte. Raccourcis-le.")
        history = history[2:]


@torch.no_grad()
def reply(model, codec, messages, max_new_tokens=64, temperature=0.0, top_k=40, preserve_history=False):
    if not 1 <= max_new_tokens <= 512 or not math.isfinite(temperature) or temperature < 0 or top_k < 0:
        raise ValueError("Limites de génération invalides")
    reserve = min(max_new_tokens, model.cfg.context // 2)
    if preserve_history:
        ids, _ = encode_messages(messages, codec)
        ids.append(codec.vocab_size + 1)
        if len(ids) > model.cfg.context:
            raise ValueError("Contexte d'évaluation trop long : aucune suppression silencieuse")
        history = list(messages)
    else:
        ids, history = prompt_ids(messages, codec, model.cfg.context, reserve)
    model.eval()
    output = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids[-model.cfg.context:]], dtype=torch.long))[0, -1].clone()
        logits[[codec.bos, codec.vocab_size, codec.vocab_size + 1]] = float("-inf")
        if temperature == 0:
            token = int(logits.argmax())
        else:
            if top_k:
                threshold = torch.topk(logits, min(top_k, codec.vocab_size - 1)).values[-1]
                logits[logits < threshold] = float("-inf")
            token = int(torch.multinomial(F.softmax(logits / temperature, dim=-1), 1))
        if token == codec.eos:
            break
        ids.append(token)
        output.append(token)
    return codec.decode(output), history


def evaluate(args):
    if args.limit < 1 or args.threads < 1:
        raise ValueError("limit et threads doivent être positifs")
    if args.output and args.output.exists():
        raise FileExistsError("Rapport existant : choisir un nouveau nom")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    groups, data_hash = load(args.data)
    if not groups[args.split]:
        raise ValueError("Le split demandé ne contient aucun exemple")
    model, codec = load_model(args.checkpoint)
    by_skill = {}
    for item in sorted(groups[args.split], key=lambda x: x["id"]):
        by_skill.setdefault(item["skill"], []).append(item)
    selected = []
    while len(selected) < args.limit and any(by_skill.values()):
        for skill in sorted(by_skill):
            if by_skill[skill] and len(selected) < args.limit:
                selected.append(by_skill[skill].pop(0))
    cases = []
    for item in selected:
        answer, _ = reply(model, codec, item["messages"][:-1], args.max_new_tokens, preserve_history=True)
        expected = item["messages"][-1]["content"]
        cases.append({"id": item["id"], "skill": item["skill"], "grading": item["grading"],
                      "messages": item["messages"][:-1], "reference": expected, "response": answer,
                      "exact_match": normalize(answer) == normalize(expected)})
    closed = [x for x in cases if x["grading"] == "exact"]
    report = {"format": "kira-dialogue-evaluation-v1", "split": args.split,
              "data_sha256": data_hash, "checkpoint_sha256": sha_file(args.checkpoint),
              "selection": "round-robin-skill-then-stable-id", "cases": cases,
              "closed_tasks": {"correct": sum(x["exact_match"] for x in closed), "total": len(closed)},
              "open_answers": "references-for-human-review-no-semantic-score",
              "scope": "narrow-synthetic-exercises-not-a-general-intelligence-score"}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.output.exists():
            raise FileExistsError("Rapport existant : choisir un nouveau nom")
        write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def chat(args):
    if args.threads < 1:
        raise ValueError("threads doit être positif")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model, codec = load_model(args.checkpoint)
    history = []
    print("KIRA — dialogue local expérimental. /reset, /quit, /calc expression (outil exact).")
    while True:
        try:
            text = input("Toi > ").strip()
        except EOFError:
            break
        if text.casefold() == "/quit":
            break
        if text.casefold() == "/reset":
            history = []
            print("Conversation effacée.")
            continue
        if text.casefold() == "/calc" or text.casefold().startswith("/calc "):
            try:
                print("Outil de calcul > " + calculate(text[5:].strip()))
            except ValueError as exc:
                print(str(exc))
            continue
        if not text:
            continue
        try:
            answer, used = reply(model, codec, history + [{"role": "user", "content": text}],
                                 args.max_new_tokens, args.temperature)
        except ValueError as exc:
            print(str(exc))
            continue
        print("KIRA > " + (answer or "[aucune réponse générée]"))
        # Une réponse vide n'est pas inventée pour la conversation suivante.
        history = used + [{"role": "assistant", "content": answer}]


def main():
    parser = argparse.ArgumentParser(description="KIRA Seed Dialogue : nos poids, apprentissage supervisé et épreuves vérifiables")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--output", type=Path, required=True, help="Fichier JSONL à créer ; un fichier identique est réutilisé")
    p.set_defaults(func=lambda a: print(json.dumps(prepare(a.output), ensure_ascii=False)))
    t = commands.add_parser("train")
    t.add_argument("--base", type=Path)
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--output", type=Path, required=True)
    for name, default in (("steps", 2000), ("batch-size", 1), ("threads", 1), ("seed", 42),
                          ("warmup", 50), ("eval-every", 100), ("checkpoint-every", 25)):
        t.add_argument("--" + name, type=int, default=default)
    t.add_argument("--lr", type=float, default=0.0005)
    t.add_argument("--replay-corpus", type=Path, help="Corpus français antérieur pour conserver son apprentissage")
    t.add_argument("--replay-ratio", type=float, default=0.25)
    t.add_argument("--max-language-regression", type=float, default=0.35,
                   help="Hausse maximale de perte par token français pour sélectionner un checkpoint")
    t.add_argument("--stop-after", type=int, help="Étape totale où terminer le segment, sans changer le calendrier LR")
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=train)
    e = commands.add_parser("evaluate")
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--split", choices=("validation", "test"), default="validation")
    e.add_argument("--output", type=Path)
    e.add_argument("--limit", type=int, default=40)
    e.set_defaults(func=evaluate)
    c = commands.add_parser("chat")
    c.add_argument("--temperature", type=float, default=0.0)
    c.set_defaults(func=chat)
    for command in (e, c):
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--threads", type=int, default=1)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, str(exc) + "\n")
    except KeyboardInterrupt:
        parser.exit(130, "Arrêt demandé. Reprise de l'apprentissage avec --resume depuis le dernier training.pt complet, s'il existe.\n")


if __name__ == "__main__":
    main()
