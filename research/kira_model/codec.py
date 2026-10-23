"""Tokenizer de KIRA LM : BPE par octets appris sur nos données, rôles hors du vocabulaire texte.

Le tokenizer texte ne contient aucun jeton spécial : aucune chaîne écrite par un utilisateur ou
présente dans un corpus ne peut produire un rôle, un début ou une fin. Les rôles sont des
identifiants réservés juste après le vocabulaire texte.
"""
from __future__ import annotations

import hashlib
import os
import unicodedata

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

SPECIALS = ("bos", "eos", "user", "assistant", "system", "tool_call", "tool_result", "pad")
RESERVED = 16  # place gardée pour de futurs rôles sans changer la forme du modèle
ROLE_NAMES = ("system", "user", "assistant")


def normalize_text(text: str) -> str:
    # NFC dans notre code plutôt que dans le tokenizer : même comportement pour tout lecteur du fichier.
    return unicodedata.normalize("NFC", text)


def train_tokenizer(texts, vocab_size: int, length: int | None = None) -> str:
    """BPE par octets de style GPT-2, appris uniquement sur les textes fournis (split train)."""
    if not 512 <= vocab_size <= 65536 - 64 - RESERVED:
        raise ValueError("vocab-size doit être compris entre 512 et 65456")
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  max_token_length=32, show_progress=False)
    backend.train_from_iterator((normalize_text(t) for t in texts), trainer=trainer, length=length)
    return backend.to_str()


class Codec:
    def __init__(self, serialized: str):
        self.backend = Tokenizer.from_str(serialized)
        if self.backend.get_added_tokens_decoder():
            raise ValueError("Le tokenizer texte ne doit contenir aucun jeton ajouté ou spécial")
        self.serialized = self.backend.to_str()
        self.sha256 = hashlib.sha256(self.serialized.encode("utf-8")).hexdigest()
        self.text_size = self.backend.get_vocab_size()
        self.special = {name: self.text_size + i for i, name in enumerate(SPECIALS)}
        self.bos, self.eos, self.pad = self.special["bos"], self.special["eos"], self.special["pad"]
        # Multiple de 64 : formes de matrices plus efficaces sur GPU. Les lignes en trop ne sont jamais produites.
        self.vocab_size = -(-(self.text_size + RESERVED) // 64) * 64
        if self.vocab_size > 65536:
            raise ValueError("Vocabulaire trop grand pour des fichiers de jetons uint16")

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(normalize_text(text), add_special_tokens=False).ids

    def encode_batch(self, texts) -> list[list[int]]:
        encoded = self.backend.encode_batch([normalize_text(t) for t in texts], add_special_tokens=False)
        return [item.ids for item in encoded]

    def decode(self, ids) -> str:
        return self.backend.decode([i for i in ids if 0 <= i < self.text_size], skip_special_tokens=False)

    def document(self, text: str) -> list[int]:
        return [self.bos, *self.encode(text), self.eos]

    def banned(self) -> list[int]:
        """Identifiants qu'une réponse ne doit jamais produire (eos reste l'arrêt normal)."""
        return [i for i in range(self.text_size, self.vocab_size) if i != self.eos]

    def conversation(self, messages: list[dict], strict: bool = True) -> tuple[list[int], list[bool]]:
        """bos, puis rôle + texte pour chaque tour ; eos après chaque réponse. Masque vrai sur ce qui s'apprend."""
        check_turns(messages, strict)
        ids, learn = [self.bos], [False]
        for turn in messages:
            assistant = turn["role"] == "assistant"
            text = self.encode(turn["content"])
            ids.append(self.special[turn["role"]])
            learn.append(False)
            ids.extend(text)
            learn.extend([assistant] * len(text))
            if assistant:
                ids.append(self.eos)
                learn.append(True)
        return ids, learn

    def prompt(self, messages: list[dict]) -> list[int]:
        """Conversation se terminant par un message utilisateur, suivie du rôle assistant à compléter."""
        if not messages or messages[-1]["role"] != "user":
            raise ValueError("La conversation doit finir par un message utilisateur")
        # Une réponse vide déjà produite reste dans l'historique observé, telle qu'elle a été générée.
        ids, _ = self.conversation(messages, strict=False)
        return ids + [self.special["assistant"]]


def check_turns(messages: list[dict], strict: bool = True):
    """Un message système facultatif au début, puis utilisateur et assistant en alternance.

    strict (données d'entraînement) : aucun message vide."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation vide")
    turns = messages[1:] if messages[0].get("role") == "system" else messages
    if not turns:
        raise ValueError("Conversation sans message utilisateur")
    for i, turn in enumerate(turns):
        expected = "user" if i % 2 == 0 else "assistant"
        if not isinstance(turn, dict) or turn.get("role") != expected:
            raise ValueError("Alternance utilisateur/assistant invalide")
        if not isinstance(turn.get("content"), str) or (strict and not turn["content"].strip()):
            raise ValueError("Message vide")
