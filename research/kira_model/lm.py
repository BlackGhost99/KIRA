"""KIRA LM 0.5 : décodeur de type Llama initialisé de zéro, entraîné sur GPU et utilisable sur CPU.

Architecture publique et standard : RMSNorm, RoPE, SwiGLU, attention à têtes clé/valeur groupées,
embeddings d'entrée et de sortie liés, aucun biais. Les noms et la convention RoPE (« rotate_half »)
suivent LlamaForCausalLM pour permettre un export ultérieur. Aucun poids d'un autre modèle n'est lu.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

FORMAT = "kira-lm-0.5"


@dataclass
class LMConfig:
    vocab_size: int
    width: int
    layers: int
    heads: int
    kv_heads: int
    hidden: int
    context: int
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("vocab_size", "width", "layers", "heads", "kv_heads", "hidden", "context"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} doit être un entier strictement positif")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width doit se répartir en têtes de dimension paire")
        if self.heads % self.kv_heads:
            raise ValueError("heads doit être un multiple de kv_heads")
        if not (math.isfinite(self.rope_theta) and self.rope_theta > 1 and 0 < self.norm_eps < 1):
            raise ValueError("rope_theta ou norm_eps invalide")

    @property
    def head_dim(self) -> int:
        return self.width // self.heads


# Formes fixées avant tout entraînement ; le vocabulaire vient du tokenizer appris.
PRESETS = {
    "test": dict(width=32, layers=2, heads=4, kv_heads=2, hidden=64, context=64),
    "kira-30m": dict(width=384, layers=12, heads=6, kv_heads=2, hidden=1024, context=1024),
    "kira-100m": dict(width=768, layers=12, heads=12, kv_heads=4, hidden=2048, context=1024),
    "kira-125m-deep": dict(width=576, layers=30, heads=9, kv_heads=3, hidden=1536, context=1024),
}


def preset(name: str, vocab_size: int, **changes) -> LMConfig:
    if name not in PRESETS:
        raise ValueError("Préréglage inconnu : " + ", ".join(sorted(PRESETS)))
    return LMConfig(vocab_size=vocab_size, **(PRESETS[name] | changes))


def config_from_dict(values: dict) -> LMConfig:
    return LMConfig(**values)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        # Calcul en float32 même sous autocast float16, comme LlamaRMSNorm.
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


def rope_tables(cfg: LMConfig, length: int, device=None):
    inverse = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim))
    angles = torch.outer(torch.arange(length, device=device, dtype=torch.float32), inverse)
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos(), angles.sin()


def rotate_half(x):
    half = x.size(-1) // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = cfg.heads, cfg.kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.width, cfg.heads * cfg.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.width, cfg.kv_heads * cfg.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.width, cfg.kv_heads * cfg.head_dim, bias=False)
        self.o_proj = nn.Linear(cfg.heads * cfg.head_dim, cfg.width, bias=False)

    def forward(self, x, cos, sin, cache=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = q * cos + rotate_half(q) * sin
        k = k * cos + rotate_half(k) * sin
        if cache is not None:
            if "k" in cache:
                k = torch.cat((cache["k"], k), dim=2)
                v = torch.cat((cache["v"], v), dim=2)
            cache["k"], cache["v"] = k, v
        past = k.size(2) - length
        if self.kv_heads != self.heads:
            # Même répartition que repeat_kv de Llama : la tête clé j sert des têtes requête consécutives.
            k = k.repeat_interleave(self.heads // self.kv_heads, dim=1)
            v = v.repeat_interleave(self.heads // self.kv_heads, dim=1)
        if past == 0:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif length == 1:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            mask = torch.ones(length, past + length, dtype=torch.bool, device=x.device).tril(diagonal=past)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(out.transpose(1, 2).reshape(batch, length, self.heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.up_proj = nn.Linear(cfg.width, cfg.hidden, bias=False)
        self.down_proj = nn.Linear(cfg.hidden, cfg.width, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.width, cfg.norm_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.width, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


class KiraLM(nn.Module):
    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.width)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.width, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.width, cfg.vocab_size, bias=False)
        cos, sin = rope_tables(cfg, cfg.context)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.reset_parameters()
        self.lm_head.weight = self.embed_tokens.weight

    def reset_parameters(self):
        # Initialisation aléatoire GPT-2/Llama : écart-type 0,02, sorties résiduelles réduites selon la profondeur.
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual = 0.02 / math.sqrt(2 * self.cfg.layers)
        for layer in self.layers:
            nn.init.normal_(layer.self_attn.o_proj.weight, mean=0.0, std=residual)
            nn.init.normal_(layer.mlp.down_proj.weight, mean=0.0, std=residual)

    def forward(self, tokens, caches=None, start: int = 0):
        length = tokens.size(1)
        if length < 1 or start < 0 or start + length > self.cfg.context:
            raise ValueError("Séquence vide ou position au-delà du contexte appris")
        x = self.embed_tokens(tokens)
        cos = self.rope_cos[start:start + length]
        sin = self.rope_sin[start:start + length]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, None if caches is None else caches[i])
        return self.lm_head(self.norm(x))


def parameter_count(model: nn.Module) -> int:
    # Les poids liés ne sont comptés qu'une fois.
    return sum(p.numel() for p in model.parameters())


def token_losses(logits, targets, with_z: bool = False):
    """Sommes (entropie croisée, z²) en float32 et nombre de cibles ; -100 ignore une position."""
    logits = logits.float().flatten(0, 1)
    targets = targets.flatten()
    loss = F.cross_entropy(logits, targets, ignore_index=-100, reduction="sum")
    mask = targets != -100
    # Pénalité z-loss (PaLM) : garde log Z proche de 0, utile en float16.
    z = torch.logsumexp(logits[mask], dim=-1).pow(2).sum() if with_z else logits.new_zeros(())
    return loss, z, mask.sum()


def filter_logits(scores, temperature: float, top_k: int, top_p: float):
    scores = scores / temperature
    if top_k:
        threshold = torch.topk(scores, min(top_k, scores.numel())).values[-1]
        scores = scores.masked_fill(scores < threshold, float("-inf"))
    if top_p < 1.0:
        ordered, order = torch.sort(scores, descending=True)
        probabilities = torch.softmax(ordered, dim=-1)
        # On écarte un jeton quand les plus probables atteignent déjà top_p ; le premier reste toujours.
        drop = probabilities.cumsum(dim=-1) - probabilities >= top_p
        scores = scores.masked_fill(torch.zeros_like(drop).scatter(0, order, drop), float("-inf"))
    return scores


@torch.no_grad()
def generate(model: KiraLM, prompt: list[int], max_new_tokens: int, *, temperature: float = 0.0,
             top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
             stop=(), banned=(), generator=None) -> list[int]:
    """Génération avec cache clé/valeur. Le prompt et la réponse doivent tenir dans le contexte appris."""
    if not prompt or not 1 <= max_new_tokens or len(prompt) + max_new_tokens > model.cfg.context:
        raise ValueError("Prompt vide ou prompt + réponse au-delà du contexte appris")
    if not math.isfinite(temperature) or temperature < 0 or top_k < 0 or not 0 < top_p <= 1 or repetition_penalty < 1:
        raise ValueError("Paramètres de génération invalides")
    model.eval()
    device = next(model.parameters()).device
    banned = torch.tensor(sorted(set(banned)), dtype=torch.long, device=device)
    caches = [{} for _ in model.layers]
    logits = model(torch.tensor([prompt], device=device), caches=caches)[0, -1]
    seen, output = list(prompt), []
    for _ in range(max_new_tokens):
        scores = logits.float().clone()
        if banned.numel():
            scores[banned] = float("-inf")
        if repetition_penalty > 1:
            # Pénalité CTRL : réduit les jetons déjà présents, sans les interdire.
            used = torch.tensor(sorted(set(seen)), dtype=torch.long, device=device)
            values = scores[used]
            scores[used] = torch.where(values > 0, values / repetition_penalty, values * repetition_penalty)
        if temperature == 0:
            token = int(scores.argmax())
        else:
            probabilities = torch.softmax(filter_logits(scores, temperature, top_k, top_p), dim=-1)
            token = int(torch.multinomial(probabilities, 1, generator=generator))
        if token in stop:
            break
        output.append(token)
        seen.append(token)
        if len(seen) >= model.cfg.context:
            break
        logits = model(torch.tensor([[token]], device=device), caches=caches, start=len(seen) - 1)[0, -1]
    return output


def describe(cfg: LMConfig) -> dict:
    model = KiraLM(cfg)
    return {"config": asdict(cfg), "parameters": parameter_count(model),
            "embedding_parameters": cfg.vocab_size * cfg.width}
