import unittest

import torch

from research.kira_model import lm
from research.kira_model.codec import Codec, check_turns, train_tokenizer

TEXTS = ["Bonjour, je m'appelle KIRA. L'été est chaud à Libreville.",
         "Les élèves étudient la physique : la force, la masse et l'énergie.",
         "Une chercheuse prépare un carnet et vérifie ses mesures."] * 40


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        cls.cfg = lm.preset("test", vocab_size=320)
        cls.model = lm.KiraLM(cls.cfg).eval()

    def test_invalid_shapes_are_refused(self):
        for changes in ({"width": 30}, {"kv_heads": 3}, {"layers": 0}, {"context": 1.5}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                lm.preset("test", vocab_size=320, **changes)
        with self.assertRaises(ValueError):
            lm.preset("inconnu", vocab_size=320)

    def test_future_tokens_never_change_past_logits(self):
        x = torch.randint(0, 320, (2, 24))
        y = x.clone()
        y[:, 12:] = torch.randint(0, 320, (2, 12))
        with torch.no_grad():
            self.assertTrue(torch.allclose(self.model(x)[:, :12], self.model(y)[:, :12], atol=1e-5))

    def test_cache_matches_full_recomputation(self):
        x = torch.randint(0, 320, (1, 30))
        with torch.no_grad():
            full = self.model(x)
            caches = [{} for _ in self.model.layers]
            parts = [self.model(x[:, :9], caches=caches), self.model(x[:, 9:20], caches=caches, start=9)]
            parts += [self.model(x[:, i:i + 1], caches=caches, start=i) for i in range(20, 30)]
        self.assertTrue(torch.allclose(full, torch.cat(parts, dim=1), atol=1e-5))

    def test_rope_scores_depend_only_on_relative_position(self):
        cos, sin = lm.rope_tables(self.cfg, 40)
        q, k = torch.randn(self.cfg.head_dim), torch.randn(self.cfg.head_dim)

        def rotate(v, p):
            return v * cos[p] + lm.rotate_half(v) * sin[p]
        self.assertAlmostEqual(float(rotate(q, 7) @ rotate(k, 3)), float(rotate(q, 30) @ rotate(k, 26)), places=4)
        self.assertAlmostEqual(float(rotate(q, 11).norm()), float(q.norm()), places=4)

    def test_tied_embeddings_and_scaled_residual_initialisation(self):
        self.assertIs(self.model.lm_head.weight, self.model.embed_tokens.weight)
        big = lm.KiraLM(lm.preset("test", vocab_size=320, layers=8, width=64, hidden=128))
        self.assertLess(big.layers[0].mlp.down_proj.weight.std(), big.layers[0].mlp.up_proj.weight.std())

    def test_presets_have_the_documented_size(self):
        cfg = lm.preset("kira-100m", vocab_size=32064)
        per_layer = (2 * cfg.width * cfg.width + 2 * cfg.width * cfg.kv_heads * cfg.head_dim
                     + 3 * cfg.width * cfg.hidden + 2 * cfg.width)
        expected = cfg.vocab_size * cfg.width + cfg.layers * per_layer + cfg.width
        with torch.device("meta"):
            model = lm.KiraLM(cfg)
        self.assertEqual(lm.parameter_count(model), expected)
        self.assertTrue(95e6 < expected < 105e6)

    def test_losses_ignore_masked_targets_and_report_z(self):
        logits = torch.randn(1, 4, 320)
        targets = torch.tensor([[5, -100, 7, -100]])
        loss, z, count = lm.token_losses(logits, targets, with_z=True)
        expected = torch.nn.functional.cross_entropy(logits[0, [0, 2]], torch.tensor([5, 7]), reduction="sum")
        self.assertEqual(int(count), 2)
        self.assertAlmostEqual(float(loss), float(expected), places=4)
        self.assertGreater(float(z), 0)

    def test_generation_is_bounded_and_respects_stop_and_bans(self):
        greedy = lm.generate(self.model, [1, 2, 3], 8)
        self.assertEqual(greedy, lm.generate(self.model, [1, 2, 3], 8))
        banned = list(range(0, 300))
        sampled = lm.generate(self.model, [1, 2, 3], 12, temperature=1.0, top_p=0.9, top_k=50, banned=banned,
                              repetition_penalty=1.2, generator=torch.Generator().manual_seed(3))
        self.assertTrue(all(t >= 300 for t in sampled))
        first = lm.generate(self.model, [1, 2, 3], 1)[0]
        self.assertEqual(lm.generate(self.model, [1, 2, 3], 8, stop={first}), [])
        with self.assertRaises(ValueError):
            lm.generate(self.model, list(range(60)), 8)
        with self.assertRaises(ValueError):
            lm.generate(self.model, [1], 4, top_p=0)

    def test_nucleus_filter_keeps_the_most_likely_token(self):
        scores = torch.tensor([4.0, 1.0, 0.5, -2.0])
        kept = lm.filter_logits(scores, 1.0, 0, 0.5)
        self.assertTrue(torch.isfinite(kept[0]))
        self.assertTrue(torch.isinf(kept[1:]).all())


class CodecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.codec = Codec(train_tokenizer(TEXTS, 600))

    def test_roundtrip_and_unicode_normalisation(self):
        for text in ("Bonjour KIRA !", "  deux  espaces\n", "中文 🧠", ""):
            self.assertEqual(self.codec.decode(self.codec.encode(text)), text)
        self.assertEqual(self.codec.encode("été"), self.codec.encode("été"))

    def test_role_tokens_cannot_be_written_as_text(self):
        for text in ("<|user|>", "<assistant>", "<eos>", "<|endoftext|>", "\x00"):
            self.assertTrue(all(i < self.codec.text_size for i in self.codec.encode(text)))
        self.assertEqual(self.codec.vocab_size % 64, 0)
        self.assertGreaterEqual(self.codec.vocab_size, self.codec.text_size + 16)

    def test_tokenizer_with_added_special_tokens_is_refused(self):
        from tokenizers import Tokenizer
        backend = Tokenizer.from_str(self.codec.serialized)
        backend.add_special_tokens(["<|user|>"])
        with self.assertRaises(ValueError):
            Codec(backend.to_str())

    def test_conversation_learns_only_answers_and_their_end(self):
        messages = [{"role": "system", "content": "Tu es KIRA."}, {"role": "user", "content": "Bonjour"},
                    {"role": "assistant", "content": "Bonjour !"}]
        ids, learn = self.codec.conversation(messages)
        answer = self.codec.encode("Bonjour !")
        self.assertEqual(ids[0], self.codec.bos)
        self.assertEqual(ids[-1], self.codec.eos)
        self.assertEqual([t for t, keep in zip(ids, learn) if keep], answer + [self.codec.eos])
        prompt = self.codec.prompt(messages[:2])
        self.assertEqual(prompt[-1], self.codec.special["assistant"])

    def test_turn_order_and_emptiness(self):
        with self.assertRaises(ValueError):
            check_turns([{"role": "assistant", "content": "Salut"}])
        with self.assertRaises(ValueError):
            check_turns([{"role": "user", "content": " "}])
        check_turns([{"role": "user", "content": "a"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "b"}],
                    strict=False)

    def test_banned_ids_cover_roles_and_padding_but_not_end(self):
        banned = set(self.codec.banned())
        self.assertNotIn(self.codec.eos, banned)
        self.assertTrue({self.codec.bos, self.codec.pad, self.codec.special["user"], self.codec.vocab_size - 1} <= banned)
        self.assertTrue(all(i >= self.codec.text_size for i in banned))


if __name__ == "__main__":
    unittest.main()
