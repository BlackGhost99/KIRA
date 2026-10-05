"""Base de données, mémoire, authentification."""
import time
import unittest

from app import auth, config, db, memory
from tests.helpers import KiraTestCase


class DatabaseTests(KiraTestCase):
    def test_insert_returns_id_and_kv_roundtrip(self):
        d = db.get_db()
        first = d.insert("memories", {"kind": "fact", "content": "a", "tags": "", "source": "", "created_at": "x", "updated_at": "x"})
        second = d.insert("memories", {"kind": "fact", "content": "b", "tags": "", "source": "", "created_at": "x", "updated_at": "x"})
        self.assertEqual(second, first + 1)
        d.kv_set("k", "1")
        d.kv_set("k", "2")
        self.assertEqual(d.kv_get("k"), "2")
        self.assertEqual(d.kv_get("absent", "défaut"), "défaut")

    def test_foreign_key_cascade_on_conversation_delete(self):
        d = db.get_db()
        d.run("INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c1','t','x','x')")
        d.insert("messages", {"conversation_id": "c1", "role": "user", "content": "salut", "meta": "{}", "created_at": "x"})
        d.run("DELETE FROM conversations WHERE id = 'c1'")
        self.assertEqual(d.q("SELECT * FROM messages"), [])

    def test_sql_with_percent_and_question_mark_params(self):
        d = db.get_db()
        d.insert("memories", {"kind": "note", "content": "100% sûr ?", "tags": "", "source": "", "created_at": "x", "updated_at": "x"})
        rows = d.q("SELECT content FROM memories WHERE content LIKE ?", ["%100%%"])
        self.assertEqual(len(rows), 1)


class MemoryTests(KiraTestCase):
    def test_add_dedupes_and_validates(self):
        a = memory.add("fact", "Brice étudie la mécanique quantique")
        b = memory.add("fact", "Brice étudie la mécanique quantique")
        self.assertEqual(a["id"], b["id"])
        with self.assertRaises(ValueError):
            memory.add("fact", "   ")
        self.assertEqual(memory.add("inconnu", "x")["kind"], "fact")

    def test_search_finds_by_meaning_words_with_accents_and_plurals(self):
        memory.add("profile", "Brice n'est pas à l'aise avec les équations différentielles")
        memory.add("fact", "Le serveur Render dort après quinze minutes")
        memory.add("knowledge", "Les ondes gravitationnelles ont été détectées en 2015", source="https://exemple.org/ligo")
        hits = memory.search("équation différentielle", limit=3)
        self.assertEqual(hits[0]["kind"], "profile")
        hits = memory.search("gravitationnelle détection", kinds=("knowledge",))
        self.assertEqual(len(hits), 1)
        self.assertEqual(memory.search("zzzzqqq inexistant"), [])

    def test_empty_query_returns_recent(self):
        memory.add("note", "premier")
        memory.add("note", "second")
        self.assertEqual(memory.search("")[0]["content"], "second")

    def test_update_delete_profile_counts(self):
        item = memory.add("fact", "ancien")
        memory.update(item["id"], content="nouveau", kind="profile")
        self.assertEqual(memory.profile()[0]["content"], "nouveau")
        self.assertEqual(memory.counts(), {"profile": 1})
        self.assertTrue(memory.delete(item["id"]))
        self.assertFalse(memory.delete(item["id"]))

    def test_search_messages(self):
        d = db.get_db()
        d.run("INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c','t','x','x')")
        d.insert("messages", {"conversation_id": "c", "role": "user", "content": "Explique la relativité restreinte", "meta": "{}", "created_at": "2026-01-01"})
        d.insert("messages", {"conversation_id": "c", "role": "assistant", "content": "Cuisine du poulet", "meta": "{}", "created_at": "2026-01-01"})
        found = memory.search_messages("relativité")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["role"], "user")


class AuthTests(KiraTestCase):
    def test_token_roundtrip_tamper_and_expiry(self):
        token = auth.make_token()
        self.assertEqual(auth.verify_token(token)["sub"], "owner")
        body, sig = token.split(".")
        self.assertIsNone(auth.verify_token(body + "." + sig[:-2] + "AA"))
        self.assertIsNone(auth.verify_token("n'importe quoi"))
        expired = auth.make_token(days=-1)
        self.assertIsNone(auth.verify_token(expired))

    def test_token_from_other_secret_is_rejected(self):
        token = auth.make_token()
        config.settings.secret_key = "autre-cle"
        self.assertIsNone(auth.verify_token(token))

    def test_no_secret_configured_means_no_token(self):
        config.settings.secret_key = ""
        config.settings.owner_password = ""
        self.assertIsNone(auth.verify_token("a.b"))
        with self.assertRaises(auth.AuthError):
            auth.make_token()

    def test_password_checks(self):
        self.assertTrue(auth.check_password("mot-de-passe-test"))
        self.assertFalse(auth.check_password("faux"))
        self.assertFalse(auth.check_password(""))
        config.settings.owner_password = ""
        self.assertFalse(auth.check_password(""))
        self.assertFalse(auth.check_cron_token("x"))

    def test_throttle_blocks_after_five_failures_and_resets_on_success(self):
        t = auth.LoginThrottle(max_failures=3, window=300)
        for _ in range(3):
            self.assertFalse(t.blocked("ip"))
            t.fail("ip")
        self.assertTrue(t.blocked("ip"))
        self.assertFalse(t.blocked("autre"))
        t.ok("ip")
        self.assertFalse(t.blocked("ip"))

    def test_throttle_window_expires(self):
        t = auth.LoginThrottle(max_failures=1, window=0)
        t.fail("ip")
        time.sleep(0.01)
        self.assertFalse(t.blocked("ip"))


if __name__ == "__main__":
    unittest.main()


class DotenvTests(KiraTestCase):
    def test_reads_values_without_overriding_the_environment(self):
        import os
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "# commentaire\n\nKIRA_T_A=un\nKIRA_T_B = \"deux mots\"\nKIRA_T_C='trois'\nKIRA_T_VIDE=\nKIRA_T_PRIS=nouveau\nligne sans egal\n",
                encoding="utf-8",
            )
            os.environ["KIRA_T_PRIS"] = "ancien"
            for k in ("KIRA_T_A", "KIRA_T_B", "KIRA_T_C", "KIRA_T_VIDE"):
                os.environ.pop(k, None)
            self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("KIRA_T_A", "KIRA_T_B", "KIRA_T_C", "KIRA_T_VIDE", "KIRA_T_PRIS")])
            self.assertEqual(config.load_dotenv(path), 3)
            self.assertEqual(
                (os.environ["KIRA_T_A"], os.environ["KIRA_T_B"], os.environ["KIRA_T_C"], os.environ["KIRA_T_PRIS"]),
                ("un", "deux mots", "trois", "ancien"),
            )
            self.assertNotIn("KIRA_T_VIDE", os.environ)
            self.assertEqual(config.load_dotenv(Path(tmp) / "absent.env"), 0)
