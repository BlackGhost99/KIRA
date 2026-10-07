"""Contrats d'accès : cookies, rotation, révocation, CSRF et confirmation sensible."""
import concurrent.futures
import time
from unittest.mock import patch

from app import auth, config, db
from app.main import app
from tests.helpers import KiraTestCase, call


class SessionTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        auth.throttle.ok("203.0.113.9")
        from app.api import _global_throttle
        _global_throttle.ok("*")

    def login(self, **headers):
        return call(app, "POST", "/api/auth/login", {"password": config.settings.owner_password},
                    headers={"x-kira-csrf": "1", **headers})

    def cookie_headers(self, reply, **extra):
        return {"cookie": "; ".join(f"{k}={m.value}" for k, m in reply.cookies.items()),
                "x-kira-csrf": "1", **extra}

    def test_login_has_only_httponly_secure_cookies_and_hashes_in_database(self):
        r = self.login()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.json(), {"name": "Brice"})
        for name, path in ((auth.ACCESS_COOKIE, "/api"), (auth.REFRESH_COOKIE, "/api/auth")):
            cookie = r.cookies[name]
            self.assertTrue(cookie["httponly"])
            self.assertTrue(cookie["secure"])
            self.assertEqual(cookie["samesite"], "strict")
            self.assertEqual(cookie["path"], path)
            self.assertFalse(cookie["domain"])
            self.assertNotIn(cookie.value, str(db.get_db().q("SELECT * FROM owner_sessions")))
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(r)).status, 200)

    def test_access_expiry_refresh_rotation_and_replay_revoke_the_session(self):
        r = self.login()
        row = db.get_db().q1("SELECT * FROM owner_sessions")
        db.get_db().update("owner_sessions", row["id"], {"access_expires_at": time.time() - 1})
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(r)).status, 401)
        rotated = call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(r))
        self.assertEqual(rotated.status, 200)
        self.assertNotEqual(rotated.cookies[auth.REFRESH_COOKIE].value, r.cookies[auth.REFRESH_COOKIE].value)
        new_row = db.get_db().q1("SELECT * FROM owner_sessions")
        self.assertEqual(row["expires_at"], new_row["expires_at"])
        self.assertEqual(row["reauth_at"], new_row["reauth_at"])
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(rotated)).status, 200)
        replay = call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(r))
        self.assertEqual(replay.status, 401)
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(rotated)).status, 401)
        self.assertEqual(call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(rotated)).status, 401)

    def test_two_simultaneous_rotations_cannot_create_two_valid_sessions(self):
        _, _, refresh = auth.create_session()
        def rotate():
            try:
                return auth.refresh_session(refresh)
            finally:
                db.get_db().close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            replies = list(pool.map(lambda _: rotate(), range(2)))
        self.assertEqual(sum(r is not None for r in replies), 1)
        successful = next(r for r in replies if r)
        self.assertIsNone(auth.authenticate_access(successful[1]))

    def test_logout_revokes_server_side_and_clears_both_cookies(self):
        r = self.login()
        h = self.cookie_headers(r)
        out = call(app, "POST", "/api/auth/logout", headers=h)
        self.assertEqual(out.status, 200)
        self.assertEqual(set(out.cookies), {auth.ACCESS_COOKIE, auth.REFRESH_COOKIE})
        self.assertTrue(all(m["max-age"] == "0" for m in out.cookies.values()))
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=h).status, 401)
        self.assertEqual(call(app, "POST", "/api/auth/refresh", headers=h).status, 401)

    def test_session_list_contains_no_credentials_and_revocation_is_effective(self):
        one, two = self.login(), self.login()
        sessions = call(app, "GET", "/api/auth/sessions", headers=self.cookie_headers(one)).json()["sessions"]
        self.assertEqual(len(sessions), 2)
        self.assertEqual(sum(s["current"] for s in sessions), 1)
        self.assertFalse(any("hash" in k for s in sessions for k in s))
        other = next(s for s in sessions if not s["current"])
        out = call(app, "DELETE", "/api/auth/sessions/" + other["id"], headers=self.cookie_headers(one))
        self.assertEqual(out.status, 200)
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(two)).status, 401)

    def test_password_and_secret_changes_invalidate_existing_sessions(self):
        r = self.login()
        original = config.settings.owner_password
        config.settings.owner_password += "-changed"
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=self.cookie_headers(r)).status, 401)
        config.settings.owner_password = original
        config.settings.secret_key += "-changed"
        self.assertEqual(call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(r)).status, 401)

    def test_legacy_bearer_is_disabled_by_default_and_subject_must_be_owner(self):
        token = auth.make_token()
        h = {"authorization": "Bearer " + token}
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=h).status, 401)
        config.settings.allow_legacy_bearer = True
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=h).status, 200)
        h["authorization"] = "Bearer " + auth.make_token(subject="device")
        self.assertEqual(call(app, "GET", "/api/auth/me", headers=h).status, 401)

    def test_csrf_header_and_same_origin_are_required_for_cookie_mutations(self):
        self.assertEqual(call(app, "POST", "/api/auth/login", {"password": config.settings.owner_password}).status, 403)
        self.assertEqual(self.login(origin="https://evil.example").status, 403)
        config.settings.public_origin = "https://kira.example"
        r = self.login(origin="https://kira.example")
        h = self.cookie_headers(r)
        self.assertEqual(call(app, "POST", "/api/memory", {"content": "test"}, headers={"cookie": h["cookie"]}).status, 403)
        for path in ("/api/memory", "/api/auth/refresh", "/api/auth/logout"):
            self.assertEqual(call(app, "POST", path, {"content": "test"}, headers={**h, "origin": "https://evil.example"}).status, 403)
        self.assertEqual(call(app, "POST", "/api/memory", {"content": "test"}, headers={**h, "origin": "https://kira.example"}).status, 200)

    def test_sensitive_action_requires_recent_password_before_any_effect(self):
        r = self.login()
        h = self.cookie_headers(r)
        session = db.get_db().q1("SELECT * FROM owner_sessions")
        db.get_db().update("owner_sessions", session["id"], {"reauth_at": time.time() - 3600})
        with patch("app.api.evolution.approve", return_value={"ok": True}) as approve:
            self.assertEqual(call(app, "POST", "/api/evolution/1/approve", {}, headers=h).status, 428)
            approve.assert_not_called()
            bad = call(app, "POST", "/api/auth/reauth", {"password": "wrong"}, headers=h)
            self.assertEqual(bad.status, 401)
            self.assertEqual(call(app, "GET", "/api/auth/me", headers=h).status, 200)
            self.assertEqual(call(app, "POST", "/api/auth/reauth", {"password": config.settings.owner_password}, headers=h).status, 200)
            self.assertEqual(call(app, "POST", "/api/evolution/1/approve", {}, headers=h).status, 200)
            approve.assert_called_once_with(1)

    def test_expired_refresh_cannot_be_renewed(self):
        r = self.login()
        session = db.get_db().q1("SELECT * FROM owner_sessions")
        db.get_db().update("owner_sessions", session["id"], {"expires_at": time.time() - 1})
        self.assertEqual(call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(r)).status, 401)

    def test_unpause_cannot_bypass_confirmation_with_falsy_non_booleans(self):
        r = self.login()
        h = self.cookie_headers(r)
        session = db.get_db().q1("SELECT * FROM owner_sessions")
        db.get_db().update("owner_sessions", session["id"], {"reauth_at": time.time() - 3600})
        for value in (0, "", None, []):
            for method, route in (("PATCH", "/api/devices/1"), ("POST", "/api/devices/pause-all")):
                with self.subTest(value=value, route=route):
                    self.assertEqual(call(app, method, route, {"paused": value}, headers=h).status, 400)
        with patch("app.api.devices.set_global_pause") as change:
            self.assertEqual(call(app, "POST", "/api/devices/pause-all", {"paused": False}, headers=h).status, 428)
            change.assert_not_called()

    def test_revocation_rotation_and_confirmation_leave_no_tokens_in_audit(self):
        from app import audit
        r = self.login()
        rotated = call(app, "POST", "/api/auth/refresh", headers=self.cookie_headers(r))
        call(app, "POST", "/api/auth/reauth", {"password": config.settings.owner_password}, headers=self.cookie_headers(rotated))
        call(app, "POST", "/api/auth/logout", headers=self.cookie_headers(rotated))
        entries = audit.recent()
        self.assertTrue({"login", "session_refresh", "reauth", "logout"}.issubset({e["action"] for e in entries}))
        for reply in (r, rotated):
            for morsel in reply.cookies.values():
                self.assertNotIn(morsel.value, str(entries))
