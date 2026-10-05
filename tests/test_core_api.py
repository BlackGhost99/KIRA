"""Routes du cœur : séparation stricte propriétaire / machine, appairage, accords, résultats, limites de taille."""
import base64
import json

from app import api, auth, db, devices
from app.main import app
from tests.helpers import call
from tests.test_api import ApiTestCase

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 60
ALL = ["monitor", "fs_read", "fs_write", "fs_delete", "exec", "screen", "input"]
INFO = {"enabled": ALL, "roots": ["/home/b/KIRA"], "system": {"hostname": "pc"}, "version": "1.0"}


class CoreApiCase(ApiTestCase):
    def setUp(self):
        super().setUp()
        devices.pair_throttle = auth.LoginThrottle(max_failures=8, window=600)
        api._pair_global = auth.LoginThrottle(max_failures=40, window=600)

    def dev(self, method, path, body=None, token=None, **kw):
        headers = {"authorization": f"Device {token}"} if token else {}
        return call(app, method, path, json_body=body, headers=headers, **kw)

    def pair(self, name="PC-bureau"):
        code = self.api("POST", "/api/devices/pair-code").json()["code"]
        r = self.dev("POST", "/api/core/pair", {"code": code, "name": name, "platform": "linux", "info": INFO, "version": "1.0"})
        self.assertEqual(r.status, 200, r.text)
        return r.json()


class AccessSeparationTests(CoreApiCase):
    def test_owner_routes_refuse_anonymous_forged_and_device_tokens(self):
        paired = self.pair()
        routes = [
            ("GET", "/api/devices"), ("POST", "/api/devices/pair-code"), ("POST", "/api/devices/pause-all"),
            ("PATCH", "/api/devices/1"), ("DELETE", "/api/devices/1"), ("GET", "/api/devices/1/actions"),
            ("GET", "/api/actions"), ("GET", "/api/actions/pending"), ("GET", "/api/actions/1"),
            ("POST", "/api/actions/1/approve"), ("POST", "/api/actions/1/deny"), ("GET", "/api/files/" + "0" * 32),
        ]
        for method, path in routes:
            for headers in ({}, {"authorization": "Bearer faux.jeton"}, {"authorization": f"Bearer {paired['token']}"},
                            {"authorization": f"Device {paired['token']}"}):
                r = call(app, method, path, json_body={} if method != "GET" else None, headers=headers)
                self.assertEqual(r.status, 401, f"{method} {path} {headers}")

    def test_machine_routes_refuse_owner_token_and_garbage(self):
        owner = {"authorization": f"Bearer {self.token}"}
        for path in ("/api/core/poll", "/api/core/result"):
            self.assertEqual(call(app, "POST", path, json_body={}, headers=owner).status, 401)
            self.assertEqual(call(app, "POST", path, json_body={}).status, 401)
            self.assertEqual(call(app, "POST", path, json_body={}, headers={"authorization": "Device kdev_inconnu"}).status, 401)
            self.assertEqual(call(app, "POST", path, json_body={}, headers={"authorization": f"Device {self.token}"}).status, 401)

    def test_revoked_device_is_locked_out_immediately(self):
        paired = self.pair()
        self.assertEqual(self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, paired["token"]).status, 200)
        self.assertEqual(self.api("DELETE", f"/api/devices/{paired['device_id']}").status, 200)
        self.assertEqual(self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, paired["token"]).status, 401)


class PairingApiTests(CoreApiCase):
    def test_pairing_flow_and_listing(self):
        paired = self.pair("Mon PC")
        listing = self.api("GET", "/api/devices").json()
        self.assertEqual([d["name"] for d in listing["devices"]], ["Mon PC"])
        self.assertNotIn("token", json.dumps(listing))
        self.assertEqual(listing["never_auto"], ["fs_delete"])
        self.assertEqual(listing["devices"][0]["policy"]["exec"], "ask")
        self.assertEqual(paired["name"], "Mon PC")

    def test_wrong_codes_are_throttled_per_address(self):
        for _ in range(8):
            self.assertEqual(self.dev("POST", "/api/core/pair", {"code": "AAAA-BBBB", "name": "x", "platform": "linux"}).status, 400)
        good = self.api("POST", "/api/devices/pair-code").json()["code"]
        r = self.dev("POST", "/api/core/pair", {"code": good, "name": "x", "platform": "linux"})
        self.assertEqual(r.status, 429)  # même le bon code est refusé tant que l'adresse est bloquée

    def test_pair_body_is_capped(self):
        r = call(app, "POST", "/api/core/pair", raw_body=b'{"code":"' + b"A" * 70_000 + b'"}')
        self.assertEqual(r.status, 413)


class ActionFlowTests(CoreApiCase):
    def test_full_cycle_with_approval(self):
        paired = self.pair()
        token, did = paired["token"], paired["device_id"]
        # lecture : libre -> en file ; écriture : sur accord -> en attente
        read = devices.request_action(did, "list_dir", {"path": "/home/b/KIRA"})
        write = devices.request_action(did, "write_file", {"path": "/home/b/KIRA/n.txt", "content": "bonjour"})
        self.assertEqual((read["status"], write["status"]), ("queued", "pending"))

        pending = self.api("GET", "/api/actions/pending").json()["actions"]
        self.assertEqual([a["id"] for a in pending], [write["id"]])
        self.assertIn("n.txt", pending[0]["summary"])
        self.assertEqual(pending[0]["detail"], "bonjour")

        got = self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, token).json()
        self.assertEqual(got["action"]["id"], read["id"])  # seule l'action autorisée sort
        self.assertEqual(self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, token).json()["action"], None)
        done = self.dev("POST", "/api/core/result", {"action_id": read["id"], "ok": True, "output": "a.txt\nb.txt"}, token)
        self.assertEqual((done.status, done.json()["status"]), (200, "done"))
        # rejouer le même résultat est refusé
        self.assertEqual(self.dev("POST", "/api/core/result", {"action_id": read["id"], "ok": True, "output": "x"}, token).status, 400)

        self.assertEqual(self.api("POST", f"/api/actions/{write['id']}/approve").json()["status"], "queued")
        got = self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, token).json()
        self.assertEqual((got["action"]["id"], got["action"]["kind"]), (write["id"], "write_file"))
        self.assertEqual(got["action"]["args"]["content"], "bonjour")
        self.assertEqual(self.api("POST", f"/api/actions/{write['id']}/approve").status, 400)  # déjà décidée

        shown = self.api("GET", f"/api/actions/{read['id']}").json()
        self.assertEqual((shown["status"], shown["result"]), ("done", "a.txt\nb.txt"))
        self.assertEqual(len(self.api("GET", f"/api/devices/{did}/actions").json()["actions"]), 2)

    def test_denied_request_never_reaches_the_machine(self):
        paired = self.pair()
        did = paired["device_id"]
        act = devices.request_action(did, "run_command", {"command": "echo salut"})
        self.assertEqual(self.api("POST", f"/api/actions/{act['id']}/deny").json()["status"], "denied")
        self.assertIsNone(self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, paired["token"]).json()["action"])

    def test_machine_cannot_report_on_another_machines_action(self):
        a, b = self.pair("A"), self.pair("B")
        act = devices.request_action(a["device_id"], "list_dir", {"path": "/x"})
        self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, a["token"])
        r = self.dev("POST", "/api/core/result", {"action_id": act["id"], "ok": True, "output": "piraté"}, b["token"])
        self.assertEqual(r.status, 400)
        self.assertEqual(self.api("GET", f"/api/actions/{act['id']}").json()["status"], "running")

    def test_policy_changes_and_delete_can_never_be_free(self):
        did = self.pair()["device_id"]
        ok = self.api("PATCH", f"/api/devices/{did}", {"policy": {"exec": "auto", "fs_write": "deny"}}).json()
        self.assertEqual((ok["policy"]["exec"], ok["policy"]["fs_write"]), ("auto", "deny"))
        self.assertEqual(self.api("PATCH", f"/api/devices/{did}", {"policy": {"fs_delete": "auto"}}).status, 400)
        self.assertEqual(self.api("PATCH", f"/api/devices/{did}", {"policy": {"inconnu": "ask"}}).status, 400)
        self.assertEqual(self.api("PATCH", f"/api/devices/{did}", {}).status, 400)
        self.assertEqual(self.api("PATCH", f"/api/devices/{did}", {"name": "  Portable   de Brice "}).json()["name"], "Portable de Brice")

    def test_global_pause_cancels_waiting_requests_and_stops_polling(self):
        paired = self.pair()
        devices.request_action(paired["device_id"], "list_dir", {"path": "/x"})
        self.assertEqual(self.api("POST", "/api/devices/pause-all", {"paused": True}).json()["paused"], True)
        got = self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, paired["token"]).json()
        self.assertEqual((got["action"], got.get("paused")), (None, True))
        self.assertEqual(devices.request_action(paired["device_id"], "list_dir", {"path": "/x"})["status"], "denied")
        self.api("POST", "/api/devices/pause-all", {"paused": False})
        self.assertEqual(devices.request_action(paired["device_id"], "list_dir", {"path": "/x"})["status"], "queued")

    def test_screenshot_is_stored_checked_and_served_only_to_the_owner(self):
        paired = self.pair()
        act = devices.request_action(paired["device_id"], "screenshot", {})
        self.api("POST", f"/api/actions/{act['id']}/approve")
        self.dev("POST", "/api/core/poll", {"info": INFO, "wait": 0}, paired["token"])
        good = base64.b64encode(PNG).decode()
        fake = base64.b64encode(b"MZ-pas-une-image").decode()
        r = self.dev("POST", "/api/core/result", {"action_id": act["id"], "ok": True, "output": "1280x720",
                                                  "images": [{"b64": good}, {"b64": fake}, {"b64": "%%%"}]}, paired["token"])
        self.assertEqual(r.status, 200)
        shown = self.api("GET", f"/api/actions/{act['id']}").json()
        self.assertEqual(len(shown["files"]), 1)  # seule la vraie image est gardée
        url = shown["files"][0]
        self.assertEqual(call(app, "GET", url).status, 401)  # une capture d'écran n'est jamais publique
        served = self.api("GET", url)
        self.assertEqual((served.status, served.body), (200, PNG))
        self.assertEqual(served.headers["cache-control"], "private, no-store")

    def test_result_body_is_capped(self):
        paired = self.pair()
        big = b'{"action_id":1,"ok":true,"output":"' + b"x" * (api.RESULT_BODY_LIMIT + 10) + b'"}'
        r = call(app, "POST", "/api/core/result", raw_body=big, headers={"authorization": f"Device {paired['token']}"})
        self.assertEqual(r.status, 413)

    def test_poll_body_is_capped_and_wait_is_bounded(self):
        paired = self.pair()
        r = call(app, "POST", "/api/core/poll", raw_body=b'{"info":{"x":"' + b"a" * 70_000 + b'"}}',
                 headers={"authorization": f"Device {paired['token']}"})
        self.assertEqual(r.status, 413)
        self.assertEqual(self.dev("POST", "/api/core/poll", {"wait": "pas un nombre", "info": {}, }, paired["token"]).status, 200)


class AuditTests(CoreApiCase):
    def test_everything_is_in_the_audit_log_and_no_token_leaks(self):
        paired = self.pair()
        act = devices.request_action(paired["device_id"], "run_command", {"command": "ls"})
        self.api("POST", f"/api/actions/{act['id']}/approve")
        entries = json.dumps(self.api("GET", "/api/audit", query="limit=300").json())
        for kind in ("device_pair_code", "device_paired", "device_action_requested", "device_action_approved"):
            self.assertIn(kind, entries)
        self.assertNotIn(paired["token"], entries)
        self.assertNotIn(paired["token"], json.dumps([dict(r) for r in db.get_db().q("SELECT * FROM devices")]))  # seul le hash est stocké
