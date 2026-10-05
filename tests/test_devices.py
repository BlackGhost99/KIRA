"""Le cœur côté serveur : appairage, jetons, politique par niveaux, accords, file d'actions, révocation."""
import base64
import time

from app import audit, auth, db, devices, files
from app.devices import DeviceError
from tests.helpers import KiraTestCase

ALL = ["monitor", "fs_read", "fs_write", "fs_delete", "exec", "screen", "input"]
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 40


def pair_device(name="PC-bureau", enabled=None, platform="linux"):
    code = devices.create_pair_code()["code"]
    return devices.pair(code, name, platform, {"enabled": ALL if enabled is None else enabled, "roots": ["/home/b/KIRA"],
                                               "system": {"hostname": "h"}}, "1.0")


class ArgsTests(KiraTestCase):
    def test_valid_and_defaults(self):
        self.assertEqual(devices.validate_args("read_file", {"path": "/a/b"}), {"path": "/a/b", "max_bytes": 100000, "offset": 0})
        self.assertEqual(devices.validate_args("run_command", {"command": "ls"})["timeout"], 60)

    def test_rejects_bad_input(self):
        bad = [
            ("rm_rf", {}), ("read_file", {}), ("read_file", {"path": "/a", "extra": 1}), ("read_file", {"path": 5}),
            ("read_file", {"path": "a\x00b"}), ("read_file", {"path": "/a", "max_bytes": 10**9}),
            ("run_command", {"command": "x" * 5000}), ("run_command", {"command": "ls", "timeout": True}),
            ("write_file", {"path": "/a", "content": "x", "mode": "explode"}),
            ("mouse", {"action": "click"}), ("mouse", {"action": "scroll", "amount": 0}),
            ("keyboard", {}), ("keyboard", {"text": "a", "keys": "ctrl+c"}),
        ]
        for kind, args in bad:
            with self.assertRaises(DeviceError, msg=f"{kind} {args}"):
                devices.validate_args(kind, args)

    def test_summary_is_computed_from_the_real_arguments(self):
        summary, detail = devices.describe("run_command", {"command": "rm -rf /tmp/x", "timeout": 5})
        self.assertEqual(summary, "Lancer une commande")
        self.assertEqual(detail, "rm -rf /tmp/x")  # la commande exacte est toujours montrée à Brice
        self.assertIn("/etc/hosts", devices.describe("delete", {"path": "/etc/hosts"})[0])

    def test_what_is_approved_is_shown_in_full_never_a_harmless_head_with_a_hidden_tail(self):
        # Écriture et script ne sont acceptés que s'ils tiennent EN ENTIER dans ce que Brice voit.
        biggest = "é" * devices.MAX_SHOWN_WRITE
        self.assertEqual(devices.describe("write_file", devices.validate_args(
            "write_file", {"path": "/a", "content": biggest}))[1], biggest)
        code = "print(1)\n" * (devices.MAX_SHOWN_CODE // 9)
        self.assertEqual(devices.describe("run_python", devices.validate_args("run_python", {"code": code}))[1], code)
        with self.assertRaises(DeviceError):
            devices.validate_args("write_file", {"path": "/a", "content": biggest + "x"})
        with self.assertRaises(DeviceError):
            devices.validate_args("run_python", {"code": "x" * (devices.MAX_SHOWN_CODE + 1)})
        for kind, args in (("run_command", {"command": "x" * 4000}), ("keyboard", {"text": "y" * 2000})):
            clean = devices.validate_args(kind, args)
            self.assertEqual(len(devices.describe(kind, clean)[1]), len(next(iter(v for v in args.values() if isinstance(v, str)))))


class PairingTests(KiraTestCase):
    def test_code_is_single_use_and_stored_hashed(self):
        made = devices.create_pair_code()
        self.assertRegex(made["code"], r"^[A-HJKMNP-Z2-9]{4}-[A-HJKMNP-Z2-9]{4}$")
        raw = made["code"].replace("-", "")
        self.assertEqual(db.get_db().q("SELECT code_hash FROM pair_codes")[0]["code_hash"] == raw, False)
        paired = devices.pair(made["code"].lower(), "Mon PC", "windows", {"enabled": ["monitor"]}, "1.0")  # casse et tiret libres
        self.assertTrue(paired["token"].startswith("kdev_"))
        with self.assertRaises(DeviceError):
            devices.pair(made["code"], "Autre", "windows", {})
        with self.assertRaises(DeviceError):
            devices.pair("ZZZZ-ZZZZ", "Autre", "windows", {})

    def test_expired_code_is_refused(self):
        made = devices.create_pair_code()
        db.get_db().run("UPDATE pair_codes SET expires_at = '2000-01-01T00:00:00+00:00'")
        with self.assertRaises(DeviceError):
            devices.pair(made["code"], "PC", "linux", {})

    def test_token_is_never_stored_in_clear(self):
        paired = pair_device()
        row = db.get_db().q1("SELECT * FROM devices")
        self.assertNotIn(paired["token"], str(row))
        self.assertEqual(len(row["token_hash"]), 64)

    def test_authentication_scheme_is_separate_from_the_owner_token(self):
        paired = pair_device()
        self.assertEqual(devices.authenticate("Device " + paired["token"])["id"], paired["device_id"])
        for header in ("", "Device", "Device kdev_faux", "Bearer " + paired["token"], "Device " + paired["token"][:-1],
                       "Device " + auth.make_token()):
            self.assertIsNone(devices.authenticate(header), header)
        self.assertIsNone(auth.verify_token(paired["token"]))  # et un jeton d'appareil n'ouvre pas l'API du propriétaire

    def test_revocation_cuts_the_device_at_once(self):
        paired = pair_device()
        action = devices.request_action(paired["device_id"], "status", {})
        devices.revoke(paired["device_id"])
        self.assertIsNone(devices.authenticate("Device " + paired["token"]))
        self.assertEqual(devices.get_action(action["id"])["status"], "cancelled")
        with self.assertRaises(DeviceError):
            devices.request_action(paired["device_id"], "status", {})
        self.assertEqual(devices.list_devices(), [])


class PolicyTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        self.dev = pair_device()["device_id"]

    def mode(self, kind, args=None, tainted=False):
        args = {"path": "/x"} if args is None else args
        return devices.request_action(self.dev, kind, args, tainted=tainted)["status"]

    def test_default_levels(self):
        self.assertEqual(self.mode("status", {}), "queued")             # surveiller : libre
        self.assertEqual(self.mode("read_file"), "queued")              # lire : libre
        self.assertEqual(self.mode("write_file", {"path": "/x", "content": "a"}), "pending")
        self.assertEqual(self.mode("run_command", {"command": "ls"}), "pending")
        self.assertEqual(self.mode("screenshot", {}), "pending")
        self.assertEqual(self.mode("mouse", {"action": "click", "x": 1, "y": 1}), "pending")
        self.assertEqual(self.mode("delete"), "pending")

    def test_deletion_can_never_be_free(self):
        with self.assertRaises(DeviceError):
            devices.set_policy(self.dev, {"fs_delete": "auto"})
        db.get_db().update("devices", self.dev, {"policy": '{"fs_delete": "auto"}'})  # même une ligne trafiquée en base
        self.assertEqual(self.mode("delete"), "pending")

    def test_owner_can_loosen_or_forbid_per_category(self):
        devices.set_policy(self.dev, {"exec": "auto", "screen": "deny", "fs_read": "ask"})
        self.assertEqual(self.mode("run_command", {"command": "ls"}), "queued")
        self.assertEqual(self.mode("screenshot", {}), "denied")
        self.assertEqual(self.mode("read_file"), "pending")
        with self.assertRaises(DeviceError):
            devices.set_policy(self.dev, {"exec": "whatever"})
        with self.assertRaises(DeviceError):
            devices.set_policy(self.dev, {"teleport": "auto"})

    def test_machine_side_ceiling_wins_over_policy(self):
        low = pair_device("Vieux PC", enabled=["monitor", "fs_read"])["device_id"]
        devices.set_policy(low, {"exec": "auto"})
        act = devices.request_action(low, "run_command", {"command": "ls"})
        self.assertEqual(act["status"], "denied")
        self.assertIn("désactivé sur la machine", act["result"])

    def test_web_content_in_the_turn_downgrades_free_actions_to_ask(self):
        self.assertEqual(self.mode("read_file", tainted=True), "pending")
        self.assertEqual(self.mode("status", {}, tainted=True), "queued")  # l'état de la machine reste libre

    def test_pauses(self):
        devices.set_paused(self.dev, True)
        self.assertEqual(self.mode("status", {}), "denied")
        devices.set_paused(self.dev, False)
        self.assertEqual(self.mode("status", {}), "queued")
        pending = devices.request_action(self.dev, "run_command", {"command": "ls"})
        devices.set_global_pause(True)
        self.assertEqual(devices.get_action(pending["id"])["status"], "cancelled")
        self.assertEqual(self.mode("status", {}), "denied")
        devices.set_global_pause(False)
        self.assertEqual(self.mode("status", {}), "queued")

    def test_local_pause_reported_by_the_machine(self):
        devices.heartbeat(self.dev, {"enabled": ALL, "local_pause": True})
        self.assertEqual(self.mode("status", {}), "denied")


class FlowTests(KiraTestCase):
    def setUp(self):
        super().setUp()
        self.paired = pair_device()
        self.dev = self.paired["device_id"]
        self.device = devices.authenticate("Device " + self.paired["token"])

    def poll(self, **info):
        return devices.poll(self.device, {"enabled": ALL, **info}, wait=0)

    def test_approve_run_report(self):
        act = devices.request_action(self.dev, "run_command", {"command": "echo salut"})
        self.assertEqual(act["status"], "pending")
        self.assertIsNone(self.poll()["action"])  # rien ne part avant l'accord
        self.assertEqual(devices.pending_actions()[0]["id"], act["id"])
        devices.approve(act["id"])
        claimed = self.poll()["action"]
        self.assertEqual((claimed["id"], claimed["kind"], claimed["args"]["command"]), (act["id"], "run_command", "echo salut"))
        self.assertIsNone(self.poll()["action"])  # une action n'est remise qu'une fois
        done = devices.submit_result(self.device, act["id"], True, "salut\n")
        self.assertEqual((done["status"], done["result"], done["decided_by"]), ("done", "salut\n", "owner"))
        with self.assertRaises(DeviceError):
            devices.submit_result(self.device, act["id"], True, "deux fois")

    def test_free_action_goes_straight_through(self):
        act = devices.request_action(self.dev, "status", {})
        self.assertEqual(act["decided_by"], "policy")
        self.assertEqual(self.poll()["action"]["id"], act["id"])

    def test_deny_and_double_decision(self):
        act = devices.request_action(self.dev, "run_command", {"command": "ls"})
        self.assertEqual(devices.deny(act["id"])["status"], "denied")
        with self.assertRaises(DeviceError):
            devices.approve(act["id"])
        with self.assertRaises(DeviceError):
            devices.deny(act["id"])
        self.assertIsNone(self.poll()["action"])

    def test_pending_and_queued_requests_expire(self):
        a = devices.request_action(self.dev, "run_command", {"command": "ls"})
        b = devices.request_action(self.dev, "status", {})
        db.get_db().run("UPDATE device_actions SET created_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", [a["id"]])
        db.get_db().run("UPDATE device_actions SET decided_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", [b["id"]])
        devices.expire_stale()
        self.assertEqual((devices.get_action(a["id"])["status"], devices.get_action(b["id"])["status"]), ("expired", "expired"))
        with self.assertRaises(DeviceError):
            devices.approve(a["id"])

    def test_a_device_cannot_report_for_another_device_or_an_unclaimed_action(self):
        other = pair_device("Autre")
        other_dev = devices.authenticate("Device " + other["token"])
        act = devices.request_action(self.dev, "status", {})
        with self.assertRaises(DeviceError):
            devices.submit_result(other_dev, act["id"], True, "x")      # pas la sienne
        with self.assertRaises(DeviceError):
            devices.submit_result(self.device, act["id"], True, "x")    # pas encore remise : pas « running »

    def test_poll_reports_pause_and_revocation(self):
        devices.set_paused(self.dev, True)
        self.assertTrue(self.poll().get("paused"))
        devices.set_paused(self.dev, False)
        self.assertTrue(self.poll(local_pause=True).get("paused"))
        devices.revoke(self.dev)
        self.assertTrue(self.poll().get("revoked"))

    def test_heartbeat_updates_presence_and_caps(self):
        self.poll(roots=["/data"], system={"hostname": "box", "uptime": 5})
        d = devices.list_devices()[0]
        self.assertTrue(d["online"])
        self.assertEqual((d["roots"], d["info"]["hostname"]), (["/data"], "box"))
        db.get_db().update("devices", self.dev, {"last_seen": "2000-01-01T00:00:00+00:00"})
        self.assertFalse(devices.list_devices()[0]["online"])

    def test_images_are_validated_and_stored_privately(self):
        act = devices.request_action(self.dev, "screenshot", {})
        devices.approve(act["id"])
        self.poll()
        good = {"b64": base64.b64encode(PNG).decode()}
        bad = {"b64": base64.b64encode(b"<html>pas une image</html>").decode()}
        done = devices.submit_result(self.device, act["id"], True, "ok", [good, bad, {"b64": "%%%"}])
        self.assertEqual(len(done["files"]), 1)
        stored = files.get(done["files"][0].rsplit("/", 1)[-1])
        self.assertEqual((stored["mime"], stored["data"]), ("image/png", PNG))
        self.assertEqual(devices.result_images(done)[0]["mime"], "image/png")

    def test_wait_for_returns_when_the_owner_decides(self):
        act = devices.request_action(self.dev, "run_command", {"command": "ls"})
        start = time.time()
        self.assertEqual(devices.wait_for(act["id"], 0.6, step=0.1)["status"], "pending")
        self.assertGreaterEqual(time.time() - start, 0.5)
        devices.deny(act["id"])
        self.assertEqual(devices.wait_for(act["id"], 5, step=0.1)["status"], "denied")

    def test_everything_is_audited(self):
        act = devices.request_action(self.dev, "run_command", {"command": "ls"})
        devices.approve(act["id"])
        self.poll()
        devices.submit_result(self.device, act["id"], False, "boom")
        actions = [e["action"] for e in audit.recent(50)]
        for expected in ("device_paired", "device_action_requested", "device_action_approved", "device_action_done"):
            self.assertIn(expected, actions)

    def test_cleanup_and_prompt_summary(self):
        act = devices.request_action(self.dev, "status", {})
        db.get_db().run("UPDATE device_actions SET status = 'done', finished_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", [act["id"]])
        files.save("capture-1-old.png", "image/png", PNG)
        db.get_db().run("UPDATE files SET created_at = '2000-01-01T00:00:00+00:00'")
        self.assertEqual(devices.cleanup(), {"actions_deleted": 1, "results_erased": 0, "captures_deleted": 1})
        text = devices.prompt_summary()
        self.assertIn("PC-bureau", text)
        self.assertIn("Sur accord de Brice", text)

    def test_what_a_machine_sent_back_is_erased_after_a_week_but_the_log_stays(self):
        act = devices.request_action(self.dev, "read_file", {"path": "/home/b/KIRA/prive.txt"})
        self.poll()
        devices.submit_result(self.device, act["id"], True, "contenu très privé")
        week_ago = (time.time() - 8 * 86400)
        stamp = __import__("datetime").datetime.fromtimestamp(week_ago, __import__("datetime").timezone.utc).isoformat(timespec="seconds")
        db.get_db().run("UPDATE device_actions SET finished_at = ? WHERE id = ?", [stamp, act["id"]])
        self.assertEqual(devices.cleanup()["results_erased"], 1)
        kept = devices.get_action(act["id"])
        self.assertEqual(kept["result"], "(effacé)")
        self.assertIn("prive.txt", kept["summary"])  # on sait encore ce qui a été fait, pas ce qui a été lu
        self.assertNotIn("contenu très privé", str(db.get_db().q("SELECT * FROM device_actions")))
