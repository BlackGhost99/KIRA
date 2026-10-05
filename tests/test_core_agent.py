"""L'agent « cœur » (core/kira_core.py) : garde-fous locaux, actions, corbeille, ligne de commande, et un vrai aller-retour
avec le serveur (uvicorn dans un thread)."""
import base64
import contextlib
import importlib.util
import io
import os
import socket
import stat
import sys
import struct
import threading
import time
import unittest
import zlib
from pathlib import Path

from app import auth, db, devices
from tests.helpers import KiraTestCase

SPEC = importlib.util.spec_from_file_location("kira_core", Path(__file__).resolve().parents[1] / "core" / "kira_core.py")
kc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kc)

ALL = list(kc.CATEGORIES)


def make_png(w, h):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x10\x20\x30" * w for _ in range(h))
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


class AgentCase(KiraTestCase):
    """Machine fictive : HOME, dossier du cœur et dossier autorisé sont jetables."""

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self._tmp, "home")
        self.core_home = os.path.join(self._tmp, "home", ".kira-core")
        self.root = os.path.join(self._tmp, "home", "KIRA")
        os.makedirs(self.root)
        for key, value in (("HOME", self.home), ("KIRA_CORE_HOME", self.core_home), ("USERPROFILE", self.home)):
            self.addCleanup(self._restore_env, key, os.environ.get(key))
            os.environ[key] = value
        self.cfg = kc.Config()
        self.cfg.data.update({"enabled": list(ALL), "roots": [self.root]})
        self.core = kc.Core(self.cfg)

    @staticmethod
    def _restore_env(key, old):
        if old is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old

    def run_ok(self, kind, **args):
        r = self.core.execute(kind, args)
        self.assertTrue(r.ok, r.output)
        return r.output

    def run_refused(self, kind, **args):
        r = self.core.execute(kind, args)
        self.assertFalse(r.ok, f"{kind} {args} aurait dû être refusé : {r.output}")
        return r.output

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def write(self, rel, text="x"):
        full = self.path(rel)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        Path(full).write_text(text, encoding="utf-8")
        return full


class ParityTests(unittest.TestCase):
    def test_action_catalog_matches_the_server(self):
        self.assertEqual(kc.KINDS, devices.KINDS)
        self.assertEqual(kc.CATEGORIES, devices.CATEGORIES)

    def test_validators_agree_on_a_corpus(self):
        corpus = [
            ("read_file", {"path": "/a"}), ("read_file", {}), ("read_file", {"path": "/a", "max_bytes": 10 ** 9}),
            ("read_file", {"path": "/a", "x": 1}), ("run_command", {"command": "ls", "timeout": True}),
            ("write_file", {"path": "/a", "content": "c", "mode": "nope"}), ("mouse", {"action": "click"}),
            ("mouse", {"action": "scroll", "amount": 0}), ("mouse", {"action": "scroll", "amount": 3}),
            ("keyboard", {}), ("keyboard", {"text": "a", "keys": "b"}), ("keyboard", {"keys": "ctrl+s"}),
            ("status", {}), ("status", {"a": 1}), ("processes", {"limit": 0}), ("nope", {}),
            ("write_file", {"path": "/a", "content": "x" * kc.MAX_SHOWN_WRITE}),
            ("write_file", {"path": "/a", "content": "x" * (kc.MAX_SHOWN_WRITE + 1)}),
            ("run_python", {"code": "x" * kc.MAX_SHOWN_CODE}), ("run_python", {"code": "x" * (kc.MAX_SHOWN_CODE + 1)}),
        ]
        for kind, args in corpus:
            try:
                theirs = devices.validate_args(kind, args)
            except devices.DeviceError:
                theirs = None
            try:
                mine = kc.validate_args(kind, args)
            except kc.Refused:
                mine = None
            self.assertEqual(mine, theirs, f"{kind} {args}")


class GuardTests(AgentCase):
    def test_inside_roots_only(self):
        self.write("a.txt", "bonjour")
        self.assertIn("bonjour", self.run_ok("read_file", path=self.path("a.txt")))
        self.assertIn("bonjour", self.run_ok("read_file", path="a.txt"))  # relatif = dans le premier dossier autorisé
        self.assertIn("Hors des dossiers", self.run_refused("read_file", path="/etc/hostname"))
        self.assertIn("Hors des dossiers", self.run_refused("list_dir", path=os.path.dirname(self.root)))
        self.assertIn("Hors des dossiers", self.run_refused("read_file", path=self.path("..", "..", "etc", "passwd")))
        self.assertIn("Hors des dossiers", self.run_refused("read_file", path=self.root + "-voisin/x"))  # préfixe de nom, pas sous-dossier

    def test_symlinks_cannot_escape(self):
        outside = os.path.join(self._tmp, "dehors")
        os.makedirs(outside)
        Path(outside, "secret.txt").write_text("trop privé")
        os.symlink(outside, self.path("lien"))
        os.symlink(os.path.join(outside, "secret.txt"), self.path("lien.txt"))
        self.assertIn("Hors des dossiers", self.run_refused("read_file", path=self.path("lien", "secret.txt")))
        self.assertIn("Hors des dossiers", self.run_refused("read_file", path=self.path("lien.txt")))
        self.assertIn("Hors des dossiers", self.run_refused("write_file", path=self.path("lien", "nouveau.txt"), content="x"))
        self.assertFalse(os.path.exists(os.path.join(outside, "nouveau.txt")))

    def test_sensitive_files_are_always_refused_even_inside_a_root(self):
        for rel in (".ssh/id_rsa", ".env", ".env.production", "cle.pem", "sub/.aws/credentials", "id_ed25519", ".git-credentials",
                    "coffre.kdbx", ".gnupg/pubring.kbx"):
            self.write(rel, "SECRET")
            self.assertIn("sensible", self.run_refused("read_file", path=self.path(rel)), rel)
        listing = self.run_ok("list_dir", path=self.root)
        self.assertNotIn("id_ed25519", listing)
        self.assertNotIn(".env", listing)
        self.assertEqual(self.run_ok("search_files", path=self.root, query="SECRET", contents=True), "Aucun résultat.")

    def test_the_core_own_folder_is_forbidden_even_when_the_root_is_home(self):
        self.cfg.data["roots"] = [self.home]
        self.core.reload = lambda: None
        self.core.guard = kc.Guard([self.home])
        os.makedirs(self.core_home, exist_ok=True)
        self.cfg.save()
        for rel in ("config.json", "PAUSE", "kira_core.py", "trash"):
            self.assertIn("KIRA Core", self.run_refused("write_file", path=os.path.join(self.core_home, rel), content="piraté", mode="overwrite"))
        self.assertIn("KIRA Core", self.run_refused("read_file", path=os.path.join(self.core_home, "config.json")))

    def test_a_root_itself_cannot_be_deleted_moved_or_replaced(self):
        self.assertIn("racine", self.run_refused("delete", path=self.root))
        self.assertIn("racine", self.run_refused("move", src=self.root, dst=self.path("sous")))
        self.assertTrue(os.path.isdir(self.root))

    def test_no_root_means_no_file_access(self):
        self.cfg.data["roots"] = []
        self.core.guard = kc.Guard([])
        self.assertIn("Aucun dossier", self.run_refused("list_dir", path="/tmp"))


class CapabilityTests(AgentCase):
    def test_disabled_categories_are_refused_locally_whatever_the_server_says(self):
        self.cfg.data["enabled"] = ["monitor", "fs_read"]
        out = self.run_refused("run_command", command="echo salut")
        self.assertIn("allow exec", out)
        self.assertIn("n'est pas activé", self.run_refused("write_file", path=self.path("a"), content="x"))
        self.assertIn("n'est pas activé", self.run_refused("screenshot"))
        self.assertIn("n'est pas activé", self.run_refused("mouse", action="click", x=1, y=1))
        self.assertFalse(os.path.exists(self.path("a")))

    def test_writing_programs_or_autorun_files_needs_exec_on_this_machine(self):
        # Sans « exec » local, déposer un script reviendrait à contourner ce plafond.
        self.cfg.data["enabled"] = ["monitor", "fs_read", "fs_write", "fs_delete"]
        for rel in ("outil.sh", "run.BAT", "setup.ps1", "x.desktop", "mon.service", ".bashrc", ".git/hooks/pre-commit",
                    "autostart/lance.txt", "lib/evil.pth", "sitecustomize.py"):
            out = self.run_refused("write_file", path=self.path(rel), content="echo piraté")
            self.assertIn("allow exec", out, rel)
            self.assertFalse(os.path.exists(self.path(rel)), rel)
        self.write("a.txt", "x")
        self.assertIn("allow exec", self.run_refused("move", src=self.path("a.txt"), dst=self.path("a.sh")))
        self.assertIn("allow exec", self.run_refused("move", src=self.path("a.txt"), dst=self.path(".git", "hooks", "post-merge")))
        self.assertTrue(os.path.exists(self.path("a.txt")))
        for rel in ("notes.md", "calcul.py", "data.json", "cours/physique.txt"):  # le travail ordinaire reste possible
            self.run_ok("write_file", path=self.path(rel), content="ok")
        self.cfg.data["enabled"].append("exec")  # Brice l'active lui-même sur la machine : plus de blocage local
        self.run_ok("write_file", path=self.path("outil.sh"), content="echo salut")

    def test_local_pause_blocks_everything_and_resume_restores(self):
        kc.set_paused(True)
        self.assertIn("pause locale", self.run_refused("status"))
        kc.set_paused(False)
        self.run_ok("status")

    def test_toggle_via_cli_takes_effect_in_a_running_core(self):
        self.cfg.data["enabled"] = ["monitor"]
        self.cfg.save()
        self.core.reload()
        self.run_refused("list_dir", path=self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(kc.main(["allow", "fs_read"]), 0)
        self.core.reload()
        self.run_ok("list_dir", path=self.root)
        with contextlib.redirect_stdout(io.StringIO()):
            kc.main(["deny", "fs_read"])
        self.core.reload()
        self.run_refused("list_dir", path=self.root)

    def test_bad_arguments_are_refused_by_the_machine_too(self):
        self.assertIn("inconnu", self.run_refused("rm_rf"))
        self.run_refused("read_file", path=self.path("a"), extra=1)
        self.run_refused("run_command", command="echo", timeout=99999)


class FileActionTests(AgentCase):
    def test_list_read_search(self):
        self.write("cours/physique.md", "# Énergie\nL'énergie se conserve.\n")
        self.write("cours/maths.txt", "dérivées")
        out = self.run_ok("list_dir", path=self.path("cours"))
        self.assertIn("physique.md", out)
        self.assertIn("maths.txt", out)
        self.assertIn("L'énergie se conserve", self.run_ok("read_file", path=self.path("cours", "physique.md")))
        first = self.run_ok("read_file", path=self.path("cours", "physique.md"), max_bytes=5)
        self.assertIn("avec offset=5", first)
        rest = self.run_ok("read_file", path=self.path("cours", "physique.md"), max_bytes=1000, offset=5)  # lecture par tranches
        self.assertIn("L'énergie se conserve", rest)
        self.assertNotIn("avec offset", rest)
        self.assertIn("physique.md", self.run_ok("search_files", path=self.root, query="PHYS"))
        hit = self.run_ok("search_files", path=self.root, query="conserve", contents=True)
        self.assertIn("physique.md:2:", hit)
        self.assertEqual(self.run_ok("search_files", path=self.root, query="introuvable"), "Aucun résultat.")

    def test_binary_files_are_not_dumped(self):
        Path(self.path("img.bin")).write_bytes(b"\x00\x01\x02" * 100)
        self.assertIn("binaire", self.run_ok("read_file", path=self.path("img.bin")))

    def test_write_modes_and_backup_on_overwrite(self):
        target = self.path("notes", "idee.txt")
        self.run_ok("write_file", path=target, content="v1")
        self.assertEqual(Path(target).read_text(), "v1")
        self.assertIn("existe déjà", self.run_refused("write_file", path=target, content="v2"))
        self.assertEqual(Path(target).read_text(), "v1")
        self.run_ok("write_file", path=target, content="\nv1bis", mode="append")
        self.assertEqual(Path(target).read_text(), "v1\nv1bis")
        out = self.run_ok("write_file", path=target, content="v2", mode="overwrite")
        self.assertIn("corbeille", out)
        self.assertEqual(Path(target).read_text(), "v2")
        saved = self.core.trash.entries()
        self.assertEqual(len(saved), 1)
        self.assertEqual(Path(self.core_home, "trash", saved[0]["id"], "data").read_text(), "v1\nv1bis")
        self.assertEqual([n for n in os.listdir(os.path.dirname(target)) if n.startswith(".kira-tmp")], [])

    def test_make_dir_and_move(self):
        self.run_ok("make_dir", path=self.path("a", "b"))
        self.write("a/f.txt", "x")
        self.run_ok("move", src=self.path("a", "f.txt"), dst=self.path("a", "b", "g.txt"))
        self.assertTrue(os.path.exists(self.path("a", "b", "g.txt")))
        self.write("a/h.txt", "y")
        self.assertIn("existe déjà", self.run_refused("move", src=self.path("a", "h.txt"), dst=self.path("a", "b", "g.txt")))
        self.assertIn("dans lui-même", self.run_refused("move", src=self.path("a"), dst=self.path("a", "b", "c")))

    def test_delete_goes_to_the_trash_and_can_be_restored(self):
        full = self.write("vieux/doc.txt", "précieux")
        out = self.run_ok("delete", path=self.path("vieux"))
        self.assertIn("corbeille", out)
        self.assertFalse(os.path.exists(self.path("vieux")))
        entry = self.core.trash.entries()[0]
        self.assertEqual(entry["original"], os.path.realpath(self.path("vieux")))
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(kc.main(["trash", "list"]), 0)
        self.assertIn(entry["id"], buf.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(kc.main(["trash", "restore", entry["id"]]), 0)
        self.assertEqual(Path(full).read_text(), "précieux")
        self.assertEqual(self.core.trash.entries(), [])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(kc.main(["trash", "restore", "../../etc"]), 2)  # identifiant invalide

    def test_old_trash_is_purged(self):
        self.write("a.txt")
        self.run_ok("delete", path=self.path("a.txt"))
        entry = os.path.join(self.core_home, "trash", self.core.trash.entries()[0]["id"])
        old = time.time() - 40 * 86400
        os.utime(entry, (old, old))
        self.core.trash.purge()
        self.assertEqual(self.core.trash.entries(), [])


@unittest.skipIf(os.name == "nt", "commandes POSIX")
class ExecTests(AgentCase):
    def test_command_output_and_exit_code(self):
        out = self.run_ok("run_command", command="echo bonjour && pwd")
        self.assertIn("Code de sortie : 0", out)
        self.assertIn("bonjour", out)
        self.assertIn(os.path.realpath(self.root), out)  # travaille dans le dossier autorisé par défaut
        failed = self.run_refused("run_command", command="echo non >&2; exit 3")
        self.assertIn("Code de sortie : 3", failed)
        self.assertIn("non", failed)

    def test_cwd_must_be_inside_roots(self):
        self.assertIn("Hors des dossiers", self.run_refused("run_command", command="ls", cwd="/etc"))

    def test_timeout_kills_the_whole_process_group(self):
        marker = self.path("vivant.txt")
        started = time.time()
        out = self.run_refused("run_command", command=f"(sleep 4; echo x > '{marker}') & sleep 30", timeout=1)
        self.assertIn("Délai de 1 s dépassé", out)
        self.assertLess(time.time() - started, 8)
        time.sleep(4.5)
        self.assertFalse(os.path.exists(marker), "le processus enfant a survécu")

    def test_pause_stops_a_running_command(self):
        threading.Timer(0.6, self.core.abort.set).start()
        started = time.time()
        out = self.run_refused("run_command", command="sleep 30", timeout=60)
        self.assertIn("mis en pause", out)
        self.assertLess(time.time() - started, 6)

    def test_huge_output_is_clipped(self):
        out = self.run_ok("run_command", command="python3 -c \"print('a'*200000)\"")
        self.assertLess(len(out), kc.MAX_OUTPUT + 200)
        self.assertIn("caractères omis", out)

    def test_run_python(self):
        out = self.run_ok("run_python", code="print(6 * 7)")
        self.assertIn("42", out)
        self.assertIn("Code de sortie : 2", self.run_refused("run_python", code="raise SystemExit(2)"))

    def test_open_refuses_programs_and_odd_schemes(self):
        for target in ("file:///etc/passwd", "javascript:alert(1)", "ms-msdt:/id", "ftp://x/y", "smb://h/s"):
            self.assertIn("Seuls les liens http", self.run_refused("open", target=target), target)
        self.write("outil.sh", "#!/bin/sh\necho x")
        os.chmod(self.path("outil.sh"), 0o755)
        self.write("setup.exe", "MZ")
        self.assertIn("programme", self.run_refused("open", target=self.path("outil.sh")))
        self.assertIn("programme", self.run_refused("open", target=self.path("setup.exe")))
        self.assertIn("Hors des dossiers", self.run_refused("open", target="/etc/hostname"))


class MonitorTests(AgentCase):
    def test_status_and_processes(self):
        out = self.run_ok("status")
        self.assertIn("Machine :", out)
        self.assertIn("Dossiers autorisés", out)
        procs = self.run_ok("processes", limit=5)
        self.assertIn("PID", procs)
        self.assertLessEqual(len(procs.splitlines()), 6)

    def test_info_sent_to_the_server_lists_ceilings_and_no_secret(self):
        self.cfg.data["token"] = "kdev_secret"
        info = self.core.info()
        self.assertEqual(info["enabled"], ALL)
        self.assertEqual(info["roots"], [self.root])
        self.assertNotIn("kdev_secret", str(info))


class ScreenAndInputTests(AgentCase):
    def test_screenshot_without_backend_says_what_to_install(self):
        self.core._screenshot_backend = lambda: ""
        self.assertIn("pip install mss", self.run_refused("screenshot"))

    def test_screenshot_returns_a_checked_image_and_remembers_the_scale(self):
        self.core._screenshot_backend = lambda: "fake"
        self.core._grab = lambda backend: make_png(1920, 1080)
        r = self.core.execute("screenshot", {})
        self.assertTrue(r.ok, r.output)
        expected = (1280, 720) if kc._has_module("PIL") else (1920, 1080)  # Pillow réduit l'image avant l'envoi
        self.assertEqual(self.core.last_shot["sent"], expected)
        self.assertEqual(len(r.images), 1)
        self.assertTrue(base64.b64decode(r.images[0]["b64"]).startswith(b"\x89PNG"))

    def test_mouse_needs_a_screenshot_first_and_converts_coordinates(self):
        calls = []
        self.core._input_backend = lambda: "pyautogui"
        self.core._screen_size = lambda: (2560, 1440)

        class Fake:
            FAILSAFE = True
            PAUSE = 0

            def click(self, x, y):
                calls.append(("click", x, y))

        sys.modules["pyautogui"] = Fake()
        self.addCleanup(sys.modules.pop, "pyautogui", None)
        self.assertIn("Prends d'abord une capture", self.run_refused("mouse", action="click", x=640, y=360))
        self.core.last_shot = {"sent": (1280, 720)}
        self.run_ok("mouse", action="click", x=640, y=360)
        self.assertEqual(calls, [("click", 1280, 720)])  # image 1280×720 -> écran 2560×1440
        self.run_ok("mouse", action="click", x=1000, y=1000, space="screen")
        self.assertEqual(calls[-1], ("click", 1000, 1000))
        self.run_ok("mouse", action="click", x=19999, y=5, space="screen")
        self.assertLessEqual(calls[-1][1], 2559)

    def test_failsafe_pauses_the_machine(self):
        self.core._input_backend = lambda: "pyautogui"
        self.core._screen_size = lambda: (100, 100)

        class FailSafeException(Exception):
            pass

        class Fake:
            def click(self, x, y):
                raise FailSafeException()

        sys.modules["pyautogui"] = Fake()
        self.addCleanup(sys.modules.pop, "pyautogui", None)
        out = self.run_refused("mouse", action="click", x=1, y=1, space="screen")
        self.assertIn("Arrêt d'urgence", out)
        self.assertTrue(kc.is_paused())

    def test_input_is_rate_limited(self):
        self.core._input_backend = lambda: "pyautogui"

        class Fake:
            FAILSAFE = True

            def hotkey(self, *keys):
                pass

        sys.modules["pyautogui"] = Fake()
        self.addCleanup(sys.modules.pop, "pyautogui", None)
        stamp = time.time()
        self.core._input_times.extend([stamp] * 60)
        self.assertIn("Trop d'actions", self.run_refused("keyboard", keys="ctrl+s"))

    def test_bad_key_names_are_refused(self):
        self.core._input_backend = lambda: "pyautogui"
        self.assertIn("Touches invalides", self.run_refused("keyboard", keys="ctrl+;rm -rf"))


class CliTests(AgentCase):
    def test_http_to_a_remote_host_is_refused(self):
        for bad in ("http://kira.example.org", "ftp://x", "kira.example.org"):
            with self.assertRaises(kc.Refused):
                kc.check_server_url(bad)
        self.assertEqual(kc.check_server_url("https://kira.onrender.com/"), "https://kira.onrender.com")
        self.assertEqual(kc.check_server_url("http://127.0.0.1:8000"), "http://127.0.0.1:8000")

    def test_config_is_private(self):
        self.cfg.data["token"] = "kdev_x"
        self.cfg.data["server"] = "https://x"
        self.cfg.save()
        if os.name != "nt":
            self.assertEqual(stat.S_IMODE(os.stat(self.cfg.path).st_mode), 0o600)

    def test_roots_command(self):
        extra = os.path.join(self._tmp, "autre")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            kc.main(["roots", "add", extra])
            kc.main(["roots", "list"])
        self.assertIn(extra, buf.getvalue())
        self.assertIn(extra, kc.Config().roots)
        with contextlib.redirect_stdout(io.StringIO()):
            kc.main(["roots", "remove", extra])
        self.assertNotIn(extra, kc.Config().roots)

    def test_pause_resume_commands(self):
        with contextlib.redirect_stdout(io.StringIO()):
            kc.main(["pause"])
            self.assertTrue(kc.is_paused())
            kc.main(["resume"])
        self.assertFalse(kc.is_paused())

    def test_autostart_files_are_well_formed(self):
        unit = kc.render_systemd_unit("/usr/bin/python3", "/home/b/My Files/kira_core.py")
        self.assertIn("ExecStart=/usr/bin/python3 '/home/b/My Files/kira_core.py' run", unit)
        self.assertIn("RestartPreventExitStatus=3", unit)  # jeton retiré -> on ne boucle pas
        plist = kc.render_launchd_plist("/usr/bin/python3", "/Users/b/.kira-core/kira_core.py", "/tmp/l.log")
        self.assertIn("<string>run</string>", plist)
        self.assertIn("com.kira.core", plist)
        boot = kc.render_termux_boot("/usr/bin/python", "/data/x/kira_core.py")
        self.assertIn("termux-wake-lock", boot)
        self.assertTrue(kc.windows_task_command("C:\\Py\\python.exe", "C:\\k\\kira_core.py").endswith('"C:\\k\\kira_core.py" run'))

    def test_agent_is_standard_library_only(self):
        source = Path(kc.__file__).read_text(encoding="utf-8")
        import ast
        allowed = set(sys.stdlib_module_names) | {"mss", "PIL", "psutil", "pyautogui"}
        for node in ast.walk(ast.parse(source)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for n in names:
                self.assertIn(n, allowed, f"dépendance non autorisée : {n}")


# ---------------------------------------------------------------------------- aller-retour avec le vrai serveur
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class EndToEndTests(AgentCase):
    def setUp(self):
        super().setUp()
        for key in ("NO_PROXY", "no_proxy"):
            self.addCleanup(self._restore_env, key, os.environ.get(key))
            os.environ[key] = "127.0.0.1,localhost"  # le serveur de test est local : jamais par le mandataire du bac à sable

    def start_server(self):
        import uvicorn
        from app.main import app

        port = free_port()
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="on"))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.1)
        self.assertTrue(server.started, "le serveur de test ne démarre pas")

        def stop():
            server.should_exit = True
            thread.join(timeout=15)

        self.addCleanup(stop)
        return f"http://127.0.0.1:{port}"

    def start_agent(self, url, *extra):
        code = devices.create_pair_code()["code"]
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertEqual(kc.main(["pair", "--server", url, "--code", code, "--name", "Machine test", "--root", self.root, *extra]), 0)
        self.assertIn("Appairé", buf.getvalue())
        cfg = kc.Config()
        self.assertTrue(cfg.paired)
        runner = kc.Runner(kc.Core(cfg), kc.logging.getLogger("test-core"))
        result = {}
        thread = threading.Thread(target=lambda: result.update(code=runner.run()), daemon=True)
        thread.start()

        def stop():
            runner.stop.set()
            if cfg.data.get("device_id") and devices.get_device(cfg.data["device_id"]):
                devices.revoke(cfg.data["device_id"])  # réveille la scrutation en cours
            thread.join(timeout=30)

        self.addCleanup(stop)
        return cfg, runner, thread, result

    def test_pairing_reading_writing_and_approval_over_real_http(self):
        url = self.start_server()
        cfg, runner, thread, result = self.start_agent(url)
        did = cfg.data["device_id"]
        self.write("lisez-moi.txt", "bonjour depuis la machine")

        # 1) lecture : libre par défaut
        act = devices.request_action(did, "read_file", {"path": self.path("lisez-moi.txt")})
        self.assertEqual(act["status"], "queued")
        done = devices.wait_for(act["id"], 20)
        self.assertEqual(done["status"], "done", done)
        self.assertIn("bonjour depuis la machine", done["result"])

        # 2) écriture : attend ton accord, rien ne se passe avant
        w = devices.request_action(did, "write_file", {"path": self.path("nouveau.txt"), "content": "écrit par KIRA"})
        self.assertEqual(w["status"], "pending")
        time.sleep(1.5)
        self.assertFalse(os.path.exists(self.path("nouveau.txt")))
        devices.approve(w["id"])
        done = devices.wait_for(w["id"], 20)
        self.assertEqual(done["status"], "done", done)
        self.assertEqual(Path(self.path("nouveau.txt")).read_text(encoding="utf-8"), "écrit par KIRA")

        # 3) un refus ne part jamais vers la machine
        d = devices.request_action(did, "write_file", {"path": self.path("refuse.txt"), "content": "non"})
        devices.deny(d["id"])
        time.sleep(1.2)
        self.assertFalse(os.path.exists(self.path("refuse.txt")))

        # 4) la machine garde ses plafonds : « exec » n'a pas été activé ici, le serveur le sait et refuse d'emblée
        e = devices.request_action(did, "run_command", {"command": "echo piraté"})
        self.assertEqual(e["status"], "denied")
        self.assertIn("désactivé sur la machine", e["result"])

        # 5) fichier sensible : la machine refuse même si Brice avait accepté
        self.write(".env", "TOKEN=abc")
        s = devices.request_action(did, "read_file", {"path": self.path(".env")})
        done = devices.wait_for(s["id"], 20)
        self.assertEqual(done["status"], "failed")
        self.assertIn("sensible", done["result"])
        self.assertNotIn("TOKEN=abc", done["result"])

        # 6) la machine apparaît en ligne avec ses plafonds
        shown = devices.list_devices()[0]
        self.assertTrue(shown["online"])
        self.assertEqual(shown["roots"], [self.root])
        self.assertNotIn("exec", shown["enabled"])

    def test_full_install_flag_and_local_allow_are_seen_by_the_server(self):
        url = self.start_server()
        cfg, runner, thread, result = self.start_agent(url, "--full")
        did = cfg.data["device_id"]
        act = devices.request_action(did, "run_command", {"command": "echo salut"})
        self.assertEqual(act["status"], "pending")  # activé localement, mais toujours sur accord de Brice
        devices.approve(act["id"])
        done = devices.wait_for(act["id"], 20)
        self.assertEqual(done["status"], "done", done)
        self.assertIn("salut", done["result"])

    def test_revoking_stops_the_agent_for_good(self):
        url = self.start_server()
        cfg, runner, thread, result = self.start_agent(url)
        time.sleep(1)
        devices.revoke(cfg.data["device_id"])
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(result.get("code"), 3)

    def test_global_pause_stops_new_work_but_not_the_connection(self):
        url = self.start_server()
        cfg, runner, thread, result = self.start_agent(url)
        did = cfg.data["device_id"]
        devices.set_global_pause(True)
        self.assertEqual(devices.request_action(did, "status", {})["status"], "denied")
        devices.set_global_pause(False)
        act = devices.request_action(did, "status", {})
        done = devices.wait_for(act["id"], 20)
        self.assertEqual(done["status"], "done", done)


if __name__ == "__main__":
    unittest.main()
