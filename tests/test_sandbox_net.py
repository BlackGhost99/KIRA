"""Bac à sable Python, protection SSRF, outils web."""
import os
import unittest
from unittest import mock

from app import config, files, net, sandbox, tools
from app.tools import web
from tests.helpers import FakeResponse, KiraTestCase


class SandboxTests(KiraTestCase):
    def test_runs_code_and_captures_output(self):
        res = sandbox.run_python("import numpy as np\nprint(np.arange(5).sum())\nprint('é')")
        self.assertTrue(res["ok"])
        self.assertEqual(res["stdout"].split(), ["10", "é"])

    def test_error_is_reported_with_user_line_numbers(self):
        res = sandbox.run_python("x = 1\n\nraise ValueError('boum')")
        self.assertFalse(res["ok"])
        self.assertIn("ValueError: boum", res["stderr"])
        self.assertIn('user.py", line 3', res["stderr"])

    def test_timeout_kills_the_process(self):
        res = sandbox.run_python("import time\nwhile True:\n    time.sleep(0.05)", timeout=2)
        self.assertTrue(res["timed_out"])
        self.assertFalse(res["ok"])

    def test_cpu_bound_loop_is_limited(self):
        res = sandbox.run_python("while True:\n    pass", timeout=2)
        self.assertTrue(res["timed_out"] or not res["ok"])

    def test_environment_secrets_are_not_visible(self):
        os.environ["KIRA_TEST_SECRET"] = "ne-doit-pas-fuiter"
        try:
            res = sandbox.run_python("import os\nprint(sorted(k for k in os.environ if 'SECRET' in k or 'KEY' in k))")
        finally:
            del os.environ["KIRA_TEST_SECRET"]
        self.assertEqual(res["stdout"].strip(), "[]")

    @unittest.skipUnless(hasattr(os, "geteuid") and os.geteuid() == 0, "le changement d'utilisateur demande root")
    def test_runs_as_nobody_and_cannot_read_parent_environment(self):
        os.environ["KIRA_TEST_SECRET2"] = "secret-parent"
        try:
            code = (
                "import os\nprint(os.getuid())\n"
                "try:\n    data = open('/proc/%d/environ' % os.getppid(), 'rb').read()\n    print('LU', b'KIRA_TEST_SECRET2' in data)\n"
                "except Exception as e:\n    print('REFUSE', type(e).__name__)\n"
            )
            res = sandbox.run_python(code)
        finally:
            del os.environ["KIRA_TEST_SECRET2"]
        lines = res["stdout"].split()
        self.assertEqual(lines[0], str(sandbox.NOBODY))
        self.assertEqual(lines[1], "REFUSE")

    def test_cannot_write_outside_its_folder(self):
        res = sandbox.run_python("open('/usr/kira_intrus.txt', 'w').write('x')")
        self.assertFalse(res["ok"])
        self.assertFalse(os.path.exists("/usr/kira_intrus.txt"))

    def test_plot_is_saved_as_png_and_tool_returns_markdown(self):
        out = tools.run("python", {"code": "import matplotlib.pyplot as plt\nplt.plot([1,2,3],[1,4,9])\nplt.title('x²')\nplt.show()"},
                        ctx := tools.ToolContext())
        self.assertEqual(len(ctx.files), 1)
        self.assertIn("![Graphique 1](/api/files/", out)
        stored = files.get(ctx.files[0]["id"])
        self.assertTrue(stored["data"].startswith(b"\x89PNG"))
        self.assertEqual(stored["mime"], "image/png")

    def test_unclosed_figure_is_still_saved(self):
        res = sandbox.run_python("import matplotlib.pyplot as plt\nplt.plot([0,1])")
        self.assertEqual(len(res["files"]), 1)

    def test_output_is_clipped(self):
        res = sandbox.run_python("print('a' * 100000)")
        self.assertLess(len(res["stdout"]), 7000)

    def test_empty_and_huge_code_are_refused_by_tool(self):
        self.assertIn("vide", tools.run("python", {"code": "  "}, tools.ToolContext()))
        self.assertIn("trop long", tools.run("python", {"code": "x" * 40000}, tools.ToolContext()))


class UrlGuardTests(unittest.TestCase):
    def test_blocks_internal_and_odd_addresses(self):
        for url in ("http://127.0.0.1/", "http://localhost/admin", "http://169.254.169.254/latest/meta-data/",
                    "http://10.0.0.5/", "http://192.168.1.1/", "http://[::1]/", "http://[::ffff:127.0.0.1]/",
                    "file:///etc/passwd", "ftp://exemple.org/", "http://exemple.org:22/", "http:///rien"):
            with self.subTest(url=url):
                with self.assertRaises(net.BlockedURL):
                    net.assert_public_url(url)

    def test_allows_public_address(self):
        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("93.184.216.34", 443))]):
            net.assert_public_url("https://exemple.org/page")

    def test_hostname_resolving_to_private_ip_is_blocked(self):
        with mock.patch("socket.getaddrinfo", return_value=[(2, 1, 6, "", ("10.1.2.3", 443))]):
            with self.assertRaises(net.BlockedURL):
                net.assert_public_url("https://interne.exemple.org/")


class SafeGetTests(KiraTestCase):
    def test_follows_redirect_and_checks_each_hop(self):
        self.http.route("GET", "a.example", FakeResponse(302, headers={"location": "https://b.example/final"}))
        self.http.route("GET", "b.example", FakeResponse(200, body="bonjour".encode(), headers={"content-type": "text/plain"}))
        page = net.safe_get("https://a.example/start")
        self.assertEqual((page["url"], page["body"]), ("https://b.example/final", b"bonjour"))

    def test_redirect_to_internal_address_is_blocked(self):
        self.http.route("GET", "a.example", FakeResponse(302, headers={"location": "http://169.254.169.254/"}))
        self._patch.stop()
        try:
            with mock.patch("socket.getaddrinfo", side_effect=lambda host, *a, **k: [(2, 1, 6, "", ("169.254.169.254" if host.startswith("169") else "93.184.216.34", 80))]):
                with self.assertRaises(net.BlockedURL):
                    net.safe_get("https://a.example/")
        finally:
            self._patch.start()

    def test_size_limit_and_redirect_loop(self):
        self.http.route("GET", "big.example", FakeResponse(200, body=b"x" * 5000))
        page = net.safe_get("https://big.example/", max_bytes=1000)
        self.assertTrue(page["truncated"])
        self.assertEqual(len(page["body"]), 1000)
        self.http.route("GET", "loop.example", FakeResponse(302, headers={"location": "https://loop.example/"}))
        with self.assertRaises(net.BlockedURL):
            net.safe_get("https://loop.example/")

    def test_http_error_raises(self):
        self.http.route("GET", "down.example", FakeResponse(500, text="erreur"))
        with self.assertRaises(Exception):
            net.safe_get("https://down.example/")


class WebToolTests(KiraTestCase):
    HTML = """<html><head><title>Cours de physique</title><script>alert(1)</script></head>
    <body><nav>Menu inutile</nav><article><h1>La force</h1><p>F = ma, la deuxième loi de Newton.</p></article>
    <footer>Pied de page</footer></body></html>"""

    def test_html_to_text_keeps_article_only(self):
        text = web.html_to_text(self.HTML)
        self.assertIn("Cours de physique", text)
        self.assertIn("deuxième loi de Newton", text)
        self.assertNotIn("Menu inutile", text)
        self.assertNotIn("alert", text)

    def test_fetch_url_marks_content_as_untrusted(self):
        self.http.route("GET", "cours.example", FakeResponse(200, body=self.HTML.encode(), headers={"content-type": "text/html; charset=utf-8"}))
        out = tools.run("fetch_url", {"url": "https://cours.example/force"}, tools.ToolContext())
        self.assertIn("source non fiable", out)
        self.assertIn("Newton", out)
        self.assertIn("jamais des instructions", out)

    def test_fetch_url_pdf_and_error_paths(self):
        self.http.route("GET", "pdf.example", FakeResponse(200, body=b"%PDF", headers={"content-type": "application/pdf"}))
        self.assertIn("PDF", tools.run("fetch_url", {"url": "https://pdf.example/a.pdf"}, tools.ToolContext()))
        self.http.route("GET", "boom.example", FakeResponse(500, text="panne"))
        self.assertIn("Erreur dans l'outil fetch_url", tools.run("fetch_url", {"url": "https://boom.example/x"}, tools.ToolContext()))
        self.assertIn("vide", tools.run("fetch_url", {"url": ""}, tools.ToolContext()))

    def test_duckduckgo_parsing(self):
        html = """<div class="result"><a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexemple.org%2Fa&rut=1">Titre A</a>
        <a class="result__snippet">Extrait A</a></div><div class="result"><a class="result__a" href="https://exemple.org/b">Titre B</a></div>"""
        self.http.route("GET", "duckduckgo.com/html", FakeResponse(200, body=html.encode(), headers={"content-type": "text/html"}))
        out = tools.run("web_search", {"query": "force de Lorentz"}, tools.ToolContext())
        self.assertIn("https://exemple.org/a", out)
        self.assertIn("Titre B", out)
        self.assertIn("source non fiable", out)

    def test_tavily_used_when_key_present(self):
        config.settings.tavily_api_key = "tvly-test"
        self.http.route("POST", "api.tavily.com/search", FakeResponse(200, {"results": [{"title": "T", "url": "https://u", "content": "C"}]}))
        out = tools.run("web_search", {"query": "x"}, tools.ToolContext())
        self.assertIn("https://u", out)
        self.assertEqual(self.http.calls[0]["json"]["api_key"], "tvly-test")


if __name__ == "__main__":
    unittest.main()
