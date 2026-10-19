"""Contrats réellement portables, exécutés sur Linux, Windows et macOS en CI."""
import os
from pathlib import Path
import tempfile
import unittest

from tests.test_core_agent import kc


class PortableCoreTests(unittest.TestCase):
    def test_unicode_file_lifecycle_and_root_boundary(self):
        with tempfile.TemporaryDirectory(prefix='kira_portable_') as folder:
            root = Path(folder) / 'Données autorisées'
            root.mkdir()
            config_home = Path(folder) / 'configuration'
            old = os.environ.get('KIRA_CORE_HOME')
            os.environ['KIRA_CORE_HOME'] = str(config_home)
            try:
                cfg = kc.Config()
                cfg.data.update({'enabled':['monitor','fs_read','fs_write','fs_delete'], 'roots':[str(root)]})
                core = kc.Core(cfg)
                path = root / 'résumé.txt'
                result = core.execute('write_file', {'path':str(path),'content':'Bonjour KIRA — Orion'})
                self.assertTrue(result.ok, result.output)
                result = core.execute('read_file', {'path':str(path)})
                self.assertTrue(result.ok, result.output)
                self.assertIn('Orion', result.output)
                result = core.execute('read_file', {'path':str(Path(folder) / 'hors-racine.txt')})
                self.assertFalse(result.ok)
            finally:
                if old is None:
                    os.environ.pop('KIRA_CORE_HOME',None)
                else:
                    os.environ['KIRA_CORE_HOME']=old

    def test_platform_is_reported_without_claiming_unsupported_capture(self):
        self.assertIn(kc.platform_name(), ('linux','windows','mac','android'))
