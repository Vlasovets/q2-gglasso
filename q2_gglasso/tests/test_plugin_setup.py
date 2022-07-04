import unittest

from q2_gglasso.plugin_setup import plugin


class PluginSetupTests(unittest.TestCase):

    def test_plugin_setup(self):
        self.assertEqual(plugin.name, 'gglasso')

if __name__ == '__main__':
    unittest.main()