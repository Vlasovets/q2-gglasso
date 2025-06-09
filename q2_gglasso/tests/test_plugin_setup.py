"""Tests for the q2-gglasso plugin setup functionality.

This module tests the basic plugin setup functionality of the q2-gglasso plugin,
ensuring that the plugin is correctly registered with the QIIME 2 framework.
"""

import unittest

from q2_gglasso.plugin_setup import plugin


class PluginSetupTests(unittest.TestCase):
    """Test case for the q2-gglasso plugin setup.

    This test class verifies that the q2-gglasso plugin is correctly set up
    and registered with the QIIME 2 framework.
    """

    def test_plugin_setup(self):
        """Test that the plugin name is correctly set to 'gglasso'."""
        self.assertEqual(plugin.name, "gglasso")


if __name__ == "__main__":
    unittest.main()
