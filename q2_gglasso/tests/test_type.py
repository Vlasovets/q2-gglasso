"""Tests for the semantic types and formats in the q2-gglasso plugin.

This module tests the registration of semantic types and formats in the q2-gglasso plugin,
ensuring that they're properly registered and linked in the QIIME 2 framework.
"""

import pytest
from qiime2.plugin.testing import TestPluginBase

from q2_gglasso._type import PairwiseFeatureData
from q2_gglasso._format import PairwiseFeatureDataDirectoryFormat


class TestTypes(TestPluginBase):
    """Test case for semantic type registration in the q2-gglasso plugin.

    This test class verifies that semantic types are properly registered with
    the QIIME 2 framework and linked to their corresponding formats.
    """

    package = 'q2-gglasso'

    def test_type_registration(self):
        """Test that PairwiseFeatureData is registered as a semantic type."""
        self.assertRegisteredSemanticType(PairwiseFeatureData)

    def test(self):
        """Test that PairwiseFeatureData is registered with its directory format."""
        self.assertSemanticTypeRegisteredToFormat(PairwiseFeatureData, PairwiseFeatureDataDirectoryFormat)


if __name__ == '__main__':
    TestTypes.main()
