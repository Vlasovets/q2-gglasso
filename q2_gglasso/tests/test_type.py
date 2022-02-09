import pytest
from qiime2.plugin.testing import TestPluginBase

from q2_gglasso._type import PairwiseFeatureData
from q2_gglasso._format import PairwiseFeatureDataDirectoryFormat


class TestTypes(TestPluginBase):
    package = 'q2_gglasso'

    def test_type_registration(self):
        self.assertRegisteredSemanticType(PairwiseFeatureData)

    def test(self):
        self.assertSemanticTypeRegisteredToFormat(PairwiseFeatureData, PairwiseFeatureDataDirectoryFormat)