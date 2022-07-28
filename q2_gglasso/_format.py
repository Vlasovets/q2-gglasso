import qiime2.plugin.model as model
from qiime2.plugin import ValidationError
import pandas as pd
from itertools import combinations
import zarr
import numpy as np


class TensorDataFormat(model.BinaryFileFormat):
    def validate(self, level):
        pass


TensorDataDirectoryFormat = model.SingleFileDirectoryFormat('TensorDataDirectoryFormat',
                                                            'tensor.zip', format=TensorDataFormat)


class GGLassoDataFormat(model.TextFileFormat):
    def validate(self, level):
        pass


PairwiseFeatureDataDirectoryFormat = model.SingleFileDirectoryFormat(
    'PairwiseFeatureDataDirectoryFormat', 'pairwise_comparisons.tsv', GGLassoDataFormat)


class ZarrProblemFormat(model.BinaryFileFormat):
    def validate(self, level):
        pass


GGLassoProblemDirectoryFormat = model.SingleFileDirectoryFormat('GGLassoProblemDirectoryFormat',
                                                                'problem.zip', format=ZarrProblemFormat)
