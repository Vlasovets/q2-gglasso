import qiime2.plugin.model as model
from qiime2.plugin import ValidationError
import pandas as pd
from itertools import combinations
import zarr
import numpy as np


class TensorDataFormat(model.BinaryFileFormat):
    """Binary format for storing tensor data.

    This format is used to store multi-dimensional data for GGLasso operations.
    """
    def validate(self, level):
        """Validate the tensor data format.

        Parameters
        ----------
        level : str
            The level of validation to perform.
        """
        pass


TensorDataDirectoryFormat = model.SingleFileDirectoryFormat('TensorDataDirectoryFormat',
                                                            'tensor.zip', format=TensorDataFormat)


class GGLassoDataFormat(model.TextFileFormat):
    """Text file format for GGLasso data.

    This format handles text-based data for GGLasso analyses.
    """
    def validate(self, level):
        """Validate the GGLasso data format.

        Parameters
        ----------
        level : str
            The level of validation to perform.
        """
        pass


PairwiseFeatureDataDirectoryFormat = model.SingleFileDirectoryFormat(
    'PairwiseFeatureDataDirectoryFormat', 'pairwise_comparisons.tsv', GGLassoDataFormat)


class ZarrProblemFormat(model.BinaryFileFormat):
    """Binary format for storing Zarr problem data.

    This format stores GGLasso problem data using the Zarr array format.
    """
    def validate(self, level):
        """Validate the Zarr problem format.

        Parameters
        ----------
        level : str
            The level of validation to perform.
        """
        pass


GGLassoProblemDirectoryFormat = model.SingleFileDirectoryFormat('GGLassoProblemDirectoryFormat',
                                                                'problem.zip', format=ZarrProblemFormat)
