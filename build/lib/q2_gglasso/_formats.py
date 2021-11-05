import qiime2.plugin.model as model
import zarr
import numpy as np


class ZarrProblemFormat(model.BinaryFileFormat):
    def validate(self, level):
        pass


CLASSOProblemDirectoryFormat = model.SingleFileDirectoryFormat('CLASSOProblemDirectoryFormat',
                                                               'problem.zip', format=ZarrProblemFormat)