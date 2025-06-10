"""Tests for the Zarr file format functionality in the q2-gglasso plugin.

This module tests the Zarr storage and retrieval functionality of the q2-gglasso plugin,
ensuring that solution objects can be properly serialized and deserialized.
"""

import unittest
import zarr
import q2_gglasso as q2g
import pandas as pd
import numpy as np
from gglasso.problem import glasso_problem

try:
    from q2_gglasso._func import solve_problem   # noqa: F401

except ImportError:
    raise ImportWarning("Qiime2 not installed.")


class TestUtil(unittest.TestCase):
    """Test case for Zarr file format functionality in the q2-gglasso plugin.

    This test class verifies that the Zarr storage and retrieval functionality
    works correctly for serializing and deserializing GGLasso solution objects.
    """

    def test_zarr_format(self):
        """Test Zarr file format for storing and retrieving GGLasso solutions.

        This test:
        1. Creates a sample GGLasso problem and solution
        2. Saves the solution to a Zarr file
        3. Loads the solution from the Zarr file
        4. Verifies that the loaded solution matches the original
        """
        table = pd.DataFrame(
            [[1, 1, 7, 3], [2, 6, 2, 4], [5, 5, 3, 3], [3, 2, 8, 1]],
            index=["s1", "s2", "s3", "s4"],
            columns=["o1", "o2", "o3", "o4"],
        )

        S = np.cov(table.values)
        n_samples = table.shape[0]
        reg_params = {"lambda1": [0.5, 0.01], "mu1": [0.5, 0.1]}

        P = glasso_problem(S, N=n_samples, latent=True)
        P.model_selection(modelselect_params=reg_params)

        # save GGLasso solution
        solution = P.__dict__["solution"]

        # write GGLasso solution to zarr forma
        zipfile = str("problem.zip")
        store = zarr.ZipStore(zipfile, mode="w")
        root = zarr.open(store=store)
        q2g.to_zarr(P.__dict__, "problem", root)
        store.close()

        # read GGLasso solution from zarr forma
        store_new = zarr.ZipStore(str("problem.zip"), mode="r")
        root_new = zarr.open(store=store_new)

        # check if these solutions are equal
        if "solution" in root_new:

            for key, value in solution.__dict__.items():
                if key in root_new["solution/"]:
                    zarr_value = np.array(root_new["solution/" + key])

                    x = True
                    if zarr_value.all() == np.array(value).all():
                        continue
                    else:
                        x = False

        self.assertTrue(
            x, msg="The content of zarr file is not equal to GGLasso solution."
        )


if __name__ == "__main__":
    unittest.main()
