import numpy as np
import torch

from function import ExperimentBase


def test_set_seed_reproducibility_numpy_and_torch():
    ExperimentBase.set_seed(123)

    a1 = np.random.rand(5)
    t1 = torch.rand(5)

    ExperimentBase.set_seed(123)

    a2 = np.random.rand(5)
    t2 = torch.rand(5)

    assert np.allclose(a1, a2)
    assert torch.allclose(t1, t2)