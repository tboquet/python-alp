"""Tests for the appcom module"""

import numpy as np
import pytest
from alp.appcom.utils import appbackend
from alp.appcom.utils import sliced


def test_sliced():
    data = {k: np.ones((100,2)) for k in range(10)}
    indices = sliced(data, 30, 20, 10)
    assert len(data[1][:indices[0]]) == 10
    assert len(data[1][indices[0]:indices[1]]) == 30
    assert len(data[1][indices[1]:indices[2]]) == 20


def test_appbackend():
    class Dumobj(object):
        def __init__(self):
            self.backend = None

    @appbackend
    def decoratedfunction(bob, mod):
        import sys
        assert "ABE" in sys.modules.keys()

    mod = {"backend": "keras"}
    bob = Dumobj("backend")

    decoratedfunction(bob, mod)


if __name__ == "__main__":
    pytest.main([__file__])
