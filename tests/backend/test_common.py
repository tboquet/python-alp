import pytest
from alp.appcom.utils import imports


def test_imports():
    import numpy

    @imports()
    def dummy():
        return 0

    assert dummy() == 0

    @imports([numpy])
    def ones_check():
        return numpy.ones((1))

    assert ones_check().sum() == 1


if __name__ == "__main__":
    pytest.main([__file__])
