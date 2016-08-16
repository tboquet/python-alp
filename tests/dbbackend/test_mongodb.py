import pytest

from alp.dbbackend import mongo_backend as mgb


def test_create_db():
    mgb.create_db(True)


if __name__ == "__main__":
    pytest.main([__file__])
