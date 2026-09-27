import pytest

from offshore_risk import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()
