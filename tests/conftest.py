import dataclasses

import pytest

from tsnscout.generator import SMALL

#: Small enough that the exact solver answers in milliseconds.
TINY = dataclasses.replace(SMALL, n_switches=(4, 5), n_flows=(8, 18))


@pytest.fixture(scope="session")
def tiny_cfg():
    return TINY
