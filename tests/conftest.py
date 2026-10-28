import random

import pytest


@pytest.fixture(autouse=True)
def _seed():
    random.seed(0)