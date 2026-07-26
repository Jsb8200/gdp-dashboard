import pytest

from test_flow_types import REAL_CSV
from dealer_gex.parsing import parse_file


@pytest.fixture(scope="module")
def real():
    pf = parse_file(REAL_CSV.encode())
    assert pf.prints is not None
    return pf
