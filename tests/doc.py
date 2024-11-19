#!/usr/bin/python

# Adding an import for testing
import pytest

# Adding basic pytest line
assert 0.1 + 0.2 == pytest.approx(0.3)
