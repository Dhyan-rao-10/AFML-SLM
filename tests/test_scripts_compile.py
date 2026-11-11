import glob
import os
import py_compile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("path", sorted(glob.glob(os.path.join(ROOT, "scripts", "*.py"))), ids=os.path.basename)
def test_script_compiles(path):
    """Scripts run as subprocesses on Kaggle/Colab, where a syntax error surfaces hours in."""
    py_compile.compile(path, doraise=True)
