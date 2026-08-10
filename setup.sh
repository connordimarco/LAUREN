#!/usr/bin/env bash
# One-shot environment setup for the AU/AL corrector package.
# Run from the package directory on the target machine:
#   bash setup.sh [python]        (python >= 3.10; default: python3.11)
set -euo pipefail
cd "$(dirname "$0")"
PY="${1:-python3.11}"
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' \
    || { echo "need python >= 3.10 (got $PY)"; exit 1; }
"$PY" -m venv venv
venv/bin/pip install --quiet --upgrade pip
venv/bin/pip install --quiet -r requirements.txt
venv/bin/python -c "
import numpy, pandas, joblib, lightgbm, sklearn, geopack
print('env ok: numpy', numpy.__version__, '| lightgbm', lightgbm.__version__,
      '| sklearn', sklearn.__version__)"
echo "next: venv/bin/python validate_model.py <corrector .joblib>"
