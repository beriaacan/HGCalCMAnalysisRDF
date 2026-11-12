#!/usr/bin/env bash
# Run after activating CMSSW (source setup.sh). Central packages stay unchanged.
set -euo pipefail
: "${CMSSW_BASE:?Activate CMSSW before setting up the training environment}"
framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
training_env_dir="${1:-$framework_dir/.venv_training}"
python3 -m venv --system-site-packages "$training_env_dir"
training_site_dir="$training_env_dir/lib/$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')/site-packages"
PYTHONPATH="$training_site_dir:${PYTHONPATH:-}" "$training_env_dir/bin/python3" -m pip install --no-cache-dir -r "$framework_dir/requirements-training.txt"
PYTHONPATH="$training_site_dir:${PYTHONPATH:-}" "$training_env_dir/bin/python3" - <<'PY'
import json,sys
from pathlib import Path
import ROOT,numpy,torch
assert torch.version.cuda is not None, 'The installed PyTorch has no CUDA support'
x=torch.from_numpy(numpy.ones(3,dtype=numpy.float32))
assert x.sum().item()==3 and ROOT.RDataFrame(3).Count().GetValue()==3
record={'python':sys.version.split()[0],'ROOT':ROOT.gROOT.GetVersion(),
        'torch':torch.__version__,'cuda_build':torch.version.cuda,'numpy':numpy.__version__}
(Path(sys.prefix)/'runtime.json').write_text(json.dumps(record,indent=2)+'\n')
print('Training environment ready:',sys.prefix,record)
print('GPU available on this host:',torch.cuda.is_available())
PY
