# Sourced in Condor jobs after setup.sh (condor_setup_file).
# CMSSW puts its CUDA 13.3 libraries, incl. a newer libnvidia-ml/libcuda (scram_cuda), in
# LD_LIBRARY_PATH. On batch GPUs with an older driver (e.g. 580) this gives
# "NVML Driver/library version mismatch" and torch sees no GPU. Torch brings its own CUDA
# runtime, so drop the CMSSW CUDA paths and use the node's own driver.
LD_LIBRARY_PATH="$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
    | grep -v -E 'scram_cuda|/external/cuda/' | paste -sd ':' -)"
export LD_LIBRARY_PATH
