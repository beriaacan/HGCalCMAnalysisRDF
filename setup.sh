#!/usr/bin/env bash
# Source this file in Bash. Optional arguments: CMSSW release directory, SCRAM_ARCH.
_hgcal_setup() {
    local framework_dir release_dir runtime
    framework_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)" || return
    release_dir="${1:-$(cd -- "$framework_dir/../../../.." && pwd)}"
    if [[ ! -d "$release_dir/.SCRAM" ]]; then
        printf 'Not a CMSSW release: %s\n' "$release_dir" >&2
        return 1
    fi
    if declare -F deactivate >/dev/null; then deactivate; fi
    source /cvmfs/cms.cern.ch/cmsset_default.sh || return
    if [[ -n "${2:-}" ]]; then export SCRAM_ARCH="$2"; fi
    cd -- "$release_dir/src" || return
    runtime="$(scram runtime -sh)" || return
    eval "$runtime" || return
    cd -- "$framework_dir" || return
}
_hgcal_setup "$@"
_hgcal_setup_status=$?
unset -f _hgcal_setup
return "$_hgcal_setup_status" 2>/dev/null || exit "$_hgcal_setup_status"
