"""Strict Nano metadata and EOS access shared by framework steps."""
from __future__ import annotations

def root_path(path: str) -> str:
    if path.startswith("/eos/cms/"):
        return "root://eoscms.cern.ch/" + path
    if path.startswith("/eos/user/"):
        return "root://eosuser.cern.ch/" + path
    return path



def read_module_metadata(filename: str) -> dict[str, tuple[int, int, int]]:
    import ROOT

    ROOT.PyConfig.IgnoreCommandLineOptions = True
    arrays = ROOT.RDataFrame("Runs", root_path(filename)).AsNumpy()
    for name in ("HGCReadout_FED", "HGCReadout_Seq", "HGCReadout_nErx"):
        if name not in arrays or len(arrays[name]) != 1:
            raise ValueError(f"Expected one Runs entry containing {name}: {filename}")
    lengths = {len(arrays[key][0]) for key in
               ("HGCReadout_FED", "HGCReadout_Seq", "HGCReadout_nErx")}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent readout metadata lengths: {filename}")
    modules = {}
    for branch, rows in arrays.items():
        if not branch.startswith("HGCTypeCodes_"):
            continue
        if len(rows) != 1 or len(rows[0]) != 1:
            raise ValueError(f"Expected one module index in {branch}: {filename}")
        index = int(rows[0][0])
        if not 0 <= index < len(arrays["HGCReadout_FED"][0]):
            raise ValueError(f"Invalid module index in {branch}: {filename}")
        fed, sequence, nerx = (int(arrays[key][0][index]) for key in
                               ("HGCReadout_FED", "HGCReadout_Seq", "HGCReadout_nErx"))
        if nerx <= 0:
            raise ValueError(f"Invalid eRx count in {branch}: {filename}")
        modules[branch.removeprefix("HGCTypeCodes_")] = (fed, sequence, nerx)
    if not modules:
        raise ValueError(f"No module metadata in {filename}")
    return modules

