"""Interactive sample progress, with plain periodic reporting in batch logs."""
from __future__ import annotations
import os
import sys
import time


def sample_total(manifest, modules, partition):
    label = ('train', 'validation')[partition]
    return sum(sum(row[label]) for module in modules
               for row in manifest['counts_by_module_run'][module].values())


def report_batches(data, total, description):
    interactive = sys.stderr.isatty() and not os.environ.get('HGCAL_BATCH') and not os.environ.get('_CONDOR_JOB_AD')
    if interactive:
        from tqdm import tqdm
        with tqdm(total=total, desc=description, unit='sample', unit_scale=True,
                  dynamic_ncols=True, leave=False, mininterval=.5) as progress:
            for batch in data:
                yield batch
                progress.update(len(batch[1]))
    else:
        seen = 0
        last = time.monotonic()
        print(f'{description}: 0/{total:,} samples', flush=True)
        for batch in data:
            yield batch
            seen += len(batch[1])
            now = time.monotonic()
            if now - last >= 60:
                print(f'{description}: {seen:,}/{total:,} samples ({100*seen/total:.1f}%)', flush=True)
                last = now
