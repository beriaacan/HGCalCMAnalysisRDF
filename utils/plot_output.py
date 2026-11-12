"""Overwrite figures at fixed local paths, recording the completed output set."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile

from analytic_allinputs import _write_manifest
from .analytic_artifacts import sha, signature

SCHEMA = 'hgcal_cm_plot_output_v1'


@dataclass
class Output:
    path: Path
    record: dict

    @property
    def token(self):
        return {'directory':str(self.path),'signature':self.record['signature'],
                'outputs':self.record['outputs']}


def publish(directory, definition, build):
    """Render first, replace each figure, then publish its completion manifest."""
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    previous_path=directory/'manifest.json'
    previous=json.loads(previous_path.read_text()) if previous_path.is_file() else {}
    work=Path(tempfile.mkdtemp(prefix='.work-',dir=directory))
    try:
        summary=build(work)
        files=sorted(p for p in work.rglob('*') if p.is_file())
        if not files:
            raise ValueError('Plot renderer produced no files')
        outputs=[{'file':str(p.relative_to(work)),'sha256':sha(p)} for p in files]
        current={row['file'] for row in outputs}
        obsolete=[]
        if previous.get('schema')==SCHEMA:
            for row in previous.get('outputs',[]):
                relative=Path(row['file'])
                if relative.is_absolute() or '..' in relative.parts:
                    raise ValueError('Invalid path in previous plot manifest')
                if row['file'] not in current:
                    obsolete.append(directory/relative)
        # Once replacement starts, an interruption must not leave a stale
        # completion claim for a partly replaced set of figures.
        owned=current | {row['file'] for row in previous.get('outputs',[])
                         if previous.get('schema')==SCHEMA}
        _write_manifest(directory,{'schema':SCHEMA,'complete':False,
                                   'outputs':[{'file':name} for name in sorted(owned)]})
        for source in files:
            destination=directory/source.relative_to(work)
            destination.parent.mkdir(parents=True,exist_ok=True)
            os.replace(source,destination)
        for path in obsolete:
            path.unlink(missing_ok=True)
        record={'schema':SCHEMA,'complete':True,'definition':definition,
                'signature':signature(definition),'summary':summary,'outputs':outputs}
        _write_manifest(directory,record)
        shutil.rmtree(work)
        return Output(directory,record)
    except Exception as exc:
        raise RuntimeError(f'{exc}. Retained plotting work: {work}') from exc
