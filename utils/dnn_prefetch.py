"""Ordered, bounded CPU preparation while the main thread trains on the GPU.

The ROOT iterator has exactly one owner thread. No fork, duplicate source
readers or extra RNG streams are introduced. Close/join it before releasing
the training lock, including on Ctrl-C and producer/consumer failures.
"""
from __future__ import annotations

from contextlib import contextmanager
from queue import Empty, Full, Queue
from threading import Event, Thread


def cpu_batch(batch, *, pin_memory):
    import torch
    tensors=tuple(torch.as_tensor(value) for value in batch)
    return tuple(value.pin_memory() for value in tensors) if pin_memory else tensors


@contextmanager
def prefetched_batches(source, *, depth, prepare):
    """Yield the same ordered batches; bound queued payloads to depth.

    There can additionally be one batch under construction and one consumed
    by training. A depth of zero selects synchronous preparation explicitly.
    The producer owns and closes source, propagating its original exception.
    """
    source=iter(source)
    if depth==0:
        try:
            yield (prepare(batch) for batch in source)
        finally:
            close=getattr(source,'close',None)
            if close is not None: close()
        return
    pending=Queue(maxsize=depth)
    stopped=Event()

    def offer(kind,value):
        while not stopped.is_set():
            try:
                pending.put((kind,value),timeout=.1)
                return True
            except Full:
                pass
        return False

    def produce():
        failure=None
        try:
            while not stopped.is_set():
                try:
                    batch=next(source)
                except StopIteration:
                    break
                if stopped.is_set(): break
                if not offer('batch',prepare(batch)): break
                # Pinned staging owns its copy. Do not keep the previous large
                # unpinned NumPy batch alive while constructing the next one.
                del batch
        except BaseException as exc:
            failure=exc
        finally:
            try:
                close=getattr(source,'close',None)
                if close is not None: close()
            except BaseException as exc:
                if failure is None: failure=exc
            offer('error',failure) if failure is not None else offer('end',None)

    def consume():
        while True:
            kind,value=pending.get()
            if kind=='error': raise value
            if kind=='end': return
            yield value

    worker=Thread(target=produce,name='hgcal-training-inputs',daemon=True)
    worker.start()
    try:
        yield consume()
    finally:
        stopped.set()
        worker.join()
        # Release any unconsumed pinned tensors promptly after an interruption.
        while True:
            try: pending.get_nowait()
            except Empty: break
