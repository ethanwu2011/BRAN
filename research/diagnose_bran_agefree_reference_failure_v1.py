"""One-shot local replay diagnostic; disclose code locations, never exception text.

Does not fit a representation, modify frozen inputs, or save reference arrays.
The original failed experiment and its scientific thresholds remain untouched.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PIN = 'b38969cb251baf71314688414537c863e17707de2577b79e9bbe9dff47a291ca'
FAILURE_PIN = 'da8790aaa8ec6a2e9cdbf378d125f669c1cbac767114d5a233c92fd608d510af'
OUT = ROOT / 'BRAN_AGEFREE_REFERENCE_DIAGNOSTIC_V1'
LOCK = Path('/private/tmp/bran_retinal_extraction_v1.lock')
TYPES = {kind: kind.__name__ for kind in (
    ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError,
    AssertionError, OSError, FileNotFoundError, PermissionError, MemoryError,
    ImportError, ModuleNotFoundError, ZeroDivisionError)}


@contextmanager
def quiet():
    sys.stdout.flush(); sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    null = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(null, 1); os.dup2(null, 2)
        yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(saved[0], 1); os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def safe_trace(error, root, allowed_names):
    """Closed exception types + authenticated repo file/line, no text or locals."""
    allowed = {}
    root = Path(root).resolve()
    for name in allowed_names:
        if type(name) is not str:
            continue
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or path.suffix != '.py':
            continue
        full = (root / path).resolve()
        if full.is_relative_to(root):
            allowed[str(full)] = name
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        frames = []
        tb = error.__traceback__
        while tb is not None:
            name = allowed.get(tb.tb_frame.f_code.co_filename)
            if name is not None:
                frames.append({'code_file': name, 'line': tb.tb_lineno})
            tb = tb.tb_next
        result.append({'exception_type': TYPES.get(type(error), 'Exception'),
                       'frames': frames[-8:]})
        error = error.__cause__ if error.__cause__ is not None else error.__context__
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), 'w') as handle:
        json.dump(value, handle, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())


def original_state():
    protocol = ROOT / 'BRAN_AGEFREE_UNIFIED_PROTOCOL_V1.json'
    failed = ROOT / 'BRAN_AGEFREE_UNIFIED_V1'
    assert sha(protocol) == PIN
    assert {p.name for p in failed.iterdir()} == {'progress.json', 'failure.json'}
    assert sha(failed / 'failure.json') == FAILURE_PIN
    assert json.loads((failed / 'failure.json').read_bytes()) == {
        'status': 'failed', 'phase': 'reference_replay', 'patient_level_output_emitted': False}
    assert json.loads((failed / 'progress.json').read_bytes()) == {
        'phase': 'reference_replay', 'fold': None, 'patient_level_output_emitted': False}
    p = json.loads(protocol.read_bytes())
    for name, digest in p['code_sha256'].items():
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(ROOT / path) == digest
    return p


def main():
    answer = {'status': 'diagnostic_failed', 'phase': 'authentication',
              'patient_level_output_emitted': False, 'representation_fitted': False}
    owned, allowed = False, ()
    with quiet():
        try:
            with LOCK.open('a') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                p = original_state()
                allowed = tuple(p['code_sha256'])
                import run_bran_agefree_unified_v1 as run
                run.storage.authenticate(run.PRIVATE, p['prepared_private_sha256'])
                OUT.mkdir(mode=0o700)
                owned = True

                def progress(phase):
                    answer['phase'] = phase
                    temporary = OUT / 'progress.tmp'
                    write_json(temporary, {'phase': phase, 'patient_level_output_emitted': False})
                    os.replace(temporary, OUT / 'progress.json')

                progress('authentication')
                try:
                    assert run.r.equal(run.load_protocol(PIN), p)
                    progress('input_binding')
                    inputs = run._inputs(p, p['prepared_private_sha256'])
                    run.torch.set_num_threads(p['parameters']['cpu_threads'])
                    progress('reference_replay')
                    run._references(inputs, p, PIN)
                    answer['status'] = 'reference_replay_passed'
                except Exception as error:
                    answer['trace'] = safe_trace(error, ROOT, allowed)
                    if answer['phase'] == 'reference_replay':
                        answer['status'] = 'failure_reproduced'
                assert run.r.equal(original_state(), p)
                run.storage.authenticate(run.PRIVATE, p['prepared_private_sha256'])
                answer.update({'original_artifacts_preserved': True,
                               'protocol_sha256': PIN, 'original_failure_sha256': FAILURE_PIN,
                               'diagnostic_code_sha256': sha(Path(__file__))})
                write_json(OUT / 'diagnostic.json', answer)
        except Exception:
            answer = {'status': 'diagnostic_failed', 'phase': answer['phase'],
                      'patient_level_output_emitted': False, 'representation_fitted': False}
            if owned:
                try:
                    write_json(OUT / 'failure.json', answer)
                except Exception:
                    pass
    print(json.dumps(answer, sort_keys=True))
    return int(answer['status'] == 'diagnostic_failed')


if __name__ == '__main__':
    raise SystemExit(main())
