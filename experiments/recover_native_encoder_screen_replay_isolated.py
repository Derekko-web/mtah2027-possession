"""R71: R61 cross-runtime screen-only recovery under the isolated interpreter. Preparation and verification use stdlib only.

Never restarts refits or assessments. The original frozen screen function is
loaded only after a root-reviewed recovery freeze, with its output suffix as
the sole AST change. Frozen numerical bodies remain exact; an explicit entry shim binds the isolated runtime.
"""
import argparse
import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SELF = Path(__file__).resolve()
ROUND = ROOT / 'data/control_native_encoder_continuation_extended_v1'
PLAN = ROOT / 'experiments/R61_SCREEN_REPLAY_ISOLATED_PLAN.json'
ORIGINAL = ROOT / 'experiments/research_native_encoder_continuation_extended.py'
ORIGINAL_CORE = ROOT / 'experiments/native_encoder_continuation_extended.py'
ORIGINAL_PLAN = ROOT / 'experiments/NATIVE_ENCODER_CONTINUATION_EXTENDED_PLAN.json'
OUTPUT = ROUND / 'screen_replay_isolated_v1'
FREEZE = ROUND / 'screen_replay_isolated_protocol_freeze_v1'
RESULT = ROUND / 'screen_replay_isolated_result_v1.json'
OLD_SUFFIX = '_replay'
NEW_SUFFIX = '_replay_isolated_v1'
PROTOCOLS = ('buffered_minutes', 'purged_spells', 'buffered_100_seconds')
TERMINALS = ('manifest.json', 'selection.json', 'scores.csv', 'fold_scores.csv',
             'segment_diagnostics.csv', 'official_comparison.csv')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')


def normalized(node):
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def screen_ast(source=None):
    tree = ast.parse(ORIGINAL.read_text() if source is None else source)
    found = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'screen']
    if len(found) != 1:
        raise ValueError('Exactly one original screen definition is required')
    return found[0]


def output_suffix_node(node, expected):
    assignments = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
        and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == 'out']
    expected_value = ast.parse("ROUND / (stage+" + repr(expected) + " if replay else stage)", mode='eval').body
    if len(assignments) != 1 or normalized(assignments[0].value) != normalized(expected_value):
        raise ValueError('The original screen output assignment changed')
    value = assignments[0].value.right.body.right
    if not isinstance(value, ast.Constant) or value.value != expected:
        raise ValueError('Output suffix constant is not unique at the required assignment')
    return value


def recovered_ast(original=None):
    original = screen_ast() if original is None else original
    changed = copy.deepcopy(original)
    output_suffix_node(changed, OLD_SUFFIX).value = NEW_SUFFIX
    # Reversing precisely this edit must recover the complete original AST.
    restored = copy.deepcopy(changed)
    output_suffix_node(restored, NEW_SUFFIX).value = OLD_SUFFIX
    if normalized(restored) != normalized(original):
        raise ValueError('Any screen AST change beyond the output suffix is forbidden')
    ast.fix_missing_locations(changed)
    return changed


def inventory(folder):
    folder = Path(folder)
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def audit_manifest(folder):
    folder = Path(folder)
    manifest = read_json(folder / 'manifest.json')
    for name, expected in manifest['cache_sha256'].items():
        if digest(folder / name) != expected:
            raise ValueError('Existing manifest mismatch: ' + str(folder / name))
    return manifest



ISOLATED = ROOT / 'data/control_isolated_python_v1'
HEAD = ISOLATED / 'head_runtime/bin/python'
TORCH25 = ISOLATED / 'torch25_runtime/bin/python'
LEGACY_INTERPRETER = '/Users/derekko/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12'
LEGACY_SHA = 'd59f5251bce588951dc870bde45f0cd93de39327b1ed2431a8d26bfb567aac10'
ISOLATED_SHA = 'ab9c532a75ce81eb96562bda04138e7a0b7528bd242c82a436ebeb1405aa361f'


def runtime_binding_matches(path, expected):
    # New protocol explicitly replaces only the obsolete executable binding.
    # This never declares the old shared-runtime verifier successful.
    if str(path) == LEGACY_INTERPRETER:
        if expected != LEGACY_SHA:
            raise ValueError('Historical interpreter identity changed')
        return digest(ISOLATED / 'python/bin/python3.12') == ISOLATED_SHA
    return digest(ROOT / path) == expected


def check_isolated_runtime():
    plan = read_json(PLAN)
    if digest(ISOLATED / 'acquisition.json') != plan['isolated_runtime_acquisition_sha256']:
        raise ValueError('Isolated runtime acquisition changed')
    identity = read_json(ISOLATED / 'acquisition.json')
    if identity['isolated_interpreter_sha256'] != ISOLATED_SHA:
        raise ValueError('Unexpected isolated interpreter')
    for relative, expected in identity['files_sha256'].items():
        if digest(ISOLATED / relative) != expected:
            raise ValueError('Isolated Python/stdlib changed: ' + relative)
    for relative, target in identity['symlinks'].items():
        path = ISOLATED / relative
        if not path.is_symlink() or str(path.readlink()) != target:
            raise ValueError('Isolated interpreter/library link changed')
    for name, expected in plan['isolated_venv_configuration_sha256'].items():
        if digest(ISOLATED / name / 'pyvenv.cfg') != expected:
            raise ValueError('Isolated virtual environment changed')
    if digest(HEAD) != ISOLATED_SHA or digest(TORCH25) != ISOLATED_SHA:
        raise ValueError('Active isolated executable differs')


def rebound_check_ast():
    tree = ast.parse(ORIGINAL_CORE.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'check_inputs')
    changed = copy.deepcopy(original)
    wanted = ast.parse('digest(name) != expected', mode='eval').body
    matches = [n for n in ast.walk(changed) if isinstance(n, ast.If) and normalized(n.test) == normalized(wanted)]
    if len(matches) != 1:
        raise ValueError('Expected one original runtime verification predicate')
    matches[0].test = ast.parse('not runtime_binding_matches(name, expected)', mode='eval').body
    restored = copy.deepcopy(changed)
    predicate = ast.parse('not runtime_binding_matches(name, expected)', mode='eval').body
    edits = [n for n in ast.walk(restored) if isinstance(n, ast.If) and normalized(n.test) == normalized(predicate)]
    if len(edits) != 1:
        raise ValueError('Unexpected verification rewrite')
    edits[0].test = wanted
    if normalized(restored) != normalized(original):
        raise ValueError('Any other core verification change is forbidden')
    return ast.fix_missing_locations(changed)


def load_isolated_controller(role):
    if role not in ('head', 'torch'):
        raise ValueError('Unknown isolated runtime role')
    if Path(sys.base_prefix).resolve() != (ISOLATED / 'python').resolve() or digest(Path(sys.executable)) != ISOLATED_SHA:
        raise ValueError('Only project-local isolated Python may execute recovery')
    expected_prefix = ISOLATED / ('head_runtime' if role == 'head' else 'torch25_runtime')
    if Path(sys.prefix).resolve() != expected_prefix.resolve():
        raise ValueError('Unexpected isolated scientific-package environment')
    check_isolated_runtime()
    sys.path.insert(0, str(ROOT / 'experiments'))
    import native_encoder_continuation_extended as core
    core.runtime_binding_matches = runtime_binding_matches
    exec(compile(ast.Module(body=[rebound_check_ast()], type_ignores=[]), str(ORIGINAL_CORE)+':isolated-runtime', 'exec'), core.__dict__)
    core.PYTHON, core.HEAD_PYTHON = TORCH25, HEAD
    spec = importlib.util.spec_from_file_location('r71_original_controller_runtime_rebound', ORIGINAL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.__file__ = str(SELF)  # Child encoders enter this shim, not obsolete runtime checks.
    if module.PYTHON != TORCH25 or module.HEAD_PYTHON != HEAD or module.check_inputs is not core.check_inputs:
        raise ValueError('Isolated execution/verification binding failed')
    return module


def encode_worker(model_folder, destination, parent):
    verify_freeze()
    if not destination.resolve().is_relative_to(OUTPUT.resolve()):
        raise ValueError('Encoder output escapes the new recovery directory')
    if not model_folder.resolve().is_relative_to((ROUND / 'neural_replay').resolve()):
        raise ValueError('Encoder must use an already-complete refitted checkpoint')
    if parent is None or not parent.resolve().is_relative_to((ROUND / 'screen/encodings').resolve()):
        raise ValueError('Every fresh encoding requires its original exact parent')
    if not os.environ.get('R61_RECOVERY_SUPERVISOR_PID'):
        raise ValueError('Encoder must be owned by the supervised recovery tree')
    module = load_isolated_controller('torch')
    module.encode_worker(model_folder, destination, parent)
    verify_freeze()


def check_existing():
    check_isolated_runtime()
    plan = read_json(PLAN)
    for path, expected in plan['immutable_sha256'].items():
        if digest(ROOT / path) != expected:
            raise ValueError('Bound evidence changed: ' + path)
    for folder, expected in plan['exact_inventory_sha256'].items():
        if inventory(ROOT / folder) != expected:
            raise ValueError('Bound folder inventory changed: ' + folder)
    old_plan = read_json(ORIGINAL_PLAN)
    original_freeze = ROUND / 'protocol_freeze'
    if ORIGINAL_PLAN.read_bytes() != (original_freeze / 'plan.json').read_bytes():
        raise ValueError('Original R61 freeze/plan differ')
    for group in (old_plan['input_sha256'], read_json(original_freeze / 'implementation.json'),
                  read_json(original_freeze / 'protected_artifacts.json'), old_plan['runtime_sha256']):
        for path, expected in group.items():
            if not runtime_binding_matches(path, expected):
                raise ValueError('Original source/data or explicitly rebound runtime differs: ' + path)
    ledger = ROOT / 'experiments/EXPERIMENT_LEDGER.csv'
    if not ledger.read_bytes().startswith((original_freeze / 'EXPERIMENT_LEDGER.csv').read_bytes()):
        raise ValueError('Original historical ledger prefix changed')
    for path, expected in read_json(original_freeze / 'symlinks.json').items():
        candidate = ROOT / path
        if not candidate.is_symlink() or str(candidate.readlink()) != expected:
            raise ValueError('Original source symlink changed: ' + path)
    audit_manifest(ROUND / 'screen')
    main_selection = read_json(ROUND / 'screen/selection.json')
    if main_selection['passed'] or main_selection['eligible'] or main_selection['selected'] is not None:
        raise ValueError('Recovery must retain the rejected main quality verdict')
    folders = sorted((ROUND / 'neural_replay').rglob('manifest.json'))
    if len(folders) != 14:
        raise ValueError('Exactly fourteen completed independent refits required')
    for path in folders:
        folder = path.parent
        record = audit_manifest(folder)
        if not all(record.get(k) is True for k in ('passed', 'strict_reload_exact', 'independent_refit_exact')) or record['steps'] != 400:
            raise ValueError('Independent refit did not complete exactly: ' + str(folder))
        parent = ROUND / 'neural' / folder.relative_to(ROUND / 'neural_replay')
        audit_manifest(parent)
        for name in ('model.pt', 'training_loss.npy', 'probe_probability.npy'):
            if digest(folder / name) != digest(parent / name):
                raise ValueError('Independent model/loss/probe bytes differ: ' + str(folder / name))
        supervision = read_json(folder.parent / (folder.name + '_supervision.json'))
        if supervision.get('exit_code') != 0 or supervision.get('no_retry') is not True:
            raise ValueError('Independent refit supervision did not complete')
    audit_manifest(ROUND / 'external_assessment')
    audit_manifest(ROUND / 'external_assessment_replay')
    external = read_json(ROUND / 'external_assessment_replay/selection.json')
    if external['eligible'] != ['native_continuation', 'source_continuation'] or external['models'] != 14 or len(external['pairs']) != 84:
        raise ValueError('Both externally passing arms / 84 cells required')
    if not all(all(row['checks'].values()) for row in external['pairs']):
        raise ValueError('Required external absolute gate failed')
    external_files = sorted((ROUND / 'external_assessment_replay').glob('*_probability.npy'))
    if len(external_files) != 28:
        raise ValueError('Exactly twenty-eight external replay probability files required')
    for path in external_files + [ROUND / 'external_assessment_replay/selection.json']:
        if digest(path) != digest(ROUND / 'external_assessment' / path.name):
            raise ValueError('External probability/gate replay differs')
    for name in plan['interruption_observation']['missing_terminal_files']:
        if (ROUND / 'screen_replay' / name).exists():
            raise ValueError('Original partial replay has acquired a terminal artifact')
    if len(list((ROUND / 'screen').rglob('*.ubj'))) != 26 or len(list((ROUND / 'screen/encodings').rglob('manifest.json'))) != 14:
        raise ValueError('Original complete main screen must retain 26 heads and 14 encodings')
    return plan


def preflight():
    plan = check_existing()
    original, changed = screen_ast(), recovered_ast()
    if plan['transformation'] != {'function': 'screen', 'assignment_target': 'out',
        'old_constant': OLD_SUFFIX, 'new_constant': NEW_SUFFIX, 'only_change': True,
        'call': {'seed': 42, 'replay': True}}:
        raise ValueError('Recovery transformation/call changed')
    if plan['output'] != str(OUTPUT.relative_to(ROOT)):
        raise ValueError('Recovery output path changed')
    # No original module import or array loading happens in this function.
    return dict(passed=True, standard_library_only=True, scientific_imports=False,
        original_screen_ast_sha256=hashlib.sha256(normalized(original).encode()).hexdigest(),
        recovery_screen_ast_sha256=hashlib.sha256(normalized(changed).encode()).hexdigest(),
        reverse_suffix_normalized_ast_exact=True, original_numerical_bodies_retained=True, isolated_entry_shim_declared=True, original_shared_runtime_verifier_currently_valid=False,
        completed_refits=14, external_probability_files=28, external_cells=84,
        partial_files=len(inventory(ROUND / 'screen_replay')), main_heads=26,
        main_encodings=14, recovery_output_exists=OUTPUT.exists(),
        source_sha256=digest(SELF), plan_sha256=digest(PLAN),
        main_quality_verdict='REJECTED; no eligible method; no confirmation/full build')


def freeze(reviewed_source, reviewed_plan):
    proof = preflight()
    if reviewed_source != digest(SELF) or reviewed_plan != digest(PLAN):
        raise ValueError('Explicit root-reviewed source and plan hashes must match')
    if OUTPUT.exists() or RESULT.exists():
        raise ValueError('Recovery output/result must be absent before freeze')
    FREEZE.mkdir(exist_ok=False)
    (FREEZE / SELF.name).write_bytes(SELF.read_bytes())
    (FREEZE / PLAN.name).write_bytes(PLAN.read_bytes())
    write_new(FREEZE / 'root_review.json', dict(reviewer='root', approved_for_screen_only_recovery=True,
        source_sha256=reviewed_source, plan_sha256=reviewed_plan, preflight=proof,
        no_refits_no_external_assessments_no_confirmations_no_full_build=True))
    return verify_freeze()


def verify_freeze():
    if SELF.read_bytes() != (FREEZE / SELF.name).read_bytes() or PLAN.read_bytes() != (FREEZE / PLAN.name).read_bytes():
        raise ValueError('Recovery source/plan differ from reviewed freeze')
    review = read_json(FREEZE / 'root_review.json')
    if review.get('reviewer') != 'root' or review.get('approved_for_screen_only_recovery') is not True or review['source_sha256'] != digest(SELF) or review['plan_sha256'] != digest(PLAN):
        raise ValueError('Root recovery review is missing or changed')
    return preflight()


def verify_recovered():
    manifest = audit_manifest(OUTPUT)
    old = ROUND / 'screen'
    encodings = sorted((OUTPUT / 'encodings').rglob('manifest.json'))
    heads = sorted(OUTPUT.rglob('*.ubj'))
    if len(encodings) != 14 or len(heads) != 26 or manifest['new_heads'] != 0:
        raise ValueError('Recovery must freshly encode all fourteen models and reuse twenty-six heads')
    equal = []
    paths = [OUTPUT / name for name in TERMINALS if name != 'manifest.json']
    paths += [OUTPUT / protocol / name for protocol in PROTOCOLS for name in
        ('native_continuation_probability.npy', 'source_continuation_probability.npy', 'oof_predictions.csv')]
    paths += heads
    paths += sorted((OUTPUT / 'encodings').rglob('*.npy'))
    paths += sorted(OUTPUT.glob('*/official/reports/*'))
    paths += sorted(OUTPUT.glob('*/official/scoreboard.csv'))
    paths += sorted(OUTPUT.glob('*/scorer_inputs/*.csv'))
    for path in paths:
        relative = path.relative_to(OUTPUT)
        if digest(path) != digest(old / relative):
            raise ValueError('Recovered artifact differs from main screen: ' + str(relative))
        equal.append(str(relative))
    selection = read_json(OUTPUT / 'selection.json')
    if selection['passed'] or selection['eligible'] or selection['selected'] is not None:
        raise ValueError('Main rejection must remain exact')
    return dict(passed=True, fresh_query_encodings=14, reused_heads=26, new_head_fits=0,
        byte_equal_main_artifacts=equal, manifest_sha256=digest(OUTPUT / 'manifest.json'),
        main_quality_verdict='REJECTED', selected_artifacts_changed=False,
        confirmations_or_full_build_authorized=False)


def screen_worker():
    before = verify_freeze()
    if OUTPUT.exists() or RESULT.exists():
        raise ValueError('No recovery rerun or overwrite is allowed')
    original_plan = read_json(ORIGINAL_PLAN)
    expected_python = HEAD.absolute()
    if Path(sys.executable).absolute() != expected_python:
        raise ValueError('Use the isolated fixed head interpreter')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        if os.environ.get(key) != '2':
            raise ValueError('Recovery requires the original two-thread environment')
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise ValueError('Recovery requires the original Python hash seed')
    if os.environ.get('PYTHONDONTWRITEBYTECODE') != '1':
        raise ValueError('All inherited evaluator children must disable bytecode writes')
    if os.environ.get('R61_RECOVERY_SUPERVISOR_PID') != str(os.getppid()):
        raise ValueError('Only the two-GiB recovery supervisor may launch this worker')
    sys.dont_write_bytecode = True
    module = load_isolated_controller('head')
    if module.ROUND != ROUND or module.encode.__globals__ is not module.__dict__ or module.__file__ != str(SELF):
        raise ValueError('Numerical globals or isolated child entry changed')
    only_screen = ast.Module(body=[recovered_ast()], type_ignores=[])
    exec(compile(only_screen, str(ORIGINAL), 'exec'), module.__dict__)
    module.screen(seed=42, replay=True)
    after = verify_recovered()
    verify_freeze()  # All old evidence, selected artifacts and partial bytes remain exact.
    return after


def process_tree_rss(worker_pid):
    rows = {}
    output = subprocess.check_output(['/bin/ps', '-axo', 'pid=,ppid=,rss='], text=True)
    for line in output.splitlines():
        pid, parent, rss = map(int, line.split())
        rows[pid] = (parent, rss)
    descendants = {worker_pid}
    while True:
        expanded = descendants | {pid for pid, (parent, _) in rows.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    # Include the stdlib supervisor itself, in addition to the scientific tree.
    return sum(rows[pid][1] for pid in descendants if pid in rows) + rows.get(os.getpid(), (0, 0))[1]


def run():
    before = verify_freeze()
    if OUTPUT.exists() or RESULT.exists():
        raise ValueError('No recovery rerun or overwrite is allowed')
    plan = read_json(PLAN)
    if plan['resources'] != {'cpu_threads': 2, 'sampled_combined_rss_limit_gib': 2,
        'rss_sample_interval_seconds': .1, 'aggregate_admission_limit_gib': 8,
        'other_admitted_worker_limit_gib': 6, 'instantaneous_peak_guarantee': False}:
        raise ValueError('Recovery resource admission/supervision changed')
    interpreter = str(HEAD)
    env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'OPENBLAS_NUM_THREADS': '2', 'PYTHONHASHSEED': '42', 'PYTHONDONTWRITEBYTECODE': '1',
        'R61_RECOVERY_SUPERVISOR_PID': str(os.getpid())}
    worker = subprocess.Popen([interpreter, '-B', str(SELF), '--stage', '_screen-worker'],
        cwd=ROOT, env=env, start_new_session=True)
    peak_kib, samples, aborted = 0, 0, None
    started = time.monotonic()
    try:
        while worker.poll() is None:
            rss = process_tree_rss(worker.pid)
            peak_kib, samples = max(peak_kib, rss), samples + 1
            if rss > 2 * 1024**2:
                aborted = 'combined_supervisor_and_worker_tree_RSS_exceeded_2GiB'
                os.killpg(worker.pid, signal.SIGKILL)
                break
            time.sleep(.1)
    except BaseException:
        if worker.poll() is None:
            os.killpg(worker.pid, signal.SIGKILL)
        worker.wait()
        raise
    code = worker.wait()
    supervision = dict(exit_code=code, aborted=aborted, seconds=time.monotonic()-started,
        peak_sampled_combined_GiB=peak_kib / 1024**2, samples=samples,
        sample_interval_seconds=.1, instantaneous_peak_guarantee=False)
    result = dict(recovery_completed=code == 0 and aborted is None, supervision=supervision,
        before=before, old_screen_replay_retained=True,
        original_interruption_cause='unknown; no numerical classification inferred',
        source_sha256=digest(SELF), plan_sha256=digest(PLAN))
    if result['recovery_completed']:
        result.update(verify_recovered())
        verify_freeze()
        result['all_bound_old_bytes_preserved'] = True
    else:
        result['passed'] = False
    write_new(RESULT, result)
    if not result['recovery_completed']:
        raise RuntimeError('Recovery worker failed; new partial output and supervision retained')
    return result


class PureASTTests(unittest.TestCase):
    def test_exact_reverse_transformation(self):
        node = recovered_ast()
        output_suffix_node(node, NEW_SUFFIX).value = OLD_SUFFIX
        self.assertEqual(normalized(node), normalized(screen_ast()))

    def test_runtime_predicate_rewrite_is_exact_and_unique(self):
        node = rebound_check_ast()
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'runtime_binding_matches' for n in ast.walk(node)), 1)

    def test_runtime_migration_accepts_only_declared_historical_identity(self):
        with patch.dict(globals(), ISOLATED_SHA=digest(ISOLATED / 'python/bin/python3.12')):
            self.assertTrue(runtime_binding_matches(LEGACY_INTERPRETER, LEGACY_SHA))
            with self.assertRaises(ValueError):
                runtime_binding_matches(LEGACY_INTERPRETER, 'wrong historical identity')
        with patch(__name__ + '.digest', return_value='different'):
            self.assertFalse(runtime_binding_matches('experiments/player_representation.py', 'expected'))

    def test_input_tree_unchanged(self):
        node = screen_ast()
        before = normalized(node)
        recovered_ast(node)
        self.assertEqual(before, normalized(node))

    def test_replay_model_root_unchanged(self):
        self.assertIn("Constant(value='neural_replay')", normalized(recovered_ast()))

    def test_external_replay_root_unchanged(self):
        self.assertIn("Constant(value='external_assessment_replay')", normalized(recovered_ast()))

    def test_old_main_root_unchanged(self):
        assignment = [n for n in ast.walk(recovered_ast()) if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'old' for t in n.targets)]
        self.assertEqual(len(assignment), 1)
        self.assertEqual(normalized(assignment[0].value), normalized(ast.parse('ROUND / stage', mode='eval').body))

    def test_no_output_edit_elsewhere(self):
        original = screen_ast()
        changed = recovered_ast()
        count = lambda n, v: sum(isinstance(x, ast.Constant) and x.value == v for x in ast.walk(n))
        self.assertEqual(count(changed, NEW_SUFFIX), 1)
        self.assertEqual(count(original, OLD_SUFFIX), count(changed, OLD_SUFFIX) + 1)

    def test_changed_output_assignment_rejected(self):
        node = screen_ast()
        output_suffix_node(node, OLD_SUFFIX).value = '_different'
        with self.assertRaises(ValueError):
            recovered_ast(node)

    def test_duplicate_screen_rejected(self):
        with self.assertRaises(ValueError):
            screen_ast('def screen(): pass\ndef screen(): pass')

    def test_absent_screen_rejected(self):
        with self.assertRaises(ValueError):
            screen_ast('def other(): pass')

    def test_default_signature_unchanged(self):
        self.assertEqual(normalized(screen_ast().args), normalized(recovered_ast().args))

    def test_original_encode_child_path(self):
        tree = ast.parse(ORIGINAL.read_text())
        encode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
        self.assertIn("Name(id='__file__'", normalized(encode))
        self.assertNotIn(str(SELF), ast.unparse(encode))

    def test_replay_head_branch_loads_instead_of_fit(self):
        branches = [n for n in ast.walk(screen_ast()) if isinstance(n, ast.If)
            and isinstance(n.test, ast.Name) and n.test.id == 'replay'
            and any(isinstance(x, ast.Attribute) and x.attr == 'load_model'
                for item in n.body for x in ast.walk(item))]
        self.assertEqual(len(branches), 1)
        self.assertNotIn('fit_models', ast.unparse(ast.Module(body=branches[0].body, type_ignores=[])))
        self.assertIn('fit_models', ast.unparse(ast.Module(body=branches[0].orelse, type_ignores=[])))

    def test_process_tree_counts_descendants_and_supervisor(self):
        data = f'{os.getpid()} 1 100\n900001 1 200\n900002 900001 300\n900003 900002 400\n900004 1 999\n'
        with patch.object(subprocess, 'check_output', return_value=data):
            self.assertEqual(process_tree_rss(900001), 1000)

    def test_process_tree_handles_worker_exit(self):
        with patch.object(subprocess, 'check_output', return_value=f'{os.getpid()} 1 100\n'):
            self.assertEqual(process_tree_rss(900001), 100)

    def test_all_children_inherit_no_bytecode_writes(self):
        tree = ast.parse(SELF.read_text())
        run_node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        pairs = [(k.value, v.value) for n in ast.walk(run_node) if isinstance(n, ast.Dict)
            for k, v in zip(n.keys, n.values) if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)]
        self.assertIn(('PYTHONDONTWRITEBYTECODE', '1'), pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('test', 'preflight', 'freeze', 'verify', 'run', 'verify-result', '_screen-worker', 'encode'))
    parser.add_argument('--reviewed-source-sha256')
    parser.add_argument('--reviewed-plan-sha256')
    parser.add_argument('--model-folder', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    if args.stage == 'encode':
        if args.model_folder is None or args.destination is None:
            parser.error('Encoder requires model-folder and destination')
        encode_worker(args.model_folder, args.destination, args.parent)
        return
    if args.stage == 'test':
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(PureASTTests)
        if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
            raise SystemExit(1)
        return
    operations = {'preflight': preflight, 'verify': verify_freeze, 'run': run,
        '_screen-worker': screen_worker,
        'verify-result': verify_recovered,
        'freeze': lambda: freeze(args.reviewed_source_sha256, args.reviewed_plan_sha256)}
    print(json.dumps(operations[args.stage](), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
