"""Source-only contracts for one fixed temporal-block NCA objective.

Run with the private head interpreter in -B -I -S mode. These tests use only
standard-library objects and invented scalar metadata. They never import a
scientific package, create feature arrays, construct a learner or call a fit.
"""
import ast
import copy
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
GATE_PATH = ROOT / "experiments/temporal_block_nca_runtime_gate.py"
AUTHOR_MODEL = ROOT / "data/control_modernnca_isolated_runtime_v1/public_sources/TALENT/model/models/modernNCA.py"
AUTHOR_METHOD = ROOT / "data/control_modernnca_isolated_runtime_v1/public_sources/TALENT/model/methods/modernNCA.py"
SCIENTIFIC_ROOTS = {"torch", "numpy", "scipy", "sklearn", "pandas", "TALENT"}


def load_gate():
    spec = importlib.util.spec_from_file_location("temporal_block_nca_literal_contract", GATE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def method(path, owner, name):
    tree = ast.parse(path.read_bytes())
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body
    return next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)


def dotted_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return dotted_name(node.value) + "." + node.attr
    return ""


def calls(node, wanted):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and dotted_name(n.func) == wanted]


def source_function(gate, name):
    return ast.parse(inspect.getsource(getattr(gate, name))).body[0]


class LiteralTensor:
    """Small scalar/list broadcast fixture, never a scientific tensor or array."""
    def __init__(self, values):
        self.values = copy.deepcopy(values)

    @property
    def shape(self):
        def shape(value):
            return (len(value),) + shape(value[0]) if isinstance(value, list) and value else ()
        return shape(self.values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        if key == (slice(None), None):
            return LiteralTensor([[value] for value in self.values])
        if key == (None, slice(None)):
            return LiteralTensor([self.values])
        raise AssertionError('Unexpected fixture indexing: ' + repr(key))

    def _binary(self, other, operation):
        other = other if isinstance(other, LiteralTensor) else LiteralTensor(other)
        left, right = self.shape, other.shape
        width = max(len(left), len(right))
        a = (1,) * (width-len(left)) + left
        b = (1,) * (width-len(right)) + right
        if any(x != y and x != 1 and y != 1 for x, y in zip(a, b)):
            raise AssertionError('Invalid fixture broadcasting')
        shape = tuple(max(x, y) for x, y in zip(a, b))
        def at(tensor, padded, coordinates):
            value = tensor.values
            for axis in range(width-len(tensor.shape), width):
                value = value[0 if padded[axis] == 1 else coordinates[axis]]
            return value
        def build(axis, coordinates):
            if axis == width:
                return operation(at(self, a, coordinates), at(other, b, coordinates))
            return [build(axis+1, coordinates+(i,)) for i in range(shape[axis])]
        return LiteralTensor(build(0, ()))

    def _unary(self, operation):
        def apply(value):
            return [apply(item) for item in value] if isinstance(value, list) else operation(value)
        return LiteralTensor(apply(self.values))

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a-b)

    def __gt__(self, other):
        return self._binary(other, lambda a, b: a>b)

    def __eq__(self, other):
        return self._binary(other, lambda a, b: a == b)

    def __and__(self, other):
        return self._binary(other, lambda a, b: a and b)

    def __invert__(self):
        return self._unary(lambda a: not a)

    def abs(self):
        return self._unary(abs)

    def _flat(self):
        def flat(value):
            if isinstance(value, list):
                return [x for item in value for x in flat(item)]
            return [value]
        return flat(self.values)

    def sum(self, dim=None):
        if dim == 1:
            return LiteralTensor([sum(row) for row in self.values])
        if dim is None:
            return LiteralTensor(sum(self._flat()))
        raise AssertionError('Unexpected fixture reduction dimension')

    def all(self):
        return all(self._flat())

    def min(self):
        return LiteralTensor(min(self._flat()))

    def item(self):
        if self.shape:
            raise AssertionError('Fixture item requires a scalar')
        return self.values

    def masked_fill(self, mask, replacement):
        return self._binary(mask, lambda value, excluded: replacement if excluded else value)


class TemporalBlockNCASourceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_gate()

    def test_import_and_source_contract_are_standard_library_only(self):
        before = set(sys.modules)
        gate = load_gate()
        contract = gate.source_contract()
        self.assertIsInstance(contract, dict)
        imported = {name.split(".")[0] for name in set(sys.modules) - before}
        self.assertFalse(imported & SCIENTIFIC_ROOTS)
        self.assertFalse({name.split(".")[0] for name in sys.modules} & SCIENTIFIC_ROOTS)

    def test_no_top_level_scientific_import(self):
        imported = set()
        for node in ast.parse(GATE_PATH.read_bytes()).body:
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & SCIENTIFIC_ROOTS)

    def test_original_author_sources_remain_exact(self):
        self.assertEqual(hashlib.sha256(AUTHOR_MODEL.read_bytes()).hexdigest(),
                         "02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f")
        self.assertEqual(hashlib.sha256(AUTHOR_METHOD.read_bytes()).hexdigest(),
                         "608c85e94714718fd5690a8096890b6232f2da94d0c94d5489734278df112d3e")

    def test_exact_author_model_and_optimizer_defaults(self):
        contract = self.gate.source_contract()
        self.assertEqual(contract["model_config"], {
            "dim": 128, "dropout": .1, "d_block": 512, "n_blocks": 0,
            "temperature": 1, "sample_rate": .5,
            "num_embeddings": {"type": "PLREmbeddings", "n_frequencies": 77,
                               "frequency_scale": .04431360576139521,
                               "d_embedding": 34, "lite": True},
        })
        self.assertEqual(contract["training_config"], {"lr": .01, "weight_decay": .0002})

    def test_one_fixed_recipe_shapes_and_budgets(self):
        g = self.gate
        self.assertEqual(g.CONTEXT_ROWS, {"moderate": 600, "full": 3000})
        self.assertEqual(g.QUERY_ROWS, {"moderate": 2500, "full": 2500})
        self.assertEqual((g.EPOCHS, g.BATCH_SIZE, g.QUERY_SIZE, g.CPU_THREADS), (200, 1024, 512, 2))
        self.assertEqual((g.FIT_SECONDS, g.PREDICT_SECONDS, g.WORKER_SECONDS, g.LIMIT_BYTES),
                         (1800, 300, 2400, 6 * 1024 ** 3))
        self.assertEqual(g.STAGES, (("moderate", "first"), ("moderate", "rebuild"),
                                    ("full", "first"), ("full", "rebuild")))
        for key in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(g.ENVIRONMENT[key], "2")
        self.assertEqual(g.ENVIRONMENT["PYTHONHASHSEED"], "0")
        self.assertEqual(g.ENVIRONMENT["PYTORCH_ENABLE_MPS_FALLBACK"], "0")

    def test_external_pool_excludes_all_batch_rows_before_uniform_sampling(self):
        got = self.gate.temporal_candidate_literal(8, [0, 4, 2], [2, 3, 0, 4, 1])
        self.assertEqual(got["external"], [1, 3, 5, 6, 7])
        self.assertEqual(got["sampled"], [5, 6])
        self.assertEqual(got["candidates"], [0, 4, 2, 5, 6])
        self.assertFalse(set(got["external"]) & {0, 4, 2})
        self.assertEqual(len(got["sampled"]), math.floor(.5 * len(got["external"])))

    def test_candidate_sampling_is_row_sampling_with_exact_permutation_order(self):
        # Two different row permutations retain different views of a frame.
        first = self.gate.temporal_candidate_literal(8, [0, 4, 2], [2, 3, 0, 4, 1])
        second = self.gate.temporal_candidate_literal(8, [0, 4, 2], [3, 2, 4, 1, 0])
        self.assertEqual(first["sampled"], [5, 6])
        self.assertEqual(second["sampled"], [6, 5])
        self.assertEqual(first["candidates"][:3], [0, 4, 2])
        self.assertEqual(second["candidates"][:3], [0, 4, 2])

    def test_partial_batch_keeps_author_floor_and_candidate_order(self):
        got = self.gate.temporal_candidate_literal(10, [9, 0, 3], [6, 0, 5, 1, 4, 2, 3])
        self.assertEqual(got["external"], [1, 2, 4, 5, 6, 7, 8])
        self.assertEqual(got["sampled"], [8, 1, 7])
        self.assertEqual(got["candidates"], [9, 0, 3, 8, 1, 7])

    def test_full_batch_has_empty_external_pool_and_unchanged_batch_order(self):
        got = self.gate.temporal_candidate_literal(4, [2, 0, 3, 1], [])
        self.assertEqual(got, {"external": [], "sampled": [], "candidates": [2, 0, 3, 1]})

    def test_batch_and_permutation_corruption_fail_closed(self):
        bad = ((0, [], []), (4, [], [0, 1, 2, 3]), (4, [1, 1], [0, 1]),
               (4, [-1], [0, 1, 2]), (4, [4], [0, 1, 2]),
               (4, [True], [0, 1, 2]), (4, [1], [0, 0, 2]),
               (4, [1], [0, 1]), (4, [1], [0, 1, 3]),
               (4, [1], [0, 1, 2, 3]))
        for total, batch, permutation in bad:
            with self.subTest(total=total, batch=batch, permutation=permutation):
                with self.assertRaises((ValueError, TypeError)):
                    self.gate.temporal_candidate_literal(total, batch, permutation)

    def test_temporal_distance_boundary_250_excluded_251_allowed(self):
        got = self.gate.temporal_mask_literal([10000], [9750, 10250, 9749, 10251, 10000])
        self.assertEqual(got, [[False, False, True, True, False]])

    def test_real_5hz_boundary_250_excluded_255_allowed(self):
        got = self.gate.temporal_mask_literal([10000], [9750, 10250, 9745, 10255])
        self.assertEqual(got, [[False, False, True, True]])

    def test_same_frame_original_and_swapped_mates_masked_in_both_parts(self):
        frames = [0, 5, 500, 505, 0, 5, 500, 505]
        labels = [0, 1, 0, 1, 1, 0, 1, 0]
        schedule = self.gate.temporal_candidate_literal(8, [0, 4, 2], [2, 3, 0, 4, 1])
        candidate = schedule["candidates"]
        mask = self.gate.temporal_mask_literal([frames[i] for i in [0, 4, 2]],
                                               [frames[i] for i in candidate])
        self.assertEqual(mask, [[False, False, True, False, True],
                                [False, False, True, False, True],
                                [True, True, False, True, False]])
        support = self.gate.support_literal(mask, [labels[i] for i in [0, 4, 2]],
                                            [labels[i] for i in candidate])
        self.assertEqual(support, [{"eligible": 2, "positive": 1, "opposite": 1},
                                   {"eligible": 2, "positive": 1, "opposite": 1},
                                   {"eligible": 3, "positive": 2, "opposite": 1}])

    def test_mask_uses_native_frame_values_not_augmented_row_distance(self):
        self.assertEqual(self.gate.temporal_mask_literal([0], [0, 500, 5, 505]),
                         [[False, True, False, True]])
        self.assertNotEqual(self.gate.temporal_mask_literal([0], [0, 500, 5, 505]),
                            self.gate.temporal_mask_literal([0], [0, 1, 2, 3]))

    def test_each_anchor_has_its_own_mask_not_union_of_batch_windows(self):
        self.assertEqual(self.gate.temporal_mask_literal([0, 1000], [0, 1000, 500]),
                         [[False, True, True], [True, False, True]])

    def test_temporal_metadata_must_be_nonboolean_integers(self):
        for anchors, candidates in (([True], [500]), ([0], [False]), ([0.0], [500]),
                                    ([0], [500.0]), ([], [500]), ([0], [])):
            with self.subTest(anchors=anchors, candidates=candidates):
                with self.assertRaises((ValueError, TypeError)):
                    self.gate.temporal_mask_literal(anchors, candidates)

    def test_zero_eligible_support_fails_with_both_labels_present(self):
        with self.assertRaises(self.gate.SupportError):
            self.gate.support_literal([[False, False]], [0], [0, 1])

    def test_zero_own_class_support_fails_without_resampling(self):
        with self.assertRaises(self.gate.SupportError):
            self.gate.support_literal([[True, True]], [0], [1, 1])

    def test_missing_opposite_class_is_counted_and_allowed(self):
        self.assertEqual(self.gate.support_literal([[True, True]], [0], [0, 0]),
                         [{"eligible": 2, "positive": 2, "opposite": 0}])

    def test_actual_mask_helper_masks_both_candidate_parts_and_records_support(self):
        model = types.SimpleNamespace(support_records=[])
        distances = LiteralTensor([[float(10*r+c) for c in range(5)] for r in range(3)])
        args = (model, distances, LiteralTensor([0, 0, 500]), LiteralTensor([0, 0, 500, 5, 500]),
                LiteralTensor([0, 1, 0]), LiteralTensor([0, 1, 0, 0, 1]))
        fingerprint = lambda value: {"shape": list(value.shape), "values": copy.deepcopy(value.values)}
        with patch.object(self.gate, "torch", types.SimpleNamespace(inf=math.inf), create=True), \
                patch.object(self.gate, "array_record", side_effect=fingerprint):
            got = self.gate.apply_temporal_mask(*args)
        self.assertEqual(got.values, [[math.inf, math.inf, 2., math.inf, 4.],
                                      [math.inf, math.inf, 12., math.inf, 14.],
                                      [20., 21., math.inf, 23., math.inf]])
        self.assertEqual(distances.values, [[float(10*r+c) for c in range(5)] for r in range(3)])
        self.assertEqual(len(model.support_records), 1)
        record = model.support_records[0]
        self.assertEqual(record["eligible"]["values"], [2, 2, 3])
        self.assertEqual(record["positive"]["values"], [1, 1, 2])
        self.assertEqual(record["opposite"]["values"], [1, 1, 1])
        self.assertEqual((record["anchors"], record["candidates"], record["minimum_eligible"],
                          record["minimum_positive"], record["missing_opposite"]), (3, 5, 2, 1, 0))

    def test_actual_mask_helper_zero_eligible_or_own_class_fails_before_recording(self):
        fixtures = (([0, 5], [0, 1]), ([500, 505], [1, 1]))
        for candidate_frames, candidate_labels in fixtures:
            model = types.SimpleNamespace(support_records=[])
            with self.subTest(candidate_frames=candidate_frames), \
                    patch.object(self.gate, "torch", types.SimpleNamespace(inf=math.inf), create=True), \
                    patch.object(self.gate, "array_record") as fingerprint:
                with self.assertRaises(self.gate.SupportError):
                    self.gate.apply_temporal_mask(model, LiteralTensor([[0., 1.]]), LiteralTensor([0]),
                                                 LiteralTensor(candidate_frames), LiteralTensor([0]),
                                                 LiteralTensor(candidate_labels))
            self.assertEqual(model.support_records, [])
            fingerprint.assert_not_called()

    def test_actual_mask_helper_missing_opposite_is_allowed_without_altering_probabilities(self):
        model = types.SimpleNamespace(support_records=[])
        fingerprint = lambda value: {"values": copy.deepcopy(value.values)}
        with patch.object(self.gate, "torch", types.SimpleNamespace(inf=math.inf), create=True), \
                patch.object(self.gate, "array_record", side_effect=fingerprint):
            got = self.gate.apply_temporal_mask(model, LiteralTensor([[3., 7.]]), LiteralTensor([0]),
                                               LiteralTensor([500, 505]), LiteralTensor([0]),
                                               LiteralTensor([0, 0]))
        self.assertEqual(got.values, [[3., 7.]])
        self.assertEqual(model.support_records[0]["missing_opposite"], 1)

    def test_support_shape_and_label_errors_fail(self):
        fixtures = (([[True]], [0, 1], [0]), ([[True, False]], [0], [0]),
                    ([[True]], [2], [0]), ([[True]], [0], [2]),
                    ([[1]], [0], [0]))
        for mask, anchors, candidates in fixtures:
            with self.subTest(mask=mask, anchors=anchors, candidates=candidates):
                with self.assertRaises((ValueError, TypeError)):
                    self.gate.support_literal(mask, anchors, candidates)

    def test_support_and_mask_functions_cannot_resample_or_skip(self):
        for name in ("support_literal", "temporal_mask_literal"):
            node = source_function(self.gate, name)
            called = {dotted_name(n.func).split(".")[-1] for n in ast.walk(node) if isinstance(n, ast.Call)}
            self.assertFalse(called & {"randperm", "choice", "choices", "shuffle", "sample"})
            self.assertFalse(any(isinstance(n, (ast.Continue, ast.Break)) for n in ast.walk(node)))

    def test_paired_metadata_equal_frame_groups_and_complemented_targets(self):
        frames = [10000, 10005, 10500, 10000, 10005, 10500]
        self.gate.validate_paired_metadata(frames, [0, 1, 0, 1, 0, 1])

    def test_paired_metadata_allows_native_gaps(self):
        self.gate.validate_paired_metadata([10000, 11000, 10000, 11000], [0, 1, 1, 0])

    def test_paired_metadata_rejects_wrong_groups_order_and_targets(self):
        fixtures = (([], []), ([10000, 10000, 10000], [0, 1, 0]),
                    ([10000, 10000], [0, 0]),
                    ([10000, 10005, 10005, 10000], [0, 1, 1, 0]),
                    ([10005, 10000, 10005, 10000], [0, 1, 1, 0]),
                    ([10000, 10000, 10000, 10000], [0, 1, 1, 0]),
                    ([10001, 10001], [0, 1]), ([True, True], [0, 1]),
                    ([10000, 10000], [0, 2]))
        for frames, labels in fixtures:
            with self.subTest(frames=frames, labels=labels):
                with self.assertRaises((ValueError, TypeError)):
                    self.gate.validate_paired_metadata(frames, labels)

    def test_both_invented_context_shapes_have_contiguous_5hz_paired_centers(self):
        for rows in (600, 3000):
            frames = list(self.gate.invented_frames(rows))
            half = rows // 2
            self.assertEqual(len(frames), rows)
            self.assertEqual(frames[:half], frames[half:])
            self.assertTrue(all(type(f) is int and f % 5 == 0 for f in frames))
            self.assertTrue(all(b - a == 5 for a, b in zip(frames[:half - 1], frames[1:half])))
            labels = [i % 2 for i in range(half)]
            self.gate.validate_paired_metadata(frames, labels + [1 - x for x in labels])
            mask = self.gate.temporal_mask_literal([frames[0], frames[half // 2], frames[half - 1]], frames)
            got = self.gate.support_literal(mask, [labels[0], labels[half // 2], labels[-1]],
                                           labels + [1 - x for x in labels])
            self.assertTrue(all(r["eligible"] > 0 and r["positive"] > 0 and r["opposite"] > 0 for r in got))

    def test_old_64_contiguous_shape_is_intentionally_unsupported(self):
        base = list(range(10000, 10000 + 5 * 32, 5))
        frames = base + base
        mask = self.gate.temporal_mask_literal([frames[0]], frames)
        self.assertFalse(any(mask[0]))
        with self.assertRaises(self.gate.SupportError):
            self.gate.support_literal(mask, [0], [0] * 32 + [1] * 32)

    def test_clock_uses_integer_utc_and_monotonic_ns(self):
        with patch.object(self.gate.time, "time_ns", return_value=1234567890123456789), \
                patch.object(self.gate.time, "monotonic_ns", return_value=9876543210):
            got = self.gate.clock_sample()
        self.assertEqual(got, {"utc_ns": 1234567890123456789, "mono_ns": 9876543210})
        self.assertTrue(all(type(v) is int for v in got.values()))

    def test_elapsed_time_checks_both_clocks_without_float_rounding(self):
        start = {"utc_ns": 1234567890123456789, "mono_ns": 9876543210}
        end = {"utc_ns": start["utc_ns"] + 300000000001, "mono_ns": start["mono_ns"] + 17}
        self.assertEqual(self.gate.elapsed_ns(start, end), {"utc_ns": 300000000001, "mono_ns": 17})
        self.assertIsNotNone(self.gate.operation_violation(start, end, 300))

    def test_host_sleep_calendar_overrun_cannot_be_waived_by_short_monotonic(self):
        start = {"utc_ns": 1000000000000000, "mono_ns": 1000}
        end = {"utc_ns": start["utc_ns"] + 301 * 10 ** 9, "mono_ns": start["mono_ns"] + 1}
        self.assertIsNotNone(self.gate.operation_violation(start, end, 300))

    def test_monotonic_overrun_cannot_be_waived_by_short_calendar(self):
        start = {"utc_ns": 1000000000000000, "mono_ns": 1000}
        end = {"utc_ns": start["utc_ns"] + 1, "mono_ns": start["mono_ns"] + 301 * 10 ** 9}
        self.assertIsNotNone(self.gate.operation_violation(start, end, 300))

    def test_resource_limit_is_inclusive_and_one_nanosecond_over_fails(self):
        start = {"utc_ns": 1000, "mono_ns": 2000}
        exact = {k: v + 300 * 10 ** 9 for k, v in start.items()}
        self.assertIsNone(self.gate.operation_violation(start, exact, 300))
        over = dict(exact)
        over["mono_ns"] += 1
        self.assertIsNotNone(self.gate.operation_violation(start, over, 300))

    def test_clock_rollback_and_noninteger_samples_fail_closed(self):
        start = {"utc_ns": 1000, "mono_ns": 2000}
        for end in ({"utc_ns": 999, "mono_ns": 2001}, {"utc_ns": 1001, "mono_ns": 1999},
                    {"utc_ns": 1001.0, "mono_ns": 2001}, {"utc_ns": True, "mono_ns": 2001}):
            with self.subTest(end=end):
                with self.assertRaises((ValueError, TypeError)):
                    self.gate.elapsed_ns(start, end)

    def test_adapter_and_training_are_constructed_without_executing_them(self):
        forward = self.gate.adapted_forward_node()
        train = self.gate.adapted_train_epoch_node()
        self.assertIsInstance(forward, ast.FunctionDef)
        self.assertIsInstance(train, ast.FunctionDef)
        compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(forward), copy.deepcopy(train)],
                                                     type_ignores=[])), "<source-only-temporal-contract>", "exec")
        self.assertFalse({name.split(".")[0] for name in sys.modules} & SCIENTIFIC_ROOTS)

    def test_enumerated_adapter_changes_restore_both_exact_author_asts(self):
        parity = self.gate.adapter_parity_contract()
        forward = method(AUTHOR_MODEL, "ModernNCA", "forward")
        train = method(AUTHOR_METHOD, "ModernNCAMethod", "train_epoch")
        digest = lambda node: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        self.assertTrue(parity["restored_forward_exact_author_ast"])
        self.assertTrue(parity["restored_train_epoch_exact_author_ast"])
        self.assertEqual(parity["pristine_inference_forward_ast_sha256"], digest(forward))
        self.assertEqual(parity["original_train_epoch_ast_sha256"], digest(train))
        self.assertEqual(self.gate.source_contract()["adapter_parity"], parity)

    def test_adapter_parity_rejects_an_extra_numerical_change(self):
        altered = copy.deepcopy(self.gate.adapted_forward_node())
        distance_call = calls(altered, "torch.cdist")[0]
        next(keyword for keyword in distance_call.keywords if keyword.arg == "p").value = ast.Constant(1)
        with patch.object(self.gate, "adapted_forward_node", return_value=altered):
            with self.assertRaisesRegex(ValueError, "extend beyond"):
                self.gate.adapter_parity_contract()

    def test_candidate_frame_metadata_uses_identical_sample_and_batch_order(self):
        node = self.gate.adapted_forward_node()
        text = ast.unparse(node)
        self.assertIn("candidate_frames = candidate_frames[sample_idx]", text)
        self.assertIn("candidate_frames = torch.cat([anchor_frames, candidate_frames])", text)
        self.assertIn("candidate_labels = candidate_y", text)
        self.assertEqual(len(calls(node, "apply_temporal_mask")), 1)
        self.assertEqual(ast.unparse(calls(node, "apply_temporal_mask")[0]),
                         "apply_temporal_mask(self, distances, anchor_frames, candidate_frames, y, candidate_labels)")

    def test_adapter_keeps_unsquared_cdist_temperature_and_softmax(self):
        original = method(AUTHOR_MODEL, "ModernNCA", "forward")
        adapted = self.gate.adapted_forward_node()
        for wanted in ("torch.cdist", "F.softmax", "torch.mm", "torch.log"):
            self.assertEqual([ast.dump(n, include_attributes=False) for n in calls(adapted, wanted)],
                             [ast.dump(n, include_attributes=False) for n in calls(original, wanted)])
        self.assertEqual(len(calls(adapted, "torch.randperm")), 1)
        self.assertIn("p=2", ast.unparse(calls(adapted, "torch.cdist")[0]))
        self.assertIn("distances / self.T", ast.unparse(adapted))
        self.assertIn("eps = 1e-07", ast.unparse(adapted))

    def test_frame_metadata_cannot_enter_embedding_or_distance_inputs(self):
        node = self.gate.adapted_forward_node()
        feature_calls = []
        for name in ("self.encoder", "self.num_embeddings", "self.post_encoder", "torch.cdist"):
            feature_calls.extend(calls(node, name))
        self.assertTrue(feature_calls)
        for call in feature_calls:
            names = [n.id.lower() for n in ast.walk(call) if isinstance(n, ast.Name)]
            self.assertFalse(any("frame" in name or "metadata" in name for name in names))

    def test_inference_delegates_pristine_author_before_training_metadata_or_mask(self):
        node = self.gate.adapted_forward_node()
        first = node.body[0]
        self.assertIsInstance(first, ast.If)
        self.assertEqual(ast.dump(first.test, include_attributes=False),
                         ast.dump(ast.parse("not is_train", mode="eval").body, include_attributes=False))
        delegated = calls(first, "self.pristine_author_forward")
        self.assertEqual(len(delegated), 1)
        self.assertEqual(ast.unparse(delegated[0]),
                         "self.pristine_author_forward(x, y, candidate_x, candidate_y, is_train)")
        guard = ast.unparse(first.body[0])
        for wanted in ("y is not None", "anchor_frames is not None", "candidate_frames is not None"):
            self.assertIn(wanted, guard)
        called = {dotted_name(n.func) for n in ast.walk(first) if isinstance(n, ast.Call)}
        self.assertFalse({"apply_temporal_mask", "validate_tensor_metadata", "array_record"} & called)

    def test_finite_gradient_validation_occurs_between_backward_and_optimizer_step(self):
        node = self.gate.adapted_train_epoch_node()
        block = next(n for n in node.body if isinstance(n, ast.For))
        positions = {}
        for index, statement in enumerate(block.body):
            for call in (n for n in ast.walk(statement) if isinstance(n, ast.Call)):
                name = dotted_name(call.func)
                if name in ("loss.backward", "finite_gradient_check", "self.optimizer.step"):
                    positions[name] = index
        self.assertEqual(set(positions), {"loss.backward", "finite_gradient_check", "self.optimizer.step"})
        self.assertLess(positions["loss.backward"], positions["finite_gradient_check"])
        self.assertLess(positions["finite_gradient_check"], positions["self.optimizer.step"])

    def test_new_training_keeps_exact_external_pool_before_model_call(self):
        original = method(AUTHOR_METHOD, "ModernNCAMethod", "train_epoch")
        adapted = self.gate.adapted_train_epoch_node()
        wanted = ast.dump(ast.parse("candidate_indices[~torch.isin(candidate_indices, batch_idx)]",
                                   mode="eval").body, include_attributes=False)
        for node in (original, adapted):
            exclusions = [n for n in ast.walk(node) if isinstance(n, ast.Assign)
                          and ast.dump(n.value, include_attributes=False) == wanted]
            self.assertEqual(len(exclusions), 1)
            model_calls = calls(node, "self.model")
            self.assertEqual(len(model_calls), 1)
            # Compare source sequence rather than copied source locations.
            block = next(n for n in node.body if isinstance(n, ast.For))
            self.assertLess(next(i for i, n in enumerate(block.body) if exclusions[0] is n),
                            next(i for i, n in enumerate(block.body) if model_calls[0] in list(ast.walk(n))))
        call = calls(adapted, "self.model")[0]
        keyword_names = {k.arg for k in call.keywords}
        self.assertTrue({"x", "y", "candidate_x", "candidate_y", "is_train"} <= keyword_names)
        self.assertTrue(any("frame" in (name or "") for name in keyword_names))

    def test_pending_admission_rejects_model_work(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.gate, "ROUND", Path(folder)), \
                patch.object(self.gate, "verify", return_value=None, create=True):
            with self.assertRaises((ValueError, RuntimeError, FileNotFoundError, PermissionError)):
                self.gate.require_admission()

    def test_admission_requires_every_true_condition_and_exact_freeze_identity(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(self.gate, "ROUND", Path(folder)):
            root = Path(folder)
            frozen = root / "protocol_freeze/freeze.json"
            frozen.parent.mkdir()
            frozen.write_text('{"invented_only": true}\n')
            valid = {"approved_for_model_calls": True, "scope": "invented_only_temporal_block_nca_v1",
                     "predecessor_gp_closed_or_passed": True, "other_model_workers_stopped": True,
                     "freeze_sha256": hashlib.sha256(frozen.read_bytes()).hexdigest()}
            path = root / "resource_admission.json"
            path.write_text(json.dumps(valid))
            self.assertEqual(self.gate.require_admission(), valid)
            for key in valid:
                broken = dict(valid)
                broken[key] = False if isinstance(valid[key], bool) else "wrong"
                path.write_text(json.dumps(broken))
                with self.subTest(key=key), self.assertRaises(PermissionError):
                    self.gate.require_admission()
            path.write_text(json.dumps(valid))
            frozen.write_text('{"invented_only": false}\n')
            with self.assertRaises(PermissionError):
                self.gate.require_admission()

    def test_resource_aborted_worker_cannot_pass_preexisting_success_summary(self):
        process = Mock(pid=680001)
        process.poll.return_value = None
        process.wait.return_value = 0
        clock = {"utc_ns": 1000000000000000, "mono_ns": 1000000000}
        oversize = types.SimpleNamespace(returncode=0, stdout=str(self.gate.LIMIT_BYTES // 1024 + 1))
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            destination = output / "moderate/first"
            destination.mkdir(parents=True)
            (destination / "summary.json").write_text('{"passed": true, "peak_bytes": 1}')
            with patch.object(self.gate.subprocess, "Popen", return_value=process), \
                    patch.object(self.gate.subprocess, "run", return_value=oversize), \
                    patch.object(self.gate, "clock_sample", return_value=clock), \
                    patch.object(self.gate.time, "sleep"):
                got = self.gate.supervise("moderate", "first", output)
        self.assertFalse(got["passed"])
        self.assertEqual(got["aborted"], "rss_over_6GiB")
        self.assertEqual(got["exit_code"], 0)
        process.terminate.assert_called_once()

    def test_supervisor_checks_worker_and_active_operation_dual_clocks(self):
        start = {"utc_ns": 1000000000000000, "mono_ns": 1000000000}
        worker_over = {"utc_ns": start["utc_ns"] + 2401 * 10 ** 9,
                       "mono_ns": start["mono_ns"] + 1}
        self.assertEqual(self.gate.supervision_violation(start, worker_over, {}, 0),
                         "worker_utc_over_2400seconds")
        query_over = {"utc_ns": start["utc_ns"] + 301 * 10 ** 9,
                      "mono_ns": start["mono_ns"] + 1}
        status = {"active": True, "kind": "predict", "started": start, "limit_seconds": 300}
        self.assertEqual(self.gate.supervision_violation(start, query_over, status, 0),
                         "predict_utc_over_300seconds")
        status["limit_seconds"] = 301
        with self.assertRaises(ValueError):
            self.gate.supervision_violation(start, query_over, status, 0)

    def test_operation_clock_overrun_is_recorded_before_failure(self):
        start = {"utc_ns": 1000000000000000, "mono_ns": 1000000000}
        end = {"utc_ns": start["utc_ns"] + 300 * 10 ** 9 + 1,
               "mono_ns": start["mono_ns"] + 1}
        callback = Mock(return_value="invented scalar result")
        basis = types.SimpleNamespace(peak_bytes=lambda: 1)
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(self.gate, "clock_sample", side_effect=[start, end]), \
                patch.object(self.gate, "basis_module", return_value=basis):
            output = Path(folder)
            with self.assertRaises(RuntimeError):
                self.gate.operation(output, "invented_prediction", "predict", callback)
            status = json.loads((output / "operation.json").read_text())
            events = (output / "operations.jsonl").read_text().splitlines()
        callback.assert_called_once_with()
        self.assertTrue(status["failed"])
        self.assertFalse(status["active"])
        self.assertEqual(status["violation"], "utc_over_300seconds")
        self.assertEqual(json.loads(events[0]), status)

    def test_no_source_contract_requires_unfinished_gp_outputs(self):
        node = source_function(self.gate, "require_admission")
        text = ast.unparse(node)
        self.assertNotIn("control_gaussian_process", text)
        self.assertNotIn("runtime_assessment", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
