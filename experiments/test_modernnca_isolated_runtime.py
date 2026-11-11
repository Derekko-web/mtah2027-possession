"""R64 source-only contracts. No scientific package, tensors, model or fit.

These tests protect the candidate-index/self-neighbor seam and prove that the
standalone wrapper binds the particular author recipe inspected in research.
"""
import ast
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/control_modernnca_isolated_runtime_v1/public_sources"
GATE = ROOT / "experiments/modernnca_isolated_runtime_gate.py"
AUTHOR_MODEL_SHA = "02fce6a107998ab7212774507998e77535f630fbf4ee328acf8519ad7c10632f"
AUTHOR_METHOD_SHA = "608c85e94714718fd5690a8096890b6232f2da94d0c94d5489734278df112d3e"
AUTHOR_HELPER_SHA = "46f26474f559cdda4eda92c5d28d504905c3180ad0ead1bc3f296af72c74230b"
AUTHOR_DEFAULT_SHA = "ea4dc58f92ea9c5d0be7e1c35b38d28d247b6d3c589da10e16c2dcd951a74700"


def load_gate():
    spec = importlib.util.spec_from_file_location("r64_contract_gate", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_schema():
    path = ROOT / "experiments/modernnca_schema_contract.py"
    spec = importlib.util.spec_from_file_location("r64_literal_schema", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_with_hash(wanted):
    matches = [p for p in SOURCES.rglob("*") if p.is_file()
               and hashlib.sha256(p.read_bytes()).hexdigest() == wanted]
    if len(matches) != 1:
        raise AssertionError(f"Expected exactly one acquired author source: {wanted}")
    return matches[0]


def method(tree, class_name, function_name):
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == function_name)


class ModernNCASourceContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gate = load_gate()

    def test_import_and_source_contract_do_not_import_scientific_packages(self):
        before = set(sys.modules)
        load_gate().source_contract()
        imported = set(sys.modules) - before
        roots = {name.split(".")[0] for name in imported}
        self.assertFalse(roots & {"torch", "numpy", "scipy", "sklearn", "pandas", "TALENT"})

    def test_no_top_level_scientific_import(self):
        tree = ast.parse(GATE.read_text())
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertFalse(imported & {"torch", "numpy", "scipy", "sklearn", "pandas", "TALENT"})

    def test_exact_author_model_helper_and_method_are_acquired(self):
        for wanted in (AUTHOR_MODEL_SHA, AUTHOR_METHOD_SHA, AUTHOR_HELPER_SHA, AUTHOR_DEFAULT_SHA):
            source_with_hash(wanted)

    def test_official_default_configuration(self):
        config = json.loads(source_with_hash(AUTHOR_DEFAULT_SHA).read_text())["modernNCA"]
        self.assertEqual(config["model"], {
            "dim": 128, "dropout": .1, "d_block": 512, "n_blocks": 0,
            "temperature": 1, "num_embeddings": {
                "type": "PLREmbeddings", "n_frequencies": 77,
                "frequency_scale": .04431360576139521, "d_embedding": 34, "lite": True,
            }, "sample_rate": .5,
        })
        self.assertEqual(config["training"], {"lr": .01, "weight_decay": .0002})

    def test_source_contract_binds_exact_model_and_optimizer_defaults(self):
        contract = self.gate.source_contract()
        config = json.loads(source_with_hash(AUTHOR_DEFAULT_SHA).read_text())["modernNCA"]
        self.assertEqual(contract["model_config"], config["model"])
        self.assertEqual(contract["training_config"], config["training"])
        self.assertTrue(contract["definitions"])

    def test_definition_records_bind_exact_author_ast(self):
        contract = self.gate.source_contract()
        self.assertEqual(set(contract["definitions"]), {
            "ModernNCA", "make_module", "PLREmbeddings", "make_random_batches", "train_epoch", "Averager",
        })
        for record in contract["definitions"].values():
            nodes = ast.parse((ROOT / record["path"]).read_bytes()).body
            if record["parent"]:
                nodes = next(n for n in nodes if isinstance(n, ast.ClassDef) and n.name == record["parent"]).body
            node = next(n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == record["name"])
            self.assertEqual(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(), record["ast_sha256"])

    def test_extracted_training_code_matches_independent_author_compilation(self):
        fake_torch = types.SimpleNamespace(device=type("LiteralDevice", (), {}))
        actual = self.gate.extracted_definitions(fake_torch)
        contract = self.gate.source_contract()["definitions"]
        nodes = []
        for key in ("Averager", "make_random_batches", "train_epoch"):
            record = contract[key]
            choices = ast.parse((ROOT / record["path"]).read_bytes()).body
            if record["parent"]:
                choices = next(n for n in choices if isinstance(n, ast.ClassDef) and n.name == record["parent"]).body
            nodes.append(next(n for n in choices if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == record["name"]))
        from typing import Optional
        expected = {"torch": fake_torch, "Optional": Optional, "__name__": "r64_exact_author_training"}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     "<exact-pinned-author-training>", "exec"), expected)
        for name in ("make_random_batches", "train_epoch"):
            self.assertEqual(actual[name].__code__, expected[name].__code__)
        for name, value in vars(expected["Averager"]).items():
            if isinstance(value, types.FunctionType):
                self.assertEqual(getattr(actual["Averager"], name).__code__, value.__code__)

    def test_prospective_resources_shapes_and_environment_match_plan(self):
        gate = self.gate
        plan = json.loads(gate.PLAN.read_text())
        self.assertEqual(gate.CONTEXT_ROWS, {"moderate": 64, "full": 3000})
        self.assertEqual((gate.EPOCHS, gate.BATCH_SIZE, gate.QUERY_SIZE, gate.CPU_THREADS), (200, 1024, 512, 2))
        self.assertEqual((gate.FIT_SECONDS, gate.PREDICT_SECONDS, gate.WORKER_SECONDS, gate.LIMIT_BYTES),
                         (1800, 300, 2400, 6 * 1024 ** 3))
        self.assertEqual(plan["resources"], {
            "cpu_threads": 2, "fit_seconds": 1800, "prediction_seconds": 300,
            "rss_bytes": 6 * 1024 ** 3, "worker_seconds": 2400, "workers": 1,
        })
        self.assertEqual(gate.ENVIRONMENT["PYTHONHASHSEED"], "0")
        for name in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                     "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(gate.ENVIRONMENT[name], "2")

    def test_resource_aborted_worker_cannot_pass_existing_success_summary(self):
        gate = self.gate
        process = Mock(pid=640064)
        process.poll.return_value = None
        process.wait.return_value = 0
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            destination = output / "moderate" / "first"
            destination.mkdir(parents=True)
            (destination / "summary.json").write_text('{"passed": true}')
            oversize = types.SimpleNamespace(returncode=0, stdout=str(gate.LIMIT_BYTES // 1024 + 1))
            with patch.object(gate.subprocess, "Popen", return_value=process), \
                    patch.object(gate.subprocess, "run", return_value=oversize), \
                    patch.object(gate.time, "sleep"):
                record = gate.supervise("moderate", "first", output)
        self.assertFalse(record["passed"])
        self.assertEqual(record["aborted"], "rss_over_6GiB")
        self.assertEqual(record["exit_code"], 0)
        process.terminate.assert_called_once()

    def test_main_dispatches_four_stages_sequentially_and_requires_refit(self):
        gate = self.gate
        expected = (("moderate", "first"), ("moderate", "rebuild"), ("full", "first"), ("full", "rebuild"))
        self.assertEqual(gate.STAGES, expected)
        calls = []
        def fake_supervise(size, stage, output):
            calls.append((size, stage))
            if stage == "rebuild":
                destination = output / size / stage
                destination.mkdir(parents=True)
                gate.write_json(destination / "summary.json", {"checks": {"independent_refit_exact": True}})
            return {"passed": True}
        with tempfile.TemporaryDirectory() as folder, patch.object(gate, "verify"), \
                patch.object(gate, "supervise", side_effect=fake_supervise):
            output = Path(folder) / "assessment"
            gate.main(output)
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(calls, list(expected))
        self.assertTrue(summary["passed"])
        self.assertFalse(summary["native_plan_authorized"])

    def test_main_stops_immediately_on_failed_stage(self):
        gate = self.gate
        with tempfile.TemporaryDirectory() as folder, patch.object(gate, "verify"), \
                patch.object(gate, "supervise", return_value={"passed": False}) as supervise:
            output = Path(folder) / "assessment"
            with self.assertRaises(SystemExit) as raised:
                gate.main(output)
            summary = json.loads((output / "summary.json").read_text())
        self.assertEqual(raised.exception.code, 1)
        self.assertEqual(supervise.call_count, 1)
        self.assertFalse(summary["passed"])

    def test_four_completed_stages_still_require_exact_fresh_refit(self):
        gate = self.gate
        def fake_supervise(size, stage, output):
            if stage == "rebuild":
                destination = output / size / stage
                destination.mkdir(parents=True)
                gate.write_json(destination / "summary.json", {"checks": {"independent_refit_exact": False}})
            return {"passed": True}
        with tempfile.TemporaryDirectory() as folder, patch.object(gate, "verify"), \
                patch.object(gate, "supervise", side_effect=fake_supervise) as supervise:
            with self.assertRaisesRegex(ValueError, "lacks exact independent refit"):
                gate.main(Path(folder) / "assessment")
        self.assertEqual(supervise.call_count, 4)

    def test_literal_external_pool_excludes_complete_current_batch(self):
        self.assertEqual(self.gate.candidate_indices_literal(6, [1, 4]), [0, 2, 3, 5])
        self.assertEqual(self.gate.candidate_indices_literal(64, list(range(64))), [])
        self.assertEqual(self.gate.candidate_indices_literal(3000, list(range(2048, 3000))), list(range(2048)))

    def test_literal_sampling_then_append_then_diagonal_removal_has_no_self(self):
        batch = [1, 4]
        external = self.gate.candidate_indices_literal(6, batch)
        # A fixed stand-in permutation selects external indices 3,0. The
        # author forward prepends the mini-batch, then masks its diagonal.
        sampled = [external[k] for k in [2, 0]]
        combined = batch + sampled
        self.assertEqual(combined, [1, 4, 3, 0])
        for row, query_index in enumerate(batch):
            unmasked = [index for column, index in enumerate(combined) if column != row]
            self.assertNotIn(query_index, unmasked)
        # Omitting the wrapper's exclusion lets an appended self label
        # survive the model's correct diagonal mask: this is the regression.
        broken_combined = batch + [3, 1]
        self.assertIn(1, broken_combined[1:])

    def test_exact_author_training_exclusion_is_before_forward(self):
        tree = ast.parse(source_with_hash(AUTHOR_METHOD_SHA).read_text())
        train_epoch = method(tree, "ModernNCAMethod", "train_epoch")
        expected = ast.dump(ast.parse("candidate_indices[~torch.isin(candidate_indices, batch_idx)]", mode="eval").body)
        exclusions = [n for n in ast.walk(train_epoch) if isinstance(n, ast.Assign)
                      and ast.dump(n.value) == expected]
        self.assertEqual(len(exclusions), 1)
        calls = [n for n in ast.walk(train_epoch) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "model"]
        self.assertEqual(len(calls), 1)
        self.assertLess(exclusions[0].lineno, calls[0].lineno)

    def test_literal_exclusion_matches_executed_author_expression(self):
        class Mask(list):
            def __invert__(self):
                return Mask(not value for value in self)

        class Indices(list):
            def __getitem__(self, key):
                if isinstance(key, Mask):
                    return Indices(value for value, keep in zip(self, key) if keep)
                return super().__getitem__(key)

        class LiteralTorch:
            @staticmethod
            def isin(indices, batch):
                return Mask(value in batch for value in indices)

        tree = ast.parse(source_with_hash(AUTHOR_METHOD_SHA).read_text())
        train_epoch = method(tree, "ModernNCAMethod", "train_epoch")
        expected = ast.dump(ast.parse("candidate_indices[~torch.isin(candidate_indices, batch_idx)]", mode="eval").body)
        value = next(n.value for n in ast.walk(train_epoch) if isinstance(n, ast.Assign)
                     and ast.dump(n.value) == expected)
        expression = compile(ast.Expression(value), "exact_author_exclusion", "eval")
        for count, batch in [(6, [1, 4]), (7, [0, 2, 6]), (64, list(range(64)))]:
            author_result = eval(expression, {"torch": LiteralTorch}, {
                "candidate_indices": Indices(range(count)), "batch_idx": batch,
            })
            self.assertEqual(self.gate.candidate_indices_literal(count, batch), list(author_result))

    def test_exact_author_forward_uses_unsquared_euclidean_and_diagonal_mask(self):
        tree = ast.parse(source_with_hash(AUTHOR_MODEL_SHA).read_text())
        forward = method(tree, "ModernNCA", "forward")
        cdist = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "cdist"]
        self.assertEqual(len(cdist), 1)
        self.assertEqual({k.arg: ast.literal_eval(k.value) for k in cdist[0].keywords}, {"p": 2})
        mask = [n for n in ast.walk(forward) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "fill_diagonal_"]
        self.assertEqual(len(mask), 1)
        self.assertEqual(ast.dump(mask[0].args[0]), ast.dump(ast.parse("torch.inf", mode="eval").body))
        self.assertFalse(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                             and n.func.attr in {"square", "pow"} for n in ast.walk(forward)))


class ModernNCASchemaContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = load_schema()

    def test_literal166_schema_has_exact_component_counts_and_order(self):
        contract = self.schema.schema_contract()
        columns = contract["columns"]
        self.assertEqual(len(columns), 166)
        self.assertEqual(len(set(columns)), 166)
        self.assertEqual(columns[:6], ["ball_seen", "ball_speed", "ball_displacement_0.2s",
                                      "ball_displacement_1s", "ball_speed__past_1s", "ball_speed__future_1s"])
        self.assertEqual(columns[30:36], ["ball_acceleration", "ball_acceleration_past_1s",
                                        "ball_acceleration_future_1s", "ball_speed_std_past_1s",
                                        "ball_speed_std_future_1s", "ball_direction_coherence"])
        self.assertEqual(columns[102:], [f"{team}__set_latent_{i:02}" for team in ("h", "a") for i in range(32)])
        self.assertNotIn("ball_goal_line_distance", columns)
        self.assertNotIn("ball_touchline_distance", columns)
        self.assertEqual(contract["component_counts"], {
            "compact": 32, "kinematics": 6, "relative_motion": 8, "contact_memory": 10,
            "temporal_taps": 48, "dropped_boundaries": 2, "latent": 64,
        })

    def test_exact_source_swap_exchanges_names_without_sign_changes(self):
        contract = self.schema.schema_contract()
        columns, permutation = contract["columns"], contract["permutation"]
        self.assertEqual(contract["signs"], [1] * 166)
        self.assertEqual([permutation[i] for i in permutation], list(range(166)))
        self.assertEqual(sum(i == v for i, v in enumerate(permutation)), 24)
        for destination, source in enumerate(permutation):
            name = columns[destination]
            expected = (("a__" if name.startswith("h__") else "h__") + name[3:]
                        if name.startswith(("h__", "a__")) else name)
            self.assertEqual(columns[source], expected)

    def test_schema_binds_all_seven_source_builders_without_importing_them(self):
        before = set(sys.modules)
        contract = self.schema.schema_contract()
        self.assertEqual(len(contract["source_sha256"]), 7)
        for relative, expected in contract["source_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(), expected)
        self.assertFalse({n.split(".")[0] for n in set(sys.modules) - before}
                         & {"numpy", "pandas", "torch", "sklearn"})

    def test_generic_signed_involution_rejects_inconsistent_pair_signs(self):
        self.assertTrue(self.schema.validate_signed_involution([1, 0, 2], [-1, -1, 1]))
        with self.assertRaises(ValueError):
            self.schema.validate_signed_involution([1, 0, 2], [-1, 1, 1])
        with self.assertRaises(ValueError):
            self.schema.validate_signed_involution([1, 1, 2], [1, 1, 1])

    def test_generic_signed_train_only_imputation_and_scaling_fixture(self):
        # Literal fixture for any future signed-column preprocessing. Actual
        # R54 schema above contains only positive signs. No arrays are loaded.
        permutation, signs = [1, 0, 2], [-1, -1, 1]
        rows = [[1., None, 3.], [2., 1., 5.]]

        def swap(row):
            return [None if row[j] is None else signs[i] * row[j] for i, j in enumerate(permutation)]

        augmented = rows + [swap(row) for row in rows]
        mean = [0.] * 3
        scale = [0.] * 3
        seen = set()
        for column, other in enumerate(permutation):
            if column in seen:
                continue
            values = [row[column] for row in augmented if row[column] is not None]
            own_mean = math.fsum(values) / len(values) if values else 0.
            filled = [row[column] if row[column] is not None else own_mean for row in augmented]
            own_scale = math.sqrt(math.fsum((v - own_mean) ** 2 for v in filled) / len(filled)) or 1.
            mean[column], mean[other] = own_mean, signs[column] * own_mean
            scale[column] = scale[other] = own_scale
            seen.update((column, other))

        def transform(row):
            return [((mean[i] if value is None else value) - mean[i]) / scale[i]
                    for i, value in enumerate(row)]

        self.assertEqual(mean[1], -mean[0])
        self.assertEqual(scale[1], scale[0])
        for row in rows:
            self.assertEqual(transform(swap(row)), swap(transform(row)))


if __name__ == "__main__":
    unittest.main()
