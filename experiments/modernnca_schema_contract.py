"""R64: derive the exact R54/R37 166-column team swap without scientific imports.

Only source AST and small literal sentinels are evaluated. No feature builder,
tracking observation, target array, fitted artifact or scientific module loads.
"""
import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SHA256 = {
    "experiments/train_factorized_baseline.py": "6a14a94c6fce6055bc5ee8403106dbd6d23090cf61e823c46e531001d897705c",
    "experiments/ball_kinematics.py": "4214d9ce10c2b5f6db3a8707a3986aeb42513067101644cc120533ce0fc1bff4",
    "experiments/relative_motion.py": "8e02e9db1da28e17151cb222382bf668d549628ca6d11bfd65ce5fa0a2c1fc74",
    "experiments/temporal_control.py": "a2696c8992604a8ca088128db575b6c4566d294e76ed7aa2b75c46bdabce8682",
    "experiments/temporal_candidate.py": "f04ce5483862e27f483753e4ac47db309af9adb487d3e3357f264890c72b5a79",
    "experiments/rich_features.py": "172d566dacef281260d153f1310f3cfe9b48380d3d8f16fe779e45994b0cdc04",
    "experiments/player_representation.py": "def15856a29d9ae060e963a00be1c662dc99119606118af4c1903d8978acfebf",
}


def _tree(relative):
    path = ROOT / relative
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256[relative]:
        raise ValueError("Frozen schema builder source changed: " + relative)
    return ast.parse(payload, filename=str(path))


def _definition(tree, name):
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def _load_definition(relative, name):
    definition = _definition(_tree(relative), name)
    namespace = {"__builtins__": {}}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), relative, "exec"), namespace)
    return namespace[name], hashlib.sha256(ast.dump(definition).encode()).hexdigest()


def _literal(node, names):
    if isinstance(node, ast.Name):
        return names[node.id]
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal(item, names) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _literal(node.left, names) + _literal(node.right, names)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_literal(node.operand, names)
    if isinstance(node, ast.JoinedStr):
        return "".join(str(_literal(item, names)) for item in node.values)
    if isinstance(node, ast.FormattedValue):
        value = _literal(node.value, names)
        if node.conversion != -1:
            raise ValueError("Unsupported source name conversion")
        return format(value, _literal(node.format_spec, names) if node.format_spec else "")
    raise ValueError("Nonliteral name expression in schema AST: " + ast.dump(node))


def _assign_loop_target(target, value, names):
    if isinstance(target, ast.Name):
        names[target.id] = value
    elif isinstance(target, (ast.Tuple, ast.List)) and len(target.elts) == len(value):
        for node, item in zip(target.elts, value):
            _assign_loop_target(node, item, names)
    else:
        raise ValueError("Unsupported literal schema loop target")


def _dictionary_column_names(statements, dictionary, names=None):
    """Walk only literal loops/name assignments, collecting dict insertion order.

    Numeric expressions and builder calls are never evaluated. Variable dict
    assignment targets are the only part of numeric builder statements used.
    """
    names = {} if names is None else names.copy()
    result = []
    for statement in statements:
        if isinstance(statement, ast.For):
            try:
                values = _literal(statement.iter, names)
            except (KeyError, ValueError):
                continue
            for value in values:
                local = names.copy()
                _assign_loop_target(statement.target, value, local)
                result.extend(_dictionary_column_names(statement.body, dictionary, local))
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == dictionary:
                    result.append(_literal(target.slice, names))
                elif isinstance(target, ast.Name):
                    try:
                        names[target.id] = _literal(statement.value, names)
                    except (KeyError, ValueError):
                        pass
    return result


class _Loc:
    def __init__(self, frame):
        self.frame = frame

    def __getitem__(self, key):
        rows, sources = key
        if rows != slice(None):
            raise ValueError("Sentinel supports whole-row column selection only")
        return _Frame(sources, [self.frame.values[self.frame.columns.index(name)] for name in sources])


class _Frame:
    def __init__(self, columns, values):
        self.columns = list(columns)
        self.values = list(values)

    @property
    def loc(self):
        return _Loc(self)

    def copy(self):
        return _Frame(self.columns, self.values)


def validate_signed_involution(permutation, signs):
    n = len(permutation)
    if len(signs) != n or sorted(permutation) != list(range(n)) or any(s not in (-1, 1) for s in signs):
        raise ValueError("Invalid signed permutation")
    if any(permutation[permutation[i]] != i or signs[i] * signs[permutation[i]] != 1 for i in range(n)):
        raise ValueError("Signed swap does not compose to identity")
    return True


def schema_contract():
    compact, compact_sha = _load_definition("experiments/train_factorized_baseline.py", "compact_columns")
    swap, swap_sha = _load_definition("experiments/rich_features.py", "swap_teams")
    columns = compact(None)
    if len(columns) != 32:
        raise ValueError("Compact schema no longer has 32 columns")

    kinematics = _definition(_tree("experiments/ball_kinematics.py"), "build_kinematics")
    named = [node for node in ast.walk(kinematics) if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == "features" for t in node.targets)]
    if len(named) != 1 or not isinstance(named[0].value, ast.Call) or not isinstance(named[0].value.args[0], ast.Dict):
        raise ValueError("Unexpected kinematics schema expression")
    kinetic_columns = [ast.literal_eval(key) for key in named[0].value.args[0].keys]
    relative = _definition(_tree("experiments/relative_motion.py"), "build_relative_motion")
    relative_columns = _dictionary_column_names(relative.body, "columns")
    control_tree = _tree("experiments/temporal_control.py")
    contact_columns = _dictionary_column_names(_definition(control_tree, "contact_memory").body, "result")
    temporal_columns = _dictionary_column_names(_definition(control_tree, "temporal_taps").body, "result")
    if (len(kinetic_columns), len(relative_columns), len(contact_columns), len(temporal_columns)) != (6, 8, 10, 48):
        raise ValueError("Builder column counts no longer match R54")
    columns += kinetic_columns + relative_columns + contact_columns + temporal_columns

    ordered = _definition(_tree("experiments/temporal_candidate.py"), "ordered_inputs")
    returned = next(node.value for node in ordered.body if isinstance(node, ast.Return))
    if not isinstance(returned, ast.Call) or not isinstance(returned.func, ast.Attribute) or returned.func.attr != "drop":
        raise ValueError("Unexpected ordered-input boundary drop")
    dropped = next(ast.literal_eval(value.value) for value in returned.keywords if value.arg == "columns")
    if dropped != ["ball_goal_line_distance", "ball_touchline_distance"]:
        raise ValueError("Ordered-input omitted columns changed")
    columns = [name for name in columns if name not in dropped]
    if len(columns) != 102:
        raise ValueError("Ordered schema no longer has 102 columns")

    latent_tree = _tree("experiments/player_representation.py")
    latent = next(node for node in latent_tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "COLUMNS" for t in node.targets))
    namespace = {"__builtins__": {"range": range}}
    exec(compile(ast.Module(body=[latent], type_ignores=[]), "exact_latent_columns", "exec"), namespace)
    latent_columns = namespace["COLUMNS"]
    if len(latent_columns) != 64:
        raise ValueError("R37 latent schema no longer has 64 columns")
    columns += latent_columns
    if len(columns) != 166 or len(set(columns)) != 166:
        raise ValueError("Rich/latent166 schema is not unique")

    sentinel = _Frame(columns, range(1, 167))
    transformed = swap(sentinel)
    if transformed.columns != columns or sentinel.values != list(range(1, 167)):
        raise ValueError("Exact source swap changed columns or mutated sentinel input")
    permutation = [abs(value) - 1 for value in transformed.values]
    signs = [1 if value > 0 else -1 for value in transformed.values]
    validate_signed_involution(permutation, signs)
    if signs != [1] * 166 or swap(transformed).values != sentinel.values:
        raise ValueError("R54 swap has unexpected signs or fails exact double application")
    if permutation[102:134] != list(range(134, 166)) or permutation[134:166] != list(range(102, 134)):
        raise ValueError("Latent team halves do not exchange exactly")
    return {"columns": columns, "permutation": permutation, "signs": signs,
            "source_sha256": SOURCE_SHA256.copy(),
            "definition_sha256": {"compact_columns": compact_sha, "swap_teams": swap_sha},
            "component_counts": {"compact": 32, "kinematics": 6, "relative_motion": 8,
                                 "contact_memory": 10, "temporal_taps": 48,
                                 "dropped_boundaries": 2, "latent": 64},
            "scientific_imports": False, "actual_arrays_read": False}
