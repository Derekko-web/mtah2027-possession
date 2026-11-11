"""Complete R51 after its parent-only repeated-thread initialization failure.

Existing first-stage results and the original failed dispatcher remain intact.
The remaining worker uses the original, unchanged frozen R51 executable.
"""
import json
from pathlib import Path

import tabicl_context_rebuild as gate


def main():
    gate.verify()
    binding = gate.ROUND / 'protocol_freeze/orchestration_repair.json'
    record = json.loads(binding.read_text())
    for name, expected in record['sha256'].items():
        if gate.digest(gate.ROOT / name) != expected:
            raise ValueError('Execution-repair dependency changed: ' + name)
    output = gate.ROUND / 'runtime_assessment'
    first = json.loads((output / 'first.supervision.json').read_text())
    if not first['passed'] or first['exit_code'] != 0 or first['aborted']:
        raise ValueError('Original first worker did not pass')
    # The parent needs psutil's import path, not Torch's one-time thread setup.
    # The fresh subprocess invokes the original worker and initializes Torch once.
    gate.initialize = gate.cached.original.runtime_paths
    second = gate.supervise('rebuild', output)
    passed = first['passed'] and second['passed']
    gate.write_json(output / 'summary.json', dict(passed=passed, records=[first, second],
        plan_sha256=gate.digest(gate.PLAN), script_sha256=gate.digest(gate.__file__),
        execution_repair_sha256=gate.digest(__file__),
        synthetic_only=True, native_processing_started=False,
        original_dispatcher_failure_preserved=True, prior_R49_remains_failed=True))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
