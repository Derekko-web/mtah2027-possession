# Round57: supported MPS interface stops at a binary import failure

The current official TabPFN9.2.0/v2 and isolated Torch2.15 interface was frozen
after source review, metadata preflight and hardware-only MPS availability
verification. All60 distributions remain separate from existing runtimes. See
[plan](TABPFN_MPS_RUNTIME_PLAN.json) and
[source-backed feasibility](TABPFN_MPS_V2_FEASIBILITY_20261028_SOURCE_F02B3B5.md).

The first moderate worker exits1 after11.78seconds while importing SciPy.
macOS rejects `_propack/_spropack.cpython-312-darwin.so`: its
`__DATA/__thread_bss` zero-fill section has a nonzero file offset. There is no
timed fit operation, invented-context file, model forward, prediction or native
input. The monitor records a conservative owned-RSS-plus-driver peak below
.361GiB; the resource caps do not cause this failure.

The overall gate is failed. Its frozen files, traceback, supervision and summary
remain under `data/control_tabpfn_mps_v1/`. No probability, performance or native
feasibility claim follows. No score row is added; selected8 and the1,591-row
ledger remain unchanged.

A new, narrowly scoped loader-metadata repair is being investigated in separate
copies. The original failed interface and author wheel bytes will remain exact;
no model setting, package version, backend, precision or replay tolerance is
changed or accepted through this failed run. Any repaired interface requires
its own source review, freeze and synthetic assessment before use.
