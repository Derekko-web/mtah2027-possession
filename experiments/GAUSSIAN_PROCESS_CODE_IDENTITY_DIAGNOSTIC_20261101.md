# GP v1 code-identity admission failure: serialization, not changed code

The frozen v1 technical gate remains **failed and ineligible**. Its first
moderate worker exited1 at the code-identity guard before the first author
`GaussianProcessClassifier.fit` or posterior calculation. It had generated and
preprocessed invented data. The external record covers3.762702seconds UTC and
189,054,976 sampled RSS bytes. No native data or possession score was processed.
Nothing in this diagnosis turns that failure into a pass or permits a rerun of
its frozen output.

The separate [diagnostic JSON](../data/control_gaussian_process_code_identity_diagnostic_v1/diagnostic.json)
proves that the guard rejected different serialization of equal code values.
Its SHA256 is
`a17d4613cadd52cd46b424761fb4d0cc5afa3094e92cf399e0a7e59be73ec825`.
It contains the exact probe source, complete public code attributes, raw marshal
payloads, decoded full code payloads, primary-source hashes and before/after
integrity evidence. Its interpreter used the frozen private `-B -P -S` identity,
fixed environment and package paths. Imports were limited to obtaining module
and code metadata. The probe instantiated no GP/kernel/preprocessor and called
no fitting, inference or native-array loader. It changed no author code, code
object, trace setting, installed runtime, frozen input or failed assessment.

## Observed identity

Both code objects name the same actual author file and first line443. All20
available noncallable `co_*` attributes match by typed value, including bytecode,
constants, flags, filename, names, qualification, argument counts, variable/free/
cell names, stack size, first line, line table, deprecated line table and
exception table. `code == pristine_code` is true. Derived `co_lines()` and
`co_positions()` match. Adaptive byte sequences also happen to match in this
probe. `sys.gettrace()` is `None`.

The actual/pristine bytecode SHA256 is
`dcd3f34323d356d0e561789729dadeb5970636e1c3e63c62c401b214193d15c7`;
their line-table SHA256 is
`15e72a4c32aa90a69e11a99441d0962414bbd436e512058ddf490fb893bead55`.
The author source remains the pinned39,387-byte file with SHA256
`90628fca68b26aac2c4f57305b6302c549952d3873e6651209ffcafb2c95ff3d`.
The cached import path was recorded; no filename discrepancy was found.

Each bound default-marshal payload is2,481bytes, with exactly five differing
byte positions:

| Offset | Actual / pristine bytes | Decoded location and difference |
|---|---|---|
|1146|`e1` / `61`|docstring constant: same ASCII payload, reference flag present/absent|
|1479|`e7` / `67`|constant8 (`-0.5`): same binary float, reference flag present/absent|
|1526|`03` / `02`|same name reached through different reference-table indices|
|1806|`fa` / `7a`|same filename string, reference flag present/absent|
|1938|`fa` / `7a`|same qualified-name string, reference flag present/absent|

A read-only parser followed the pinned CPython `TYPE_CODE` field order and
consumed both complete payloads. After resolving only the encoded object
references for inspection, every stored code field matched, including internal
locals-plus names/kinds. Neither original code object was replaced or normalized.
The raw serialization hashes differed; both offending marshal guard terms were
true, while the unrelated active-trace term was false.

## Primary implementation explanation

CPython3.12.14's `w_ref` consults object reference counts and a pointer-keyed
reference table. It can mark an object with `FLAG_REF=0x80` and later emit that
table's index. String tags also distinguish interning. Its `TYPE_CODE` writer
serializes code fields, including filename, internal locals-plus fields and
line/exception tables. Thus equality of complete code values does not guarantee
equal default-marshal reference encoding. The observed four0x80 tag differences
and reference-index shift agree exactly with that mechanism.
[Pinned CPython marshal implementation](https://github.com/python/cpython/blob/v3.12.14/Python/marshal.c).

The Python documentation describes marshal as an internal format used mainly
for `.pyc` files; it provides no canonical-value-byte guarantee.
[Official marshal documentation](https://docs.python.org/3.12/library/marshal.html).
Code attributes and position/line iterators are documented separately.
[Official code-object data model](https://docs.python.org/3.12/reference/datamodel.html#code-objects).
The live documentation currently labels itself3.12.16, so exact3.12.14
implementation claims above rely on the pinned source, not that live version.

The v1 source-only round-trip fixture did not cover this difference between a
loaded function's code graph and a freshly compiled code graph. That admission
test was too narrow. A raw marshal hash was consequently an unsuitable strict
semantic code-value fingerprint. This is an execution-admission defect; no
learner result exists from which to select a new scientific setting.

## Separate prospective correction

Root has authorized preparation of an execution-only v2 protocol in new files
and a new directory. The prospective correction must compare a complete typed
canonical record of every documented noncallable `co_*` attribute, recursively
cover code constants, include `co_lines()`/`co_positions()`, require code equality
and reject unsupported constants. Retain filename, qualification, flags,
bytecode, constants, variable/free/cell names, source positions and all tables.
Keep the exact source hash, actual `frame.f_code` identity, trace-scope guard,
environment and runtime binding. Put raw marshal/adaptive diagnostics in a
separate nonsemantic audit record; they must not enter independent-refit
semantic comparisons.

Pure tests must reproduce the captured equal-code/different-marshal case without
executing either code object, and reject changed bytecode, constants, filename,
first line, flags and the remaining code fields. The learner, author mathematics,
two-call observer, convergence rule, training roles, data shapes, all query calls
and every resource bound must remain identical. New source/plan review, freeze,
verification and explicit v2 resource admission precede any scientific fitting.
The frozen v1 source, plan, runtime inventory and failed assessment remain
immutable; v2 is not an amended v1 success.

## Integrity and source reads

The frozen v1 verification passed before and after the diagnostic import.
The probe also compared hashes of its frozen source/plan/freeze records and
every failed assessment file before/after; all match. No existing file was
modified. The diagnostic did not acquire a package, model or executable.

| Primary3.12.14 source read in memory | Bytes | SHA256 |
|---|---:|---|
| `Python/marshal.c`: reference, string, code writer branches |52874|`6d2dc841001316ed7f9255ec03aeeb6e14780421f44ba61436da638c6f49bf0c`|
| `Include/internal/pycore_code.h`: locals-plus kinds |15835|`48e78dd1a12e6afa6e54e26f8e7c4f56e20689b84189d503507c7f6c36819092`|
| `Objects/codeobject.c`: initialization/code extraction |75495|`60365e71ac9fc8411c546f86de3a1f49b6190ebb40b893ed31c6ff0b2cb5493f`|
| `Doc/library/marshal.rst`: format/version description |5490|`b1cef3b53cb92d9f0e70a6d81364e9004fe096c2eebc6e6b518feee5421298ec`|

An independent source-only reader reached the same reference-encoding diagnosis.
No field was waived because it differed, and no model was run to test a fix.
