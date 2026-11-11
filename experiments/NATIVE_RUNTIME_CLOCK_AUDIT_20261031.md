# Native runtime clock and sleep audit

Read-only investigation begun 31 October 2026, observed at 1 November
00:09–00:12 UTC (31 October 20:09–20:12 EDT). Only the private interpreter's
standard library, read-only system power/clock queries, timing JSON/file
metadata and frozen source text were used. No scientific import, native array,
model, power preference, assertion, process signal, restart, freeze, hash update
or prior-result change occurred. This note is the only new artifact.

## Findings

**Host sleep is established; ordinary idle sleep is not established.** The
private Python's actual duration clock excludes host suspend. This explains why
several retained worker durations are hundreds of clock seconds while their
calendar file spans cover hours. It does not establish that each reported time
discrepancy came exclusively from one sleep interval or that any model result
changed after wake.

The supplied live-process description was stale by this observation. Read-only
`ps` queries found neither PID58213 nor PID76820. R65's retained
`screen/purged_spells/block5/supervision.json` records `passed=false`,
`exit_code=-15`, `aborted=worker_over_900seconds` and elapsed
`900.2050882920012`. Its modification time is
`2026-10-31T22:12:13.741392Z`. Nine earlier worker supervision records report
success. These are observations of existing timing records, not new execution
or a new interpretation of possession scores. The worker's recorded 900-second
failure remains a failure; sleep evidence supplies no waiver or retry authority.

## Actual clocks on this host

The observed OS version is27.2; private Python reports3.12.14. At
`2026-11-01T00:09:16.004318Z`, `time.get_clock_info` reports:

| Python clock | Actual implementation | Monotonic | Adjustable |
|---|---|---|---|
| `monotonic` | `mach_absolute_time()` | true | false |
| `perf_counter` | `mach_absolute_time()` | true | false |
| `time` | `clock_gettime(CLOCK_REALTIME)` | false | true |
| `process_time` | `clock_gettime(CLOCK_PROCESS_CPUTIME_ID)` | true | false |

At `2026-11-01T00:10:07.473686Z`, a standard-library `ctypes` read of the public
libSystem APIs, with the returned timebase numerator/denominator125/3, gave:

| Counter | Seconds |
|---|---:|
| `mach_absolute_time` |47,674.78316808333|
| `mach_continuous_time` |118,171.46969595832|
| Continuous minus absolute |70,496.68652787499|
| Python `monotonic`, read immediately afterward |47,674.783411333|

The two native counter calls are sequential rather than simultaneous, so their
difference is approximate at the submillisecond level. The difference is about
**19hours35minutes of accumulated suspend offset since boot**, not a measurement
of a single worker's sleep. `kern.boottime` reports30October11:20:41EDT.

Apple's public kernel header describes `mach_continuous_time` as the equivalent
that advances during sleep. This distinguishes it from `mach_absolute_time`.
The observed Python counter matches the latter, so host suspend is omitted from
its deltas. This is platform duration time including awake scheduling/I/O waits,
not process CPU time.
[Apple clock API source](https://raw.githubusercontent.com/apple-oss-distributions/xnu/main/osfmk/mach/mach_time.h).

CPython's exact3.12.14 Apple branch calls `mach_absolute_time` and applies the
timebase conversion; its non-Windows performance-counter implementation uses
the same monotonic-clock routine. The runtime observation agrees with that
primary implementation.
[CPython3.12.14 source](https://raw.githubusercontent.com/python/cpython/v3.12.14/Python/pytime.c).

Python documents monotonic time as unaffected by system clock updates and its
clock-info API as reporting the actual implementation. The generic
performance-counter description mentioning sleep is insufficient evidence that
this macOS counter includes **host suspend**; a sleeping thread and a suspended
host are different conditions. UTC/realtime remains adjustable and should not
alone be used to infer an exact suspend duration.
[Python time documentation](https://docs.python.org/3.12/library/time.html).

## Sanitized local power evidence

`pmset -g log` completed successfully. Only canonical dated Sleep/Wake/DarkWake
events for30–31October were extracted; process assertion listings and app details
were not printed or retained. Among events explicitly entering sleep, reasons
were:

| Reason | Count |
|---|---:|
| Maintenance Sleep |77|
| Sleep Service Back to Sleep |28|
| Clamshell Sleep |3|
| Software Sleep |1|
| Notification Wake Back to Sleep |1|
| Literal Idle Sleep |0|

The clamshell events are30October15:27:25,31October00:38:34 and01:15:46EDT.
Software sleep is31October01:45:20EDT. Examples of later sleep-to-next-resume
intervals include31October08:51:25→09:08:42 (1,037seconds) and
18:21:23→18:38:53 (1,050seconds), both labelled Maintenance Sleep. The latest
observed resume is20:07:20EDT, a transition from DarkWake to FullWake attributed
to user activity. These observations support real system suspend/resume rather
than merely inferring it from a long gap between messages. They do not establish
ordinary idle sleep or identify which user action caused software sleep.

Existing R65 filesystem timestamps supply a separate calendar proxy. The stdout
file is created immediately before worker launch; supervision is written after
worker exit. Their calendar span includes small setup/cleanup differences and
is not an exact new timer measurement:

| Worker | Recorded monotonic seconds | Calendar file span seconds |
|---|---:|---:|
| Buffered1 |423.636614|423.639545|
| Buffered2 |382.775454|2,822.047116|
| Buffered3 |384.640113|8,341.406287|
| Buffered4 |437.270423|12,234.468339|
| Buffered5 |496.167943|10,724.683364|
| Purged1 |428.487381|11,502.422798|
| Purged2 |337.962211|9,432.545937|
| Purged3 |455.777452|860.909065|
| Purged4 |714.615849|714.618931|
| Purged5, aborted |900.205088|900.208726|

For example, buffered4's stdout birth time is09:03:00.157202Z and its
supervision modification is12:26:54.625541Z. These calendar gaps agree with the
presence of repeated sleep cycles. File timestamps and realtime can be affected
by clock corrections, so no per-worker sleep duration is reconstructed from
their difference here. Purged5's near-equal calendar and monotonic spans also
make it inappropriate to attribute that recorded timeout to excluded suspend.

## Which frozen timing checks use which clock

R65's native supervisor uses `time.monotonic` for its300-second operation
deadline,900-second worker deadline, five-second memory-monitor freshness check
and retained `elapsed_seconds`. Its inherited MPS operation routine uses the
same clock before synchronization and after completion. The memory monitor's
`sampled` marker is also monotonic. The inherited technical supervisor calls one
field `elapsed_wall_seconds`, but computes it with this same counter.
[Native supervisor](tabpfn_native.py),
[inherited operation/monitor](tabpfn_mps_runtime_gate.py),
[isolated wrapper](tabpfn_mps_isolated_runtime_gate_v2.py).

R72's complete four-fit operation, all-query operation,900-second supervisor
and field named `wall_seconds` likewise use `time.monotonic`. Neither frozen
implementation uses UTC or `mach_continuous_time` to enforce a suspend-inclusive
calendar deadline.
[R72 implementation](kernel_svm_runtime_gate.py).

Consequently a memory sample appearing0.028seconds old in this clock does not
prove uninterrupted monitoring across host suspend. The monitor and model can
both pause and resume while that clock omits the gap. This does not, by itself,
prove that memory exceeded a bound or that outputs changed. Existing measured
resource records retain their exact source-defined semantics; nothing here
relabels an old pass, changes a limit, or authorizes a failed run to continue.

For reporting, retain the original clock values and filenames and state their
actual implementation. Separate source-defined monotonic duration from calendar
latency; avoid describing either as CPU time. Do not retrofit UTC deadlines or
subtract estimated sleeps from a recorded failure. Continue requiring the
already planned repeat/state/reload/fresh-replay checks; sleep alone supplies no
numerical-validity conclusion.

For a **future separately frozen** timing protocol, record monotonic,
suspend-inclusive continuous and UTC timestamps together at start/end, plus
clock implementation and boot identity. Specify prospectively whether each
limit concerns active-host duration, continuous end-to-end latency, or both.
A separate observer could collect those three clocks without changing model
code, but no observer or amended protocol was created here.

## Power-management recommendation

**Do not attach `caffeinate -i -w58213` now:** that PID was absent at observation,
and the identified sleep reasons include clamshell/software sleep rather than
ordinary idle sleep. This investigation does not establish that `-i` prevents
the observed sleep mechanism.

For a future verified **idle-sleep** case with a still-live, freshly checked task
PID, `/usr/bin/caffeinate -i -w PID` is a minimal task-scoped option for root to
consider. Apple's own manual says `-i` prevents system idle sleep and `-w`
releases the assertion when the specified process exits. Its implementation
maps that flag specifically to `PreventUserIdleSystemSleep`. It is not evidence
of a lid-close or explicit-software-sleep override.
[Apple manual source](https://raw.githubusercontent.com/apple-oss-distributions/PowerManagement/main/caffeinate/caffeinate.8),
[Apple assertion mapping](https://raw.githubusercontent.com/apple-oss-distributions/PowerManagement/main/caffeinate/caffeinate.c).

No caffeinate process was launched. No display assertion, power preference,
system setting, lid behavior, application or kernel state was changed, and no
R65/R72 process was signalled or restarted.
