# Next possession-score step: advisory review

Prepared 2026-10-23 after rejected Round 26. This reviews source, plans, aggregate
reports and an external dataset's published description. No competition data,
caches, media, target rows or arrays were opened; no experiment, download,
processing, fitting, scoring or candidate change was performed. The research
skill was applied by the already-dispatched background reviewer.

## Recommendation

**Prioritize a feasibility audit for new, task-compatible possession supervision
from other matches before choosing another model.** The strongest unresolved
bottleneck is reliable ownership evidence for short possessions, combined with
only 17 repeatedly inspected public possessions for learning and selection.
That makes additional independent evidence more valuable prospectively than
another alignment, threshold or decoder variant. This is a recommendation about
the next decision, not proof of an optimal method or promised score gain.
[SHORT_PRECISION_REPORT.md:62-69,157-160](SHORT_PRECISION_REPORT.md#L62-L69),
[PAIR_MOTION_REPORT.md:156-163](PAIR_MOTION_REPORT.md#L156-L163).

A concrete candidate merits that audit: Bassek et al.'s published DFL dataset
documents seven matches, 25 Hz tracking and CC-BY 4.0 availability. Its position
records explicitly contain `BallPossession` (home=1, away=2) and `BallStatus`
(inactive=0, active=1), defined in terms of player control and referee
interruptions. These are promising direct target fields, unlike event-derived
proxies. Completeness, target quality, exact semantics and transfer value remain
unverified here. See the [primary paper](https://www.nature.com/articles/s41597-025-04505-y)
and [dataset DOI](https://doi.org/10.6084/m9.figshare.28196177).
The paper's documentation was verified; current file availability and file
contents were not. Checking the repository listing and retrieval is the first
concrete feasibility check.

## Why evidence comes first

The remaining errors have different causes. Earlier diagnostics found weak raw
ownership evidence on the 71-frame miss; the later spatial heads still choose
the correct team on zero frames of it. The graph heads recover local team signal
on 23/33 frames of another short possession, but conditional decoding gives a
104-frame run with IoU .202. The 19-frame example already has full frame recall
and an overlong ending. These support a mixture of signal and extent problems;
they do not establish that conservative gates alone explain failure.
[SHORT_PRECISION_REPORT.md:62-69](SHORT_PRECISION_REPORT.md#L62-L69),
[SPATIAL_SHAPE_REPORT.md:64-68](SPATIAL_SHAPE_REPORT.md#L64-L68),
[OWNERSHIP_GRAPH_REPORT.md:109-114,163-167](OWNERSHIP_GRAPH_REPORT.md#L163-L167),
[BOUNDARY_REFINEMENT_REPORT.md:33-37](BOUNDARY_REFINEMENT_REPORT.md#L33-L37).

Generic endpoint adjustment is already tested. Nested calibration found just 13
matched case rows, with repeated underlying possessions, and learned corrections
did not support a reliable ending trim. It cannot create an absent possession.
More frames from the same few possessions do not create independent transition
examples. The official metric also averages possession matching over five IoU
thresholds; the single four-short-match diagnostic is not the entire score.
[ENDPOINT_CALIBRATION_REPORT.md:49-63,83-110,146-150](ENDPOINT_CALIBRATION_REPORT.md#L49-L63),
[EVALUATION.md:46-55,125-141](../EVALUATION.md#L46-L55).

| Possible next step | Decision value and priority |
|---|---|
| Error decomposition: signal versus gates | Useful bounded diagnostic before designing any next intervention. Cover every missed and spurious possession and trace raw probabilities through smoothing, decoding and rescue stages. Existing diagnostics already identify mixed causes; repeating a broad public-score search would add little independent evidence. |
| Fresh supervision and validation across matches | Highest prospective value. Verify target semantics, coverage and provenance, then reserve complete matches for development and untouched evaluation. It addresses both sparse training transitions and repeated same-match selection. Availability of usable samples and improved competition performance are not yet established. |
| New temporal model or decoder | Potentially useful after the evidence audit, especially for local signal with poor extent. A small temporal learner is a distinct hypothesis; choosing a TCN, Transformer or semi-Markov decoder now has no demonstrated advantage. Smoothing, persistence, Markov transitions, ordered taps, boundary methods and a small neural head already exist in the research history. |

Sources for the comparison: [OWNERSHIP_GRAPH_REPORT.md:116-167](OWNERSHIP_GRAPH_REPORT.md#L116-L167),
[EXPANDED_RESEARCH.md:25-37](EXPANDED_RESEARCH.md#L25-L37),
[evaluate_markov_decoder.py:1-6,22-46](evaluate_markov_decoder.py#L1-L6),
[NEURAL_CONTROL_PLAN.json:2-15](NEURAL_CONTROL_PLAN.json#L2-L15).

## What the proposed feasibility step must settle

The DFL lead resembles the challenge's target construction: the challenge sets
dead frames to D and otherwise uses corrected tracking's named owner. However,
challenge ownership persists through loose or contested balls, and its input
tracking is automatic and noisy. Establish compatibility around these cases,
restarts and transition timing; do not assume that similar field names certify
identical targets or that cleaner external tracking transfers without a gap.
[CHALLENGE.md:42-45,55-80,97-118](../CHALLENGE.md#L97-L118).

A later, separately authorized audit should freeze target conversion, missing
target rules, feature whitelists, whole-match splits and a simple matched baseline
before training. `BallPossession` and `BallStatus` are targets, never input
features. Use a shared representation without player identities; any missingness
simulation must be applied independently of possession labels so it cannot teach
another label-derived shortcut. Keep the competition's private/corrected full-match ground truth and
reserved evaluation material excluded. External data must remain inspectable
and appropriately attributed under its terms.
[RULES.md:3-21,50-59](../RULES.md#L3-L21).

This differs from the earlier external-data attempt: its Metrica event-derived
weak dead-ball labels reproduced a missing-ball shortcut and transferred poorly.
The proposed audit asks whether explicit frame-level owner/status fields supply
better supervision; it must stop if that premise fails.
[EXPANDED_RESEARCH.md:28,39-44](EXPANDED_RESEARCH.md#L28-L44).

Do not revive the rejected small VLM, sports-action features or alignment work
under a new name. Valid-output VLM protocols were degenerate; tactical and
broadcast action heads failed possession gates; Round 26 added no mapped rows and
its primary lost 543 historical query frames through inverse100. None establishes
that vision or temporal models can never work, but each raises the evidence bar
for repeating that avenue.
[LOCAL_OWNERSHIP_REPORT.md:15-30,61-63](LOCAL_OWNERSHIP_REPORT.md#L15-L30),
[SPORTS_ACTION_OBSERVATION_REPORT.md:25-38](SPORTS_ACTION_OBSERVATION_REPORT.md#L25-L38),
[BROADCAST_ACTION_REPORT.md:87-112](BROADCAST_ACTION_REPORT.md#L87-L112),
[TEMPORAL_GLOBAL_REPORT.md:73-92,118-138](TEMPORAL_GLOBAL_REPORT.md#L73-L92).

Keep selected8 as the comparator. New data could support one genuinely new
training experiment after feasibility and independent match splits are defined;
it does not justify automatically building a larger ensemble or relaxing existing
promotion requirements. This review starts no next round.
[PAIR_MOTION_REPORT.md:147-163](PAIR_MOTION_REPORT.md#L147-L163),
[WORK_HISTORY.md:340-355](WORK_HISTORY.md#L340-L355).
