# Local temporal team-ownership vision

This round is rejected. The output-budget and image-layout problems were diagnosed
and repaired, but the verified single-image protocol still predicts HOME for every
public sample. Selected8 remains byte-identical. No possession classifier fit,
dense video generation, new candidate, score-ledger row or promotion follows.

## Frozen protocols and result

All protocols reuse27uniform sample centers, with scenes atcenter-12,center,
center+12frames. Three scheduled centers lack usable scenes and are omitted under
the frozen source rules. Six automatically selected jersey torso references form
a HOME/AWAY board. No private/corrected labels or manual annotations are used.

| Protocol | Output / input | Result | Decision |
|---|---|---|---|
| [Original plan](LOCAL_OWNERSHIP_PLAN.json) | ExistingThinking4B,16tokens,4images |27emptyresponses;allstopinthinkingat16tokens | Output-budget failure |
| [Instruct plan](LOCAL_OWNERSHIP_INSTRUCT_PLAN.json) | ExplicitInstruct4B,64tokens,same4images |27HOME;balancedrecall.333333 | Fails;multi-image ordering also unreliable |
| [Montage plan](LOCAL_OWNERSHIP_MONTAGE_PLAN.json) | SameInstruct/options,1verifiedreference/scene montage |27HOME;balancedrecall.333333 | Fails after format/layout repair |

The27public labels comprise14H/9A/4D. Both valid-output protocols have Hrecall1,
Arecall0,Drecall0,accuracy.518519and0abstentions. They fail the unchanged
balanced-recall>=.75,H/Arecall>=.75,Drecall>=.5andabstention<=.2gate. Sample and
class coverage pass. The first protocol's formal score is0because its responses
are empty; that is not evidence of semantic abstention or zero underlying vision
accuracy. There are0literal unclear responses in that original run.

There are81competition-image calls over3dependent protocols on the same27centers,
not81independent evaluation samples. These are small adaptive source-feasibility
checks, not official full-possession scores or independent/private validation.
All original answers, failed plans and unchanged gates remain available.

## Runtime and integration findings

I set the original16-token budget too low for the installed thinking checkpoint.
All27responses are empty, all have nonemptythinking, and allstopforlengthat16.
The local metadata identifies its parent as a4BThinking build. The original model
digest is `a1b29c246319cdcb1f3842f36d29f198f737c0cf516ba1bcba34b1b974e87980`.
The [official thinking documentation](https://docs.ollama.com/capabilities/thinking)
separates thinking from final answers and makes the no-thinking control dependent
on model support. The failed protocol is retained rather than silently repaired.

The separately downloaded [official4BInstruct checkpoint](https://ollama.com/library/qwen3-vl:4b-instruct)
is pinned at `84f7ed8042fa7cdbd902f7dcb7d0370e032095047cc3244179f263b77b5bc318`. Its local manifest and every blob hash
verify. A generic text-only preflight passes before new competition inference;
Instruct runs produce27nonempty,exact-word,normal-stop answers each, with0thinking
and0lengthstops. All generation uses pinnedOllama0.32.3,temperature0,seed42 and
8192context. No public target label enters a prompt or inference producer.

All-HOME output prompted synthetic integration checks without competition data.
Two single-image word cards read correctly, but all4four-image last-image queries
fail. This establishes an unreliable multi-image protocol, not an identified
Ollama software defect or a broad claim about all multimodal models. A single2x2
image grid passes4/4. The exact shared production montage function then passes
4/4central-scene word queries. These14synthetic queries are all retained.

Only after those controls pass is the montage protocol frozen. Its1536x936canvas
places the jersey reference above three aspect-preserving scenes labeledEARLIER,
MIDDLE,LATER. The prompt changes only to describe that layout; the same underlying
images, model, generation options, exact-word parser, sample centers and feasibility
gates remain. Upscaling creates no new source detail. Even with this verified
layout, real ownership outputs remain degenerate. This rejects this small-model
protocol, not vision methods generally.

## Source, privacy and reproduction

Source preparation reads no possession targets. References use only the earlier
weak-color training240frames, requiring matching automatic team labels,
weak-modelconfidence>=.99,distinctframes and torsoheight>=8pixels. They are weak
references, not verified jersey truth. The24reserved windows and60held weak-ball
images, including250-frame exclusions, remain unused. Actual source/reference
eligibility, bounded ball-center lookup, sample schedules and hashes audit.

Original81scene PNGs, the reference board, metadata and manifest independently
rebuild byte-for-byte in a workspace without possession ground truth. Every montage
and its manifest rebuild exactly in another target-free workspace, reusing the
immutable scene PNGs. Original scene widths are175..445pixels(median274),heights
74..221(median117). Composition enlarges them1.15..2.93times(median1.87); it adds
layout clarity, not new scene information.

All3frozen feasibility summaries replay byte-for-byte in a separate workspace
with only the public target file. This repeats parsing/scoring and source audits,
not full neural generation. All170tests pass. Models and all images/responses
remain local; clients use literal127.0.0.1,bypassproxies,forbidredirects/remoteroutes,
and verify model digest/runtime before every request. No hosted competition
extracts, commits, submissions, publication or external contacts occur. After all
inference, our model is unloaded and our server is stopped; downloaded model
assets remain for reproduction.

## Preserved implementation repairs

An auxiliary format diagnostic initially missed narrative output containing one
allowed word. Its original source and failing synthetic check are retained; the
strict diagnostic is repaired. This never changes the generation prompt or
production output parser. The original thinking-truncation diagnostics reproduce
exactly after repair.

The montage runner initially recorded the correct montage input hash under the
old input-folder name. The ordinary audit catches this before target scoring.
The failed producer,manifest andlogs remain in `montage_manifest_identity_failure/`.
Only the manifest footer changes: all code through generation and the source guard
is byte-identical. A shared footer derives the actual relative input path and
excludes its own manifest from cache hashing, with an idempotent-resume regression.
After the completed producer is confirmed terminal, canonical--resumeskipsall27
samples and repairs the footer. All29cached outputs remain byte-identical and
0newneural calls occur. `montage_provenance_repair.json` records the old/new hashes
and exact scope. The repaired ordinary audit and isolated score replay pass.

## Decision and artifact index

Every selected candidate artifact and every byte of the1,402-row ledger is
unchanged:778inherited +624added in this chat. This round adds0possession rows and
0classifier fits. No confirmation seeds or full-match build are warranted. The
maximum/private full-match score remains unverified.

Evidence is local under `data/control_local_ownership_v1/`: before-source/candidate
snapshots;3plans and all81rawanswers; original/montage inputs; separate
`feasibility*.json` and `format_diagnostics*.json`; model-asset verification;
synthetic control plans/scripts/outputs;3ordinary audits; target-free source
replays; `fresh_score_parity.json`; preserved red tests and manifest repair;
`runtime_cleanup.json`; and `round_closeout.json`.

A next independent avenue is a pretrained temporal sports-action representation
that supplies touch/action evidence rather than final team labels from a general
small VLM. It has not been researched, downloaded or evaluated in this round;
no result or frozen possession plan exists for it yet.
