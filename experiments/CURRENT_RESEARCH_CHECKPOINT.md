# Current research checkpoint

Updated 2026-10-31T00:33:26.609183+00:00. User higher-score goal ACTIVE, UNACHIEVED.
Previous continuation is PROGRESS: actual R61 scores/independent audit finished,
R67 all-four technical gate finished, one-file external runtime drift diagnosed,
project-local Python isolated, and R69 real full200epoch runtime revalidation is
running. No new record, selected-model change or private accuracy is claimed.

## Objective and safeguards

Selected8 data/candidate_motion_dead_trim/candidate.csv remains SHA
3b559d4e89eb687b818b66732b250b917774391763346d038d35413d9132fbea.
Selected buffered/purged/longer spellF1=.741176470588/.685714285714/.670270270270.
Highest comparable buffered research record R37=.7542857142857143 (132/175),
which itself fails selection safeguards. Need STRICTLY higher plus all selected8
primary/meanfold/macro>=ref-.001/shortIoU/priorMatched/priorShortRecall/noextra
spurious or missed/Drecall/falseD<=ref+10 and buffered+.01, all9protocol-seed
pairs42/43/44, exact independent replay, public/full parity, two identical
199319rowCSV builds, validator and independent review before promotion/complete.
Only7500 public labels11185..18684 and17possessions. No private/manual/sealed
contents, hostedcompetitioninference, submission/publishing/commit/upload or
external/app-chat contacts. Hash-only reserved-filebinding is permitted.

All OLD FROZEN .py/source/dependency/artifact files IMMUTABLE. New standalone
protocol wrappers/copies allowed, preserving old failures and source snapshots.
Ledger now1603rows; all1591 preceding rows/605189bytes remain exact. Protected17
selected artifacts and2source symlinks unchanged. Supermemory mount searches
onlysmfs grep ifeverneeded. Preserve preexisting uncommitted/untracked work.

## Critical runtime change and durable repair

App-owned shared Python executable changed outside this experiment:
/Users/derekko/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12
oldSHA d59f5251bce588951dc870bde45f0cd93de39327b1ed2431a8d26bfb567aac10
newSHA ab9c532a75ce81eb96562bda04138e7a0b7528bd242c82a436ebeb1405aa361f.
Version/build still3.12.14 (Aug25/Clang22.1.3). Both oldvenvs pointthere.
Across R64's25130bindings ONLYthisexe differs; R68's8577headbindings ONLYtwo
aliasesofsameexe differ; R67audit27658files/60distributions same finding. Old
strict verifiers CORRECTLYFAIL; do not edit old frozen maps to fake a pass.
Cause of R61partialscreen interruption remains unknown, not proven by this drift.
Old runtime/gate results remain historical evidence, not current execution proof.

ROOT created data/control_isolated_python_v1/python: currentexe+stdlib physical
copy,1181files, no hostsitepackages/pycache. Its head_runtime andtorch25_runtime
are propervenvs using thisisolatedbase, with only respective unchanged project
.venv/SportsTorch site-packages directory linked readonly.17symlinks+pyvenvcfg
and all files bound acquisition.json; no oldglobal/runtime/dependency modified.
All futurefits should use thisisolatedbase, preventing app Python replacements.
HeadCLI=data/control_isolated_python_v1/head_runtime/bin/python.
CPU2.15workerCLI=data/control_isolated_python_v1/python/bin/python3.12 -P -S -B
with existing repairedTorch2.15/NumPy2.1.2 target, nevermixsitepackages.
Torch2.5CLI=data/control_isolated_python_v1/torch25_runtime/bin/python.

## R69 isolated ModernNCA technical gate COMPLETE PASS

New experiments/modernnca_isolated_runtime_gate.py, MODERNNCA_ISOLATED_RUNTIME_PLAN.json,
test_modernnca_isolated_runtime.py, data/control_modernnca_isolated_runtime_v1.
SourceSHA 1db961c0b5051a49f47d2c26466368afcc4e1068073aeac81267f5d0ca34f1a7
PlanSHA a26c40faba319fec47e06d75af9cbc184c80450dc3259f70887c1570a4691ed9
An exact execution-only copy of frozen R64:13numeric/training/query/worker/
supervisorfunctions AST-identical; only ROUND/PLAN/interpreter/test identity and
isolated-runtime file binding differ. Author source/defaults/data/seeds/resources
unchanged.22puretests+sourcepreflightpass; ROOT reviewed complete execution diff,
clearly no separateagentreviewclaim. Freeze92639/verify88191 terminal0.
Actual fullfourstagegate session78554 TERMINAL exit0. ALL4PASS.
Bothmoderate andbothfull independent200epochrefits finish, allweights/losses/
probabilities byte-identical withinR69 AND to oldR64all4stages. Stdlibaudit
cross_runtime_audit.json verifies all cached artifacthashes and exact crossold
state/loss/prob files. R69summarySHA
4bc2ec66b61a754d960ae6d6d11eb1697bb53a8943b65484baee6a0ea058381a.
No R69process remains. The newisolatedruntime admission is real, notoldpass
relabel. Same200epochs/CPU2/f64/resources; no scientificparameter changed.
Old R64all4PASS remains intact (fullfits262.655/261.422s/peak4.019GiB), but its
currentverifier fails sharedPython; R69isNEWnotoldpassrelabel.

## NOW R68 native ModernNCA FROZEN AND REAL SCREEN RUNNING

R68wasunfrozenbeforehostdrift. Oldapproveddraft+failedpreflight/rootfreeze
attempt archived before_isolated_runtime. ROOTexecution-only amendment changes
onlyGate toR69, isolatedheadprefix andruntimeprovenance; all13numeric/guard/
context/workerfunctions AST-exact. Oldscientificindependentreview retained;
ROOTreviewexplicitly makes no separateagentreviewclaim for newexecutionalias.
Root24puretests+currentactualR69all4PASS/sourcepreflightPASS. Newrootreview
binds allsources+newplan+amendment. Freeze83565/verify4502 terminal0.
Newfrozen sources (NEVERedit now):
modernnca_native.py SHA 8e776948f1d04fb1714e1cc2aa8a06210b77a7ea82cd5b3276d3f5ec4f3d891d
research_modernnca_native.py SHA fa037fe016670ed902abbd1ee0f200f5856a9bf14597b0bca90217be7082c8d6
test_modernnca_native.py SHA af0ffea357b6c14055389e92389ecc68f6d9feb0529a2488de856e01f4141aed
MODERNNCA_NATIVE_PLAN.json SHA 234360c9542fcdf734014fc4d629fece3c38e6d1a7aa2d61938a70ae17993053
Nativeactualscreen RUNNING session17858, log
 data/control_modernnca_native_v1/screen_seed42.log.
Launched isolatedheadCLI experiments/research_modernnca_native.py --seed42.
Latestactualpoll confirmslive; firstbufferedminutesblock1 context1706/query1500
started. No possession score yet. OneCPU2workeratime/RSS6GiB. Afterterminal,
inspectmethod_failure/global_failure orscreenmanifest/selection. OnlymainPASS
allows43/44; all9+fresh39replay/fullCSV safeguards remain. Do notrestartonsession
observationtimeout; pollactualsamehandle/process. No other6GiBworkerconcurrent.

Scientific interface unchanged: original102+R37fixed64 features,13outerfolds,
whole79owntrain footprint +250purge,5Hz owntrainH/A paired<=3000augmentedrows,
train-only orbitmean/scale, sameauthor200epochfinalCPUlearner, allvalqueries,
canonical512padlast/original+swapbackaverage,H0/A1, originalslot0Dmass andfinallive
D/Xoverlay exact. Only mainPASS permits43/44 withseed/+1/+2 RNGstreams andall9.
Independent review MODERNNCA_NATIVE_INDEPENDENT_REVIEW.md clearedscientificcore.
Closure excludesunrelatedmutableR65log. Final replayproof hardened: explicit
mode/seed/count +39actualworkerparents/summary/supervision/artifacts/hash +actual
context/state/loss/config/probbytecomparisons.24puretests passed oldexecution
identity; rerun afternew R69summary bound. No nativefit until currentgatePASS.
No fullCSV beforeall9+39freshrefits+laterfullconstructionreview.

## R61 rejected; fresh replay incomplete and currently runtime-blocked

Both14encoder/source models passed84externalcells. Main26heads/12officialrows
finish: native=.635294117647/.606060606061/.611764705882; sourceonly=
.711111111111/.697142857143/.605405405405. Bothfail, noeligible/43/44/fullbuild.
12rows appended1591->1603, sourcecsvSHA
b5787fc876fbff0d69fe07ff13df330dba07df39af41b8ed6f0b4229a5cf0ac7.
Independent official/shared/fold/segment/gates/coverage/DX/protected/ledger audits
pass; report NATIVE_ENCODER_CONTINUATION_REPORT.md.
Mandated replay42781 isMISSING; ps confirms no process. All14freshmodel.pt,
14loss,14probe and28externalprob files areBYTEIDENTICAL to originals; external
selection/84cells exact. screen_replay isPARTIAL(68filesexact), finalmanifest/
selection absent, lastmtime15:15UTC/11:15TorontoOct30. No error trace, do not
invent a terminalexit/cause. Do not restart whole originalreplay or overwritepartial.
New sourceonly recover_native_encoder_screen_replay.py and
R61_SCREEN_REPLAY_RECOVERY_PLAN.json ready: originalscreen AST changesONLYnewout
suffix, retainsallpartialfiles/oldmodels/childsourcepath;15puretestsPASS. Not
frozen/run, strictoldruntime checkBLOCKS. Needs separateexplicitisolated-runtime
recoveryamendment/review before execution; no oldverifier bypass. Futureoutput
screen_replay_recovered_v1 absent. See R61_SCREEN_REPLAY_RECOVERY_REPORT.md.

## TabPFN MPS historical technical work

R57loaderfailure/R59loader-onlyrepair/R60observationaudit/R62trace/R63metadata
init/R66allocatorbudget failures all retained; no oldcode/deps edited.
R67 source-supported sync+empty_cache inside originaltimedprediction callback
ALL4STAGES actuallyPASS in oldruntime, session43229nowMISSING/processabsent.
Source bf69d5c22fae4694a571d95d6668abff80f6c186d6eaa0b2af6ef96cba3ae073;
plan1a325754ac59b4cb25b4520e9de94195d177784eb3c7da89f14552c8c442b8d6.
Exact14moderate+12fullnpy pairs/fullrawpreproc/checkpoint/recipe/initJSON,20flushes
livetensor28978432 unchanged, maxop113.065s, peak4.263GiB verified. OfficialAPI
also clearsframeworkMPSGraph compilationcache; notTabPFNfittedcontext; no OOM
rescue/retry. Sourceproof/deps exact but currentstrictverifierfailsoldPython.
NeedNEWisolated-runtime MPSadmission ifusing thismethod, notrelabelR67currentpass.
R65 TabPFNnative sourceUNFROZEN; replayproof20tests patched butpreparerhitusage
limitbeforefinalreport. No nativeTabPFNarrays/models/scores. Do not freezeuntil
newisolatedtechnicalgateall4PASS pluscompleteROOTreview. Keep6GiB/300op/900worker,
8views/MPSf32/512 andallscientificguards; neverdisableallocatorlimit.

Agent quota errors currently report retryNov4; ROOT cancontinue withtools and
newexecution-only source reviews. This is not a goalblock: isolatedruntimegates
andnativelearnerworkremain actionable. Preservegoalwhole; no complete/paused/
blocked status without proper audit/userinstruction.

LatestcontinuedturnclassificationPROGRESS: isolatedruntimecopy+R69actual4PASS andcrossoldbyteidentity, R68actualnativefit nowstarted. GoalstillACTIVE/unachieved, selected8/ledger1603unchanged.


## Latest R71 completion and prepared fallback

R61isolatedcross-runtime screenrecovery nowCOMPLETE, session95684exit0.
Source recover_native_encoder_screen_replay_isolated.py SHA
 e89f40b1f21812bc19427484bbfe0004a9ba96e268fbebc1a924bcdd160de954;
planR61_SCREEN_REPLAY_ISOLATED_PLAN.json SHA
 28c4ac87b935cc682b21d5c397b73726e1685ed0954ad4ffc6764d59d965a342.
ROOT reviewednewexplicit singleexecutable-binding migration, unchanged original
numeric bodies and17puretests.14freshfeatures/26loadedheads/0fits,121artifacts
byte-identical tomain, alloldfiles/partial68preserved; resource390.023s/.639GiB.
Resultscreen_replay_isolated_result_v1.json passed/recovery_completedTrue.
Oldshared-runtimeverifier stillfailsunchanged; no claimitpassed. Newprotocol
validatesallotheroldbindings plusisolatedPython+stdlib+roles and exactcrossruntime
outputs. R61reproductionclosed; mainqualityREJECTED. See new
 NATIVE_ENCODER_REPLAY_ISOLATED_REPORT.md. No newscore/confirmation/fullbuild.

R70 fallbackMPS source/preflight preparedONLY (NOTfrozen/NOmodelrun):
 tabpfn_mps_isolated_runtime_gate.py/TABPFN_MPS_ISOLATED_RUNTIME_PLAN.json,
 data/control_tabpfn_mps_isolated_runtime_v1.27764historicalbindings preserveall
exceptONEexplicitobsoleteexe mapping, replacedwithisolatednewSHA; allstdlib/
configs/links verified. ExactoldR67core, cacheflush, math,8views/f32/512/caps,
all4stages required.15oldpurecallbacksourcetests+sourcepreflightPASS21233exit0.
Needrootreview/freeze/verify BEFOREanymodel. DoNOTrun6GiBMPS whileR68CPU6GiB
workeractive; resourcebudget9.5GiB. IfR68fails, thisisnextstrongmethodadmission,
thenfinishR65isolatedruntimeprerequisitebinding/review/freeze andnativefits.
No oldfrozen code/deps edits. CurrentR68screen17858stilllive,11of13workersdone
atlastactualobservation; no selection/scoringterminalyet.
