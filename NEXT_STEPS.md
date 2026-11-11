# Next steps: highest compliant score by Nov 17

*Written Nov 10, 22:30 EST.* Sources:
- six independent audits: compliance, rule gray areas, external datasets, label definitions, score levers and prior research;
- three adversarial reviews of the draft: compliance, score impact and feasibility.

All numbers come from **external** data. No competition data or ground truth was opened to produce this plan.

**Deadline:** Nov 17, 23:59 CET (17:59 EST). Aim to submit Mon Nov 16 by 12:00 EST.

## 1. Where things stand

| Candidate | Rules | External spellF1 (DFL / SkillCorner / combined) | Decision |
|---|---|---|---|
| TabPFN "selected8" (described in the current public repo) | **Not compliant.** The TabPFN context and 6 XGBoost heads are trained on the public 5 minutes, decoder repairs target specific public possessions, and the build reads the GT file. | n/a | Drop |
| Rule-based v1–v4 (Codex, Nov 10) | **Not compliant as run.** Every version was scored on the public window, and v2 was adopted after its public score. Smart Tagging constants have no external validation. | 0.454 / 0.339 / ~0.40 | Drop |
| DFL+SkillCorner fusion | Compliant after packaging and documentation fixes | 0.502 / 0.538 / 0.520 | Backup only |
| **Motion ensemble** (`data/external_motion/ensemble_release/`, SHA `ed91c236…`) | Compliant after fixes. Never scored on the public window. Needs a lineage audit. | 0.574 / 0.609 / **0.592** | **Base for the submission** |

The ensemble beats the fusion on all 8 external holdout matches, under every possession convention tested:
- vs the raw provider labels: +0.072;
- vs "controlled possession" relabeled targets: +0.103.

It rebuilds byte-identically, full inference takes under 1 min, and the release is 3.8 MB.

Two caveats on these numbers:
- The external holdout (2 DFL + 4 SkillCorner matches) has confirmed five winners in a row, so it is effectively burned.
- No untouched labelled test set exists. The only candidate is Metrica sample games 1–2, which need a licence answer.

## 2. Tonight / tomorrow morning: compliance blockers

1. **Confirm the Codex loop is stopped and stop all public-window scoring.**
   - It produced 3 public-scored candidates between 17:15 and 17:23. Nothing new has appeared since 17:23.
   - Never import rule-based outputs into the new repo.
2. **Check team registration now.** The rule requires 3–5 members with an identical team name registered before Nov 10, and that may already have closed.
3. **Teammate repo `Jesse-njx/TalentArena`** (branch `codex/possession-research-snapshot`) contains match data: homography, ATD XML, Smart Tagging, LFS pointers and full-match prediction CSVs.
   - Find out whether Codex cloud or its GitHub connector had access. If it did, the data also went to a hosted AI.
   - **Delete the repo.** This is the only reliable way to remove LFS objects. Check for forks.
   - Revoke Codex/Copilot GitHub app access to it.
4. **Public repo `Derekko-web/mtah2027-possession`** describes the GT-trained TabPFN pipeline.
   - Add a "NOT THE SUBMISSION" banner and archive it or make it private.
   - **Do not** rewrite its history or delete it. It contains no match data, and deleting it would look like destroying evidence.
5. **Keep match data away from hosted AI.**
   - Move `data/tactical` and `data/broadcast` outside every agent-readable folder.
   - A human runs competition builds in a terminal no agent can read.
   - Agents get back only pass/fail, row count and SHA.
   - Seal the Oct 24 hand-annotation labels (`data/control_human_annotations_v1`, `data/local_annotation_v1`) offline, with a hash manifest.
6. **Sync the organisers' upstream docs** (`git merge origin/main` for the docs). The Oct 22–23 commits c073f29/f8f2629 add three rules:
   - no hand labels on the game in any amount;
   - the public 5 minutes are "not training data and not a tuning set";
   - choosing between versions on the public window is "a rule, not just advice".
7. **Email the organisers by Wed 12:00, with a default action written down for each question.** All teammates should agree on the disclosure paragraph first. It should be factual, private and dated.

| # | Question | Default if there is no reply |
|---|---|---|
| Q1 | May any parameter be fit on the unlabelled competition tracking (self-training, test-time adaptation, EM)? | Don't do it |
| Q2 | May the synthetic ball-dropout rate for external training data use DATA.md's published "41% missing"? The shipped models simulate 60% visibility. | Keep the 0.6 rate in the shipped models and disclose it. Any new model uses a wide, pre-declared range, e.g. U[0.3, 0.9] per half. |
| Q3 | May hand-set Smart Tagging constants (from the schema and the Laws of the Game, never tuned) be used? | Don't use Smart Tagging |
| Q4 | May feature and decoder *designs* first explored in GT-era research be kept, if re-fit and re-validated on external data only? | Keep them, disclose the lineage, and re-select inherited constants externally (see §4) |
| Q5 | May old research stay in a separate repo available on request? And does CHALLENGE.md's "use the public GT to understand" conflict with "not a tuning set"? | Separate repo; zero public-window checks |
| Q6 | Is the Metrica `sample-data` repo usable (games 1–3: events and tracking)? It has no licence file. | Do not use it. Justify N from DFL and SkillCorner only. |
| — | Disclosure | See the list below |

The disclosure should cover:
- the GT-era research and the date it stopped;
- the Oct 24 annotation pilot (rejected, never used);
- hosted-AI exposure (Codex and Claude Code worked in a folder holding the data; two competition plots were sent on Nov 7; the teammate repo);
- the remediation.

## 3. Score work, in order of expected value

The levers below were measured on external holdouts against relabeled ("controlled possession") targets.

### A. Remove the dead-ball shortcut (largest risk; Wed–Thu)

Every external dataset tells the model the ball is dead in a way the competition tracking won't:
- DFL freezes the ball on 100% of dead frames.
- SkillCorner detects the ball on only 18% of dead frames, versus 91% of live frames.
- Metrica blanks the ball during dead play.

When the dead ball is instead carried by a player, the shipped ensemble loses **0.11–0.17 spellF1** (0.551 → 0.437 raw, 0.678 → 0.526 relabeled). Visibility parity alone doesn't fix it. Missing ball on live frames is also predicted dead 31–35% of the time on SkillCorner.

- **Fix:** augment training dead frames with simulated dead-ball handling: carried by the nearest player, placed and still at the restart spot, off the pitch, missing. Never leave the dead ball frozen. Remove any feature that is a pure "ball frozen" cue.
- **Bounded version** (the full gated chain costs 1.5–2 days): retrain only the long-context XGBoost dead head with this augmentation (~4 min compute), re-blend it, and keep the networks as they are.
- **Also add cross-team identity-switch augmentation** (a track's team label swaps). That costs −0.04 to −0.13 at 0.5–2 concurrent swaps. Same-team swaps cost about nothing, because features use anonymous per-team sets.
- **New required guards** for selection: a carried-dead-ball view, a cross-team-swap view and DFL masked. A hedge with no training: a lower stoppage threshold recovers about 15% of the loss.
- **Gate code:** `temporal_trial.py:177`, `motion_trial.py:170` and `ensemble_trial.py:102` hard-code old reference scores. Write a new frozen protocol rather than patching them around.

### B. Decoder: switch cost toward the controlled-possession convention (~1–2 h; Wed)

The ensemble already uses a 3-state Viterbi decoder (`dfl_submission/joint_decoder.py`).
- Raising its switch cost from 2 to 8 (tuned for the N=3 convention) gives:
  - +0.014 at A-B-A<3 (7 of 8 matches) and +0.028 at <4 (8 of 8);
  - −0.003 at 1.75 and −0.017 against raw labels;
  - worst-case loss across conventions 0.007, versus 0.041 as shipped.
- **Skip the HSMM rewrite.** After this change, turnover-timing headroom is only about +0.016.

Why N=3 is the convention target:
- **External anchors span 1.75–4.5 s.** Metrica raw events give about 1.75 s. Metrica events with the CHALLENGE.md control rule give 4.25–4.5 s. SkillCorner phases give 4.0 s.
- **N=3 is the minimax choice** over that interval.

Before the final test, commit a dated decision record containing:
- the rule (minimax over the externally derived interval);
- each anchor's source;
- the sensitivity table (raw / 1.75 / 3 / 4 / 4.5);
- a statement that team members saw the public window earlier and used no statistic from it.

Never change N afterwards.

### C. Ensemble lineage audit (3 h; Wed morning, gate at 12:00)

1. **Find the GT-era inherited hyperparameters.** `dfl_transfer_teacher.PARAMS` (depth 2, 120 trees, learning rate .04, λ10, min_child 15, subsample .8, colsample .9) is identical to the GT-trained `experiments/build_candidate.py`. The fusion uses it.
2. **Drop the fusion heads if they do nothing.** In the ensemble's final path the fusion heads look zero-weighted (`owner_mix=1`, `dead_mix=1`). Rebuild without them; if the CSV is byte-identical, drop them.
3. **Re-select any surviving inherited constant on external development data**, from a pre-registered grid that includes the old value. Check the same for the island constants (.68/.82, `context_trial.py:27`) and the long-context XGBoost values (`long_context.py:108`).
4. **Build a provenance table:** every constant, with its source, what it was tuned on, and the date.

If the audit fails, fall back to the fusion release (`5ec92ab6…`).

### D. Only if time remains (Thu)

- **SkillCorner 2020 release** (9 matches, MIT, git `1b99d401dd`) as an ownership-only check.
  - It has no dead-ball labels.
  - The parser takes 5–6 h because of the old schema (anonymous tracks, ball as object 55).
- **Better ownership probabilities** remain the largest oracle (+0.23 DFL / +0.21 SkillCorner), but there is no cheap method in hand.

### Cut (measured ~0 value, gray area, or not feasible in time)

- **Ball-missingness model fitted on SkillCorner:** it reintroduces the dead-ball shortcut.
- **BallRadar-style ball imputation:** about +0.02 for 3–4 risky days.
- **HSMM:** see B.
- **X abstention:** zero spellF1 value, because the scorer forward-fills X. It also hurts macroF1, the first tie-break.
- **Decoder voting**, and requiring positive evidence for D.
- **DFL event XMLs:** nothing would consume them.
- **Smart Tagging rules.**
- **PFF WC2022:** its terms §1.5(i) ban machine-learning use.
- **StatsBomb:** non-commercial licence.
- **SoccerNet action models:** trained on NDA-restricted data.
- **Broadcast camera alignment:** failed its gates.
- **The public-window "sanity check":** drop it. Replace it with checks that need no GT:
  - the official validator;
  - H/A agreement with the ATD-only `baseline_smooth50`, to catch a team swap;
  - D onsets against Smart Tagging restarts, as a diagnostic only;
  - half spans from `games.py`.

## 4. Clean submission repo (Tue night to Wed; 1–2 h plus a 20 min human replay)

- **Start point:** merge `data/external_motion/ensemble_release/` with the fusion release. The fusion release is needed so `context_trial` can be audited.
- **Include:**
  - the external-only build;
  - pinned requirements;
  - external-only weights (3–6 MB, no LFS);
  - `train` entry points that refit from external data, plus a download manifest (URL / DOI / commit, sizes, SHA-256);
  - synthetic-data unit tests;
  - the organisers' unmodified validator and scorer, under their MIT notice;
  - attribution files:
    - DFL CC BY 4.0: cite Bassek et al. 2025, *Sci Data* 12:195, and state the changes made;
    - SkillCorner: MIT notice plus credit.
- **Exclude:**
  - `data/`, `evaluation/gt/`, the GT-era `experiments/` folder, the ledger, `TECHNICAL_SUMMARY.md` and `SUBMISSION_BUILD.md`;
  - TabPFN, the annotation tooling and the VLM probes;
  - any frame-level output.
  
  Don't commit the CSV; publish its SHA only.
- **Guards:**
  - a test that fails if build code references the GT path;
  - a test that fails on frame literals outside `games.half_video_blocks`;
  - bundle-absent guards in the `robust.py` and `fusion_trial.py` training entry points.
- **Reproducibility gate:**
  - Byte-identical rebuild from a fresh clone, run by a human.
  - **Also on linux/amd64 (Docker) with `OMP_NUM_THREADS=1`**, because the organisers may re-run it. Today it is verified only on Apple Silicon. Budget 4–6 h if it differs.
  - State that the published weights are the reference: retraining on CPU gives different weights than MPS.
- **Provenance note:** add a note about where the code came from, rather than presenting a single unexplained commit.

## 5. Tech doc (draft Wed–Fri)

Required sections:
- **Approach and why.**
- **Assets used and dropped.**
- **Training-data table:** dataset, source, licence, who labelled it, amount, and role (fit / tune / validate). End with "None from the provided game".
- **Possession rules:** loose balls stay with the last team; minimum-duration conventions with N and its derivation; restarts; D.
- **Constant-provenance table**, including the 0.6 dropout rate and the 105.3×68 pitch size from `games.py`, both disclosed.
- **External results.**
- **Limitations:** domain shift, the dead-ball shortcut, definition uncertainty, the burned holdout.
- **Development history and data handling:** the disclosure.
- **Reproduction.**
- **File naming:** `<team_name>.csv` and `<team_name>.pdf`.

## 6. Schedule (EST)

| When | What | Cut line |
|---|---|---|
| Tue night | §2 items 1–5. Agent builds the clean repo; a human replays the SHA. | Day-1 entry locked |
| Wed | Organiser email by 12:00. Lineage audit (§3C), then pick ensemble vs fusion. Switch-cost retune (§3B) plus the N decision record. Doc v0. | Audit gate 12:00 |
| Thu | Dead-ball augmentation and cross-team swaps (§3A) on the long-context dead head. Stress-guard suite. | Freeze §3A or drop it at 18:00 |
| Fri | Linux/Docker byte-identity. Doc v1. Optional §3D. | — |
| Sat 14 | Adapt to the form instructions. **Model freeze 18:00.** | |
| Sun 15 | Fresh-clone rebuild on another account. Final doc. **Code and doc freeze 21:00.** | |
| Mon 16 | **Submit by 12:00.** | |
| Tue 17 | Buffer only. Hard stop 17:59. | |

**Drop order if behind:** §3D, then §3A (ship B+C only), then the Docker check (state Apple Silicon only).

Supporting analysis scripts and results are temporary, so copy them out if you need them:
- `/private/tmp/claude-502/-Users-derekko-Projects-more-than-a-hack-2027/29e15eb9-6999-4470-9878-c7d2e44ef06c/scratchpad/relabel/`
- `/private/tmp/claude-502/-Users-derekko-Projects-more-than-a-hack-2027/29e15eb9-6999-4470-9878-c7d2e44ef06c/scratchpad/errdecomp/`
- `/private/tmp/claude-502/-Users-derekko-Projects-more-than-a-hack-2027/29e15eb9-6999-4470-9878-c7d2e44ef06c/scratchpad/review/score/`
- `/private/tmp/claude-502/-Users-derekko-Projects-more-than-a-hack-2027/29e15eb9-6999-4470-9878-c7d2e44ef06c/scratchpad/review/feas/`
