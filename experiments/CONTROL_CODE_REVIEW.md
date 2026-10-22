# Control refinement code review

The `code-review` skill reviewed Standards and Spec independently against the
preceding untracked source snapshot in
`data/control_refinement/preserved/experiments_before.tar.gz`. No commits or
public data changes were created. New source files were reviewed directly.

## Standards

Zero actionable findings. The reviewer verified train-only targets, bounded
label-free contact/motion support, immutable ownership references, safe pure-D
normalization, shared inference parity, protected prior candidate directories and
the exact retained-model probability check. No substantive Fowler-smell findings
warrant changes. Forty-five targeted tests pass.

## Spec

Zero actionable findings. Model-derived short proposals are algorithmic inference,
not manual frame patches. The retained model is unchanged. All nine paired
primary scores improve, buffered short-run recall rises, false-D frames fall and
D recall/missed-stoppage counts remain unchanged.

The reviewer identified material tradeoffs to disclose, all retained in the report:
IoU-0.5 spurious predictions rise 4→5 while matches rise 11→12; live-run fragments
rise 11→12; purged minute 5 regresses; longer-block false-D counts are unchanged
and MacroF1 falls slightly within the declared tolerance. Full frame recall of
the 19-frame possession does not mean its predicted boundaries are accurate.

Final totals: Standards 0 unresolved; Spec 0. The same public labels have been
repeatedly inspected; full-match and independent accuracy remain unmeasured.
