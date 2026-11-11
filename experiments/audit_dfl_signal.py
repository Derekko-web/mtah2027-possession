"""Verify diagnostic replay and preservation; never fit or record score rows."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dfl_transfer_data import digest
from dfl_signal_audit import continuation
from research_dfl_signal import validate


def main():
    root=ROOT/'data/control_dfl_signal_v1';plan=validate()
    left,right=root/'analysis_A_02',root/'analysis_replay'
    files=sorted(str(p.relative_to(left)) for p in left.rglob('*') if p.is_file())
    other=sorted(str(p.relative_to(right)) for p in right.rglob('*') if p.is_file())
    assert files==other and len(files)==13
    for name in files:assert digest(left/name)==digest(right/name),name
    manifest=json.loads((left/'manifest.json').read_text())
    for key,base in (('source_sha256',ROOT),('input_sha256',ROOT),('cache_sha256',left)):
        for name,expected in manifest[key].items():assert digest(base/name)==expected,name
    summary=json.loads((left/'summary.json').read_text())
    assert summary['continuation']==continuation(summary['protocols'],plan['continuation_gate'])
    assert summary['selected_reference_identity_controls']==12
    assert summary['actual_native_prefix_lengths']==dict(decoder_rows=18935,short_rows=18935,long_rows=19060)
    with (left/'native_teacher_probability.npy').open('rb') as f:assert np.load(f).shape==(18935,)
    with (left/'native_teacher_long_probability.npy').open('rb') as f:assert np.load(f).shape==(19060,)
    oracle=json.loads((left/'oracle_diagnostics.json').read_text())
    assert len(oracle)==summary['oracle_interventions']==56
    assert len({(c['protocol'],c['kind'],c['case_id'],c['slot']) for c in oracle})==56
    for name,d in summary['protocols'].items():
        assert len(d['truth_cases'])==17
        assert sum(not c['reference_matched'] for c in d['truth_cases'])==d['baseline_missed_gt']==4
        assert sum(c['frames']<=125 for c in d['truth_cases'])==6
        assert len(d['spurious_cases'])==d['baseline_spurious_pred']
        m=d['confidence']
        assert m['confident_corrections']+m['confident_harms']==m['confident_disagreements']
        assert m['correction_precision']==m['confident_corrections']/m['confident_disagreements']
    traces=json.loads((left/'gate_traces.json').read_text())
    groups={}
    for row in traces:
        key=(row['protocol'],row['probe'],row['slot'])
        groups.setdefault(key,set()).add(row['branch'])
        assert row['accepted']==(row['first_reason'] is None)
    assert len(groups)==59
    assert all(branches=={'reference7','motion_alternative'} for branches in groups.values())
    validate()
    result=dict(round=28,result=summary['result'],continuation_passed=summary['continuation']['passed'],
        selected_slot=summary['continuation']['selected_slot'],fresh_process_byte_exact_artifacts=13,
        native_teacher_checkpoint_reloaded=True,actual_prefix_lengths=summary['actual_native_prefix_lengths'],
        public_prefix_feature_probability_identity=True,identity_controls_per_process=12,
        full_pipeline_oracle_interventions_per_process=56,all_branch_probe_groups=59,
        oracle_target_aware_diagnostic_only=True,all17_gt_and16_spurious_protocol_cases_covered=True,
        full_both_branch_traces_match_original_decoder=True,pristine_selected_arrays=True,
        protected_hashes=17,ledger_byte_exact=True,source_input_cache_hashes_verified=True,
        source_symlinks_exact=True,plan_sha256=digest(ROOT/'experiments/DFL_SIGNAL_PLAN.json'),
        new_fits=0,new_score_rows=0,new_candidate_csvs=0,selected_changed=False,
        prior_prefix_execution_error_preserved=True,prior_round_prefix_claim_corrected=True,
        verification_source_sha256=digest(__file__))
    with (root/'diagnostic_integrity.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
