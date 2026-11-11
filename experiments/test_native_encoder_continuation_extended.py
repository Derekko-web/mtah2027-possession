"""Invented-only R61 objectives, leakage boundaries, schedules, symmetry and replay."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_encoder_continuation_extended import (ha_targets, source_schedule, native_schedule,
    permitted_centers, apply_source, apply_native, mixed_bce, train_fixed, metrics,
    retention, retention_diagnostics, retention_eligibility, rank_eligible, PROTOCOLS)


def invented_source():
    rng = np.random.default_rng(58)
    xy = rng.uniform(-1, 1, (4, 320, 4, 2)).astype(np.float32)
    observed = np.ones((4, 320, 4), bool)
    teams = np.broadcast_to([0, 0, 1, 1], (320, 4)).astype(np.int8).copy()
    labels = (np.arange(320) % 2).astype(np.int8)
    half = np.repeat(np.arange(4), 80)
    buckets = {(str(m), k): np.array([i for i in range(m*80+16, m*80+64) if labels[i] == k])
               for m in range(4) for k in (0, 1)}
    return dict(xy=xy, observed=observed, teams=teams, half_id=half, labels=labels), buckets


def invented_native():
    rng = np.random.default_rng(59)
    return dict(xy=rng.uniform(-1, 1, (12, 31, 4, 2)).astype(np.float32),
        observed=np.ones((12, 31, 4), bool),
        teams=np.broadcast_to([0, 0, 1, 1], (12, 31, 4)).astype(np.int8).copy(),
        target=(np.arange(12) % 2).astype(np.float32))


class PureProtocolTests(unittest.TestCase):
    def test_positive_logit_means_home_and_swap_inverts_target(self):
        np.testing.assert_array_equal(ha_targets(['H', 'A', 'H']), [1., 0., 1.])
        with self.assertRaises(ValueError):
            ha_targets(['D'])

    def test_source_schedule_exact_match_class_swap_balance_and_shared_chunk(self):
        source, buckets = invented_source()
        s = source_schedule(buckets, steps=3)
        for i in range(3):
            for begin in (0, 16):
                sl = slice(begin, begin+16)
                self.assertEqual(np.bincount(s['target'][i, sl].astype(int)).tolist(), [8, 8])
                self.assertEqual(set(s['match'][i, sl]), {'0', '1', '2', '3'})
                for game in ('0', '1', '2', '3'):
                    use = s['match'][i, sl] == game
                    self.assertEqual(np.bincount(s['target'][i, sl][use].astype(int)).tolist(), [2, 2])
            full, shared = apply_source(source, s, i, 32), apply_source(source, s, i, 16)
            for a, b in zip(full, shared):
                np.testing.assert_array_equal(a[:16], b)
            np.testing.assert_array_equal(full[0][:8], full[0][8:16])
            np.testing.assert_array_equal(full[2][:8], 1-full[2][8:16])
        again = source_schedule(buckets, steps=3)
        for k in s:
            np.testing.assert_array_equal(s[k], again[k])

    def test_native_schedule_pairs_every_example_and_balances_both_teams(self):
        pool = invented_native()
        s = native_schedule(pool['target'], steps=4)
        for i in range(4):
            self.assertEqual(np.bincount(s['target'][i].astype(int)).tolist(), [8, 8])
            x, observed, teams = apply_native(pool, s, i)
            np.testing.assert_array_equal(x[:8], x[8:])
            np.testing.assert_array_equal(observed[:8], observed[8:])
            np.testing.assert_array_equal(teams[:8], 1-teams[8:])
        with self.assertRaises(ValueError):
            native_schedule(np.zeros(8))

    def test_complete_79_footprint_and_outside_label_independence(self):
        n = 20000
        train = np.arange(11185, 12000)
        validation = np.arange(12250, 12500)
        frames = np.arange(11185, 12005, 5)
        labels = np.full(n, 'X')
        labels[train] = np.where(train % 2, 'H', 'A')
        half = np.zeros((n+4)//5, int)
        support = np.ones(len(frames), bool)
        selected, eligible = permitted_centers(frames, train, validation, labels, half, support, n)
        self.assertTrue(len(selected) > 0)
        self.assertGreaterEqual(selected.min(), train.min()+79)
        self.assertLessEqual(selected.max(), train.max()-79)
        changed = labels.copy()
        changed[~np.isin(np.arange(n), train)] = 'H'
        again, eligible2 = permitted_centers(frames, train, validation, changed, half, support, n)
        np.testing.assert_array_equal(selected, again)
        np.testing.assert_array_equal(eligible, eligible2)
        with self.assertRaises(ValueError):
            permitted_centers(frames, train, train[:3], labels, half, support, n)

    def test_retention_is_disjoint_per_method_and_incomplete_native_fails(self):
        old = metrics(np.array([.1, .9]), np.array([0, 1]))
        bad = metrics(np.array([.9, .1]), np.array([0, 1]))
        self.assertTrue(all(retention(old, old).values()))
        self.assertFalse(all(retention(bad, old).values()))
        report = [dict(model='source', checks={'ok': True}) for _ in range(6)]
        report += [dict(model=f'fold{i//6}', checks={'ok': True}) for i in range(78)]
        methods, eligible = retention_eligibility(report)
        self.assertEqual(eligible, ['native_continuation', 'source_continuation'])
        report[7]['checks']['ok'] = False
        self.assertEqual(retention_eligibility(report)[1], ['source_continuation'])
        report[7]['checks']['ok'] = True
        report[0]['checks']['ok'] = False
        self.assertEqual(retention_eligibility(report)[1], ['native_continuation'])
        self.assertFalse(retention_eligibility(report[:-1])[0]['native_continuation'])

    def test_relative_source_drift_is_reported_but_not_an_eligibility_gate(self):
        baseline = metrics(np.array([.1, .9]), np.array([0, 1]))
        changed = metrics(np.array([.4, .6]), np.array([0, 1]))
        self.assertTrue(all(retention(changed, baseline).values()))
        diagnostic = retention_diagnostics(changed, baseline)
        self.assertFalse(diagnostic['retained_Brier'])
        self.assertFalse(diagnostic['relative_checks_are_eligibility'])

    def test_selection_uses_score_then_minimum_gain_changes_and_lexical(self):
        methods = ['source_continuation_seed42', 'native_continuation_seed42']
        pairs = [dict(model=m, protocol=p, checks={'all': True},
                      spellF1_delta=.02, changed_frames=10) for m in methods for p in PROTOCOLS]
        scores = [dict(model=m, protocol=p, spellF1=.77) for m in methods for p in PROTOCOLS]
        self.assertEqual(rank_eligible(pairs, scores, methods), sorted(methods))
        scores[0]['spellF1'] = .78
        self.assertEqual(rank_eligible(pairs, scores, methods)[0], methods[0])
        scores[0]['spellF1'] = .77
        pairs[1]['spellF1_delta'] = .01
        self.assertEqual(rank_eligible(pairs, scores, methods)[0], methods[1])
        pairs[1]['spellF1_delta'] = .02
        pairs[3]['changed_frames'] = 11
        self.assertEqual(rank_eligible(pairs, scores, methods)[0], methods[0])
        pairs[0]['checks']['all'] = False
        self.assertEqual(rank_eligible(pairs, scores, methods), [methods[1]])

    def test_native_technical_closure_preserves_other_arm_and_hash_errors_propagate(self):
        import research_native_encoder_continuation_extended as runner
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            evidence = root / 'training_inputs/evidence.json'
            evidence.parent.mkdir()
            evidence.write_text('{"invented": true}')
            with patch.object(runner, 'ROUND', root), patch.object(runner, 'check_inputs', return_value={}):
                runner.close_method('native', 'prepare', runner.MethodFailure('invented empty class'))
                self.assertTrue(runner.method_closed('native'))
                self.assertFalse(runner.method_closed('source'))
                evidence.write_text('{"invented": false}')
                with self.assertRaises(ValueError):
                    runner.method_closed('native')
        with tempfile.TemporaryDirectory() as d:
            with patch.object(runner, 'ROUND', Path(d)), patch.object(runner, 'check_inputs', side_effect=ValueError('frozen hash changed')):
                with self.assertRaisesRegex(ValueError, 'frozen hash'):
                    runner.close_method('native', 'fit', runner.MethodFailure('invented numerical error'))
                self.assertFalse((Path(d) / 'native_method_failure.json').exists())


class InventedTorchTests(unittest.TestCase):
    def test_bce_uses_equal_domain_means_with_correct_gradient_orientation(self):
        import torch
        logits = torch.cat([torch.zeros(16), torch.full((16,), 2.)]).requires_grad_()
        y = torch.cat([torch.ones(16), torch.zeros(16)])
        loss = mixed_bce(logits, y)
        expected = .5*np.log(2) + .5*np.log1p(np.exp(2))
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        loss.backward()
        self.assertTrue((logits.grad[:16] < 0).all())
        self.assertTrue((logits.grad[16:] > 0).all())

    def test_shared_source_dropout_is_identical_at_same_B32_N46_shape(self):
        import torch
        from masked_player_network import MaskedPlayer, initialize
        initialize()
        model = MaskedPlayer().train()
        rng = np.random.default_rng(3)
        x = torch.from_numpy(rng.normal(size=(32, 31, 46, 2)).astype(np.float32))
        seen = torch.ones((32, 31, 46), dtype=torch.bool)
        teams = torch.from_numpy(np.broadcast_to(np.arange(46) % 2, (32, 31, 46)).copy())
        other = x.clone()
        other[16:] = -other[16:]
        torch.manual_seed(580000)
        first = model(x, seen, teams)
        torch.manual_seed(580000)
        second = model(other, seen, teams)
        np.testing.assert_array_equal(first[:16].detach().numpy(), second[:16].detach().numpy())

    def test_fixed_continuation_refit_and_strict_reload_are_exact(self):
        import torch
        from masked_player_network import MaskedPlayer, initialize, probability
        source, buckets = invented_source()
        draws = source_schedule(buckets, steps=3)
        pool = invented_native()
        native = native_schedule(pool['target'], steps=3)
        initialize()
        model = MaskedPlayer()
        initial = {k: v.clone() for k, v in model.state_dict().items()}
        losses, probe = train_fixed(model, source, draws, pool, native, steps=3)
        initialize()
        again = MaskedPlayer()
        again.load_state_dict(initial, strict=True)
        other, _ = train_fixed(again, source, draws, pool, native, steps=3)
        np.testing.assert_array_equal(losses, other)
        self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items()))
        for k, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, again.state_dict()[k]))
        values = [torch.from_numpy(np.ascontiguousarray(v)) for v in probe]
        p = probability(model, *values)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'model.pt'
            torch.save(model.state_dict(), path)
            loaded = MaskedPlayer()
            loaded.load_state_dict(torch.load(path, weights_only=True), strict=True)
            np.testing.assert_array_equal(p, probability(loaded, *values))

    def test_all_missing_nan_windows_remain_neutral_after_updates(self):
        import torch
        from masked_player_network import MaskedPlayer, initialize, probability
        initialize()
        model = MaskedPlayer()
        source, buckets = invented_source()
        train_fixed(model, source, source_schedule(buckets, steps=2), steps=2)
        missing = torch.zeros((4, 31, 4), dtype=torch.bool)
        x = torch.full((4, 31, 4, 2), float('nan'))
        groups = torch.from_numpy(np.broadcast_to([0, 0, 1, 1], (4, 31, 4)).copy())
        np.testing.assert_array_equal(probability(model, x, missing, groups), np.full(4, .5))


if __name__ == '__main__':
    unittest.main()
