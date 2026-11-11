"""Regression tests for loss of supported short possession islands."""
import unittest
import numpy as np

from tabpfn_short_control import restore_supported_islands


class ShortControlTests(unittest.TestCase):
    def example(self):
        reference=np.array(list("H"*20+"A"*40+"H"*20))
        alternative=np.array(list("H"*80))
        support=np.zeros((80,2),bool);support[30:40,1]=True
        return reference,alternative,support,[(0,80)]

    def test_supported_island_erased_by_full_replacement_is_restored(self):
        ref,alt,support,blocks=self.example()
        actual,events=restore_supported_islands(ref,alt,support,blocks)
        np.testing.assert_array_equal(actual,ref)
        self.assertEqual(len(events),1)

    def test_block_edge_island_uses_the_available_opposite_flank(self):
        ref=np.array(list("A"*40+"H"*40));alt=np.array(list("H"*80));support=np.zeros((80,2),bool);support[10:20,1]=True
        actual,events=restore_supported_islands(ref,alt,support,[(0,80)])
        np.testing.assert_array_equal(actual,ref)
        self.assertEqual(len(events),1)

    def test_no_contact_support_keeps_the_alternative(self):
        ref,alt,support,blocks=self.example();support[:]=False
        actual,events=restore_supported_islands(ref,alt,support,blocks)
        np.testing.assert_array_equal(actual,alt);self.assertEqual(events,[])

    def test_retained_majority_island_is_not_expanded(self):
        ref,alt,support,blocks=self.example();alt[25:55]="A"
        actual,events=restore_supported_islands(ref,alt,support,blocks)
        np.testing.assert_array_equal(actual,alt);self.assertEqual(events,[])

    def test_dead_flank_does_not_create_a_team_turnover(self):
        ref,alt,support,blocks=self.example();ref[:20]="D";alt[:20]="D"
        actual,events=restore_supported_islands(ref,alt,support,blocks)
        np.testing.assert_array_equal(actual,alt);self.assertEqual(events,[])

    def test_long_reference_runs_are_not_restored(self):
        ref=np.array(list("H"*20+"A"*251+"H"*20));alt=np.array(list("H"*291));support=np.ones((291,2),bool)
        actual,events=restore_supported_islands(ref,alt,support,[(0,291)])
        np.testing.assert_array_equal(actual,alt);self.assertEqual(events,[])

    def test_swapping_teams_swaps_the_result(self):
        ref,alt,support,blocks=self.example()
        a,_=restore_supported_islands(ref,alt,support,blocks)
        swap=lambda values:np.where(values=="H","A",np.where(values=="A","H",values))
        b,_=restore_supported_islands(swap(ref),swap(alt),support[:,::-1],blocks)
        np.testing.assert_array_equal(b,swap(a))


if __name__=="__main__":
    unittest.main()
