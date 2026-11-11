import unittest
import numpy as np
from dead_bridge import bridge_dead


class DeadBridgeTests(unittest.TestCase):
    def example(self):
        labels=np.array(list('H'*15+'D'*10+'H'*15))
        fast=np.tile([.9,.1],(40,1));seen=np.ones(40);contact=np.zeros((40,2),bool);contact[16:20,0]=True
        restart=np.zeros(40,bool)
        return labels,fast,seen,contact,restart,[(0,40)]

    def test_supported_short_gap_is_bridged(self):
        result,trace=bridge_dead(*self.example())
        self.assertTrue((result=='H').all());self.assertEqual(sum(t['accepted'] for t in trace),1)

    def test_restart_ambiguity_missingness_and_confidence_protect_gaps(self):
        for kind in ('restart','opponent','missing','confidence','flanks'):
            values=list(self.example());original=values[0].copy()
            if kind=='restart':values[4][18]=True
            elif kind=='opponent':values[3][18,1]=True
            elif kind=='missing':values[2][15:25]=0
            elif kind=='confidence':values[1][15:25]=[.5,.5]
            else:values[0][25:40]='A';original=values[0].copy()
            result,trace=bridge_dead(*values)
            np.testing.assert_array_equal(result,original);self.assertFalse(any(t['accepted'] for t in trace))

    def test_half_or_validation_boundaries_are_never_crossed(self):
        values=list(self.example());values[-1]=[(0,20),(20,40)]
        result,_=bridge_dead(*values);np.testing.assert_array_equal(result,values[0])

    def test_team_symmetry(self):
        values=self.example();a,_=bridge_dead(*values)
        labels=np.where(values[0]=='H','A',np.where(values[0]=='A','H',values[0]))
        b,_=bridge_dead(labels,values[1][:,::-1],values[2],values[3][:,::-1],values[4],values[5])
        np.testing.assert_array_equal(b,np.where(a=='H','A',np.where(a=='A','H',a)))

    def test_long_gaps_and_existing_unknown_labels_stay_untouched(self):
        labels=np.array(list('H'*15+'D'*64+'H'*15+'X'*3));n=len(labels)
        result,trace=bridge_dead(labels,np.tile([.9,.1],(n,1)),np.ones(n),np.ones((n,2),bool),np.zeros(n,bool),[(0,n)])
        np.testing.assert_array_equal(result,labels);self.assertEqual(trace[0]['reason'],'duration')


if __name__=='__main__':unittest.main()
