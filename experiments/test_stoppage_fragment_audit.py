"""Randomized independent comparisons across every retained inference family."""
import unittest
import numpy as np
import pandas as pd
from audit_stoppage_fragments import independent_refine
from spurious_stoppage_fragments import refine as initial,VARIANTS as INITIAL
from stoppage_gap_closure import refine as predicted,VARIANTS as PREDICTED
from observed_gap_stoppage import refine as visible,VARIANTS as VISIBLE
from dual_anchor_stoppage import refine as dual,VARIANTS as DUAL
from supported_gap_stoppage import refine as supported,VARIANTS as SUPPORTED


class IndependentFragmentTests(unittest.TestCase):
    def test_randomized_all_families_teams_sources_gaps_and_blocks(self):
        rng=np.random.default_rng(20261108);changed=0
        configs=(('initial',initial,INITIAL),('predicted',predicted,PREDICTED),
            ('visible',visible,VISIBLE),('dual',dual,DUAL),('supported',supported,SUPPORTED))
        for case in range(100):
            n=900;blocks=[(20,420),(460,860)];ref=np.full(n,'X',dtype='<U1')
            heads=rng.uniform(0,1,(n,2));seen=np.ones(n);motion=np.full(n,6.)
            matched=np.zeros(n);distance=np.full((n,2),np.nan)
            for block,(left,right) in enumerate(blocks):
                team=('H','A')[(case+block)%2];opponent='A' if team=='H' else 'H'
                length=int(rng.integers(1,141));start=left+100;end=start+length
                leading=(case+block)%2==0
                ref[left:start]='D' if leading else opponent
                ref[start:end]=team;ref[end:right]=opponent if leading else 'D'
                kind=(case+block)%7
                if kind in (0,1,2):
                    seen[start:end]=0;motion[start:end]=np.nan
                    if kind==1:seen[start-10:start]=0
                    if kind==2:seen[end:end+40]=0
                elif kind in (3,4):
                    motion[start:end]=1.;heads[start:end]=.8 if kind==3 else .5
                elif kind==5:
                    seen[start:end]=rng.integers(0,2,length);motion[start:end]=rng.choice([1.,6.,np.nan],length)
                elif kind==6:
                    seen[start:end]=0;motion[start:end]=0
                    matched[start+length//2]=1;distance[start+length//2]=[1.,4.]
            raw=pd.DataFrame(dict(ball_seen=seen,ball_displacement_1s=motion))
            bc=pd.DataFrame(dict(bc_matched=matched,bc_same_ball_seen=np.ones(n),
                h__bc_same_distance_1=distance[:,0],a__bc_same_distance_1=distance[:,1]))
            original=ref.copy()
            for family,method,variants in configs:
                for variant,params in variants.items():
                    with self.subTest(case=case,variant=variant):
                        actual,_=method(ref,heads,raw,bc,blocks,**params)
                        expected=independent_refine(ref,heads,raw,bc,blocks,family,variant)
                        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(ref,original)
                        changed+=bool(np.any(actual!=ref))
        self.assertGreater(changed,100)


if __name__=='__main__':unittest.main()
