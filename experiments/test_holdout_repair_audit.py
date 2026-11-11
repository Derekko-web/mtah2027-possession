"""Independent randomized inference and focused tests for each new mechanism."""
import unittest
import numpy as np
import pandas as pd
import test_holdout_possession_repair as fixtures
from holdout_possession_repair import refine as initial,VARIANTS as INITIAL
from bounded_holdout_repair import refine as bounded,VARIANTS as BOUNDED
from audit_holdout_extras import independent_refine


class HoldoutMechanismTests(unittest.TestCase):
    def test_terminal_cross_source_contact_bout_replaces_only_leading_excess(self):
        data=fixtures.HoldoutRepairTests().fixture('A'*40+'H'*160+'A'*80)
        data[5][['bc_matched','bc_same_ball_seen']]=1.
        data[5][['h__bc_same_distance_1','a__bc_same_distance_1']]=4.
        data[5].loc[50:79,'h__bc_same_distance_1']=1.
        data[5].loc[170:189,'h__bc_same_distance_1']=1.
        data[4].loc[170:189,['h__distance_1','a__distance_1']]=[4.,1.]
        data[2][40:165]=[.2,.8]
        out,_=bounded(*data,stages=('late',))
        self.assertTrue(np.all(out[:165]=='A'));self.assertTrue(np.all(out[165:200]=='H'))

    def test_unsupported_opponent_and_tiny_trace_close_to_dead(self):
        data=fixtures.HoldoutRepairTests().fixture('H'*50+'A'*70+'H'+'D'*100)
        data[5].loc[20:39,['bc_matched','bc_same_ball_seen','h__bc_same_distance_1','a__bc_same_distance_1']]=[1,1,1,4]
        out,_=bounded(*data,stages=('release',))
        self.assertTrue(np.all(out[:70]=='H'));self.assertTrue(np.all(out[70:]=='D'))

    def test_unseen_allowance_can_decline_all_new_live_continuation(self):
        data=fixtures.HoldoutRepairTests().fixture('A'*70+'H'*30+'D'*70)
        data[4].loc[60:99,'ball_seen']=0
        out,_=bounded(*data,stages=('unseen',))
        self.assertTrue(np.all(out[:70]=='A'));self.assertTrue(np.all(out[70:]=='D'))

    def test_team_swap_keeps_physical_and_continuity_decisions_symmetric(self):
        data=fixtures.HoldoutRepairTests().fixture('D'*30+'A'*100+'D'*50)
        data[4].loc[100:,'ball_goal_line_distance']=-1
        out,_=bounded(*data)
        swap=lambda x:np.where(x=='H','A',np.where(x=='A','H',x))
        changed=list(data);changed[0]=swap(data[0]);changed[2]=data[2][:,::-1];changed[3]=data[3][:,::-1]
        rename=lambda table:table.rename(columns={c:('a__'+c[3:] if c.startswith('h__') else 'h__'+c[3:] if c.startswith('a__') else c) for c in table})
        changed[4]=rename(data[4]);changed[5]=rename(data[5])
        other,_=bounded(*changed);np.testing.assert_array_equal(other,swap(out))


class IndependentHoldoutTests(unittest.TestCase):
    def test_randomized_sources_probabilities_contexts_and_block_edges(self):
        rng=np.random.default_rng(20261109);changes=0
        for case in range(100):
            n=800;blocks=[(20,350),(400,780)];ref=np.full(n,'X',dtype='<U1')
            hprob=rng.uniform(.05,.95,n);owner=np.column_stack((hprob,1-hprob));fast=owner.copy()
            heads=rng.uniform(0,1,(n,2));visible=np.ones(n);motion=np.full(n,6.)
            td=np.tile([1.,4.],(n,1));bd=np.full((n,2),np.nan);seen=np.zeros(n);line=np.full(n,10.)
            for block,(left,right) in enumerate(blocks):
                k=(case+block)%2;team=('H','A')[k];other=('A','H')[k]
                start=left+70;end=start+int(rng.integers(8,161));pattern=(case+block)%5
                if pattern==0:ref[left:start]='D';ref[start:end]=team;ref[end:right]='D'
                elif pattern==1:ref[left:start]=other;ref[start:end]=team;ref[end:right]='D'
                elif pattern==2:ref[left:start]=team;ref[start:end]='D';ref[end:right]=team
                elif pattern==3:ref[left:start]=other;ref[start:end]=team;ref[end:right]=other
                else:
                    ref[left:start]=team;ref[start:end]=other;ref[end:end+2]=team;ref[end+2:right]='D'
                td[left:right,k]=1.;td[left:right,1-k]=4.
                owner[left:right,k]=.8;owner[left:right,1-k]=.2;fast[left:right]=owner[left:right]
                kind=(case//5+block)%7
                if kind==0:
                    visible[max(left,start-10):end]=0;motion[start:end]=np.nan
                elif kind==1:
                    line[start+int((end-start)//2):end]=-1
                elif kind==2:
                    line[start:end]=-1;motion[start:end]=1.;heads[start:end]=.8
                elif kind==3:
                    seen[start:end]=1;bd[start:end,k]=4.;bd[start:end,1-k]=1.
                    owner[start:min(end,start+15),k]=.2;owner[start:min(end,start+15),1-k]=.8
                    fast[max(start,end-15):end,k]=.2;fast[max(start,end-15):end,1-k]=.8
                elif kind==4:
                    seen[left:right]=1;bd[left:right]=4.
                    bd[start:min(end,start+20),k]=1.;bd[max(start,end-20):end,k]=1.
                    td[max(start,end-20):end,k]=4.;td[max(start,end-20):end,1-k]=1.
                    owner[start:max(start,end-25),k]=.2;owner[start:max(start,end-25),1-k]=.8
                elif kind==5:
                    motion[start:end]=1.;heads[start:end]=.5
                else:
                    visible[start:end]=rng.integers(0,2,end-start);motion[start:end]=rng.choice([1.,6.,np.nan],end-start)
            if case%3==0:owner=owner.astype(np.float32);fast=fast.astype(np.float32)
            raw=pd.DataFrame(dict(ball_seen=visible,ball_displacement_1s=motion,ball_goal_line_distance=line,
                ball_touchline_distance=np.full(n,10.),h__distance_1=td[:,0],a__distance_1=td[:,1]))
            bc=pd.DataFrame(dict(bc_matched=seen,bc_same_ball_seen=seen,h__bc_same_distance_1=bd[:,0],a__bc_same_distance_1=bd[:,1]))
            original=ref.copy()
            for method,variants,is_bounded in ((initial,INITIAL,False),(bounded,BOUNDED,True)):
                for variant,params in variants.items():
                    with self.subTest(case=case,variant=variant):
                        actual,_=method(ref,heads,owner,fast,raw,bc,blocks,**params)
                        expected=independent_refine(ref,heads,owner,fast,raw,bc,blocks,params['stages'],is_bounded)
                        np.testing.assert_array_equal(actual,expected);np.testing.assert_array_equal(ref,original)
                        np.testing.assert_array_equal(actual=='X',ref=='X');changes+=bool(np.any(actual!=ref))
        self.assertGreater(changes,100)


if __name__=='__main__':unittest.main()
