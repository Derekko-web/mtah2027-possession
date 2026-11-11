"""Half-start masking must retain centers without borrowing ungraded context."""
import unittest
import numpy as np
from masked_player_native_data import assemble_native


class NativePaddingTests(unittest.TestCase):
    def test_half_start_padding_ignores_prior_half_or_intro_positions(self):
        x=np.ones((1,100,2,2),np.float32);mask=np.ones((1,100,2),bool);teams=np.tile([0,1],(100,1));half=np.r_[np.full(20,-1),np.zeros(30),np.ones(50)].astype(int)
        a=assemble_native(x,mask,teams,half,[20])
        self.assertFalse(a[1][:,:15].any());self.assertTrue(a[1][:,15:].all());self.assertTrue((a[0][:,:15]==0).all())
        x[:,:20]=1000
        b=assemble_native(x,mask,teams,half,[20])
        for p,q in zip(a,b):np.testing.assert_array_equal(p,q)

    def test_repeat_control_retains_identical_half_padding(self):
        x=np.zeros((1,80,2,2),np.float32);x[0,:,0,0]=np.arange(80);mask=np.ones((1,80,2),bool);teams=np.tile([0,1],(80,1));half=np.r_[np.zeros(40),np.ones(40)].astype(int)
        a=assemble_native(x,mask,teams,half,[40],repeat=True)
        self.assertFalse(a[1][:,:15].any());self.assertTrue((a[0][:,15:,0,0]==40).all())
        with self.assertRaises(ValueError):assemble_native(x,mask,teams,np.full(80,-1),[40])


if __name__=='__main__':unittest.main()
