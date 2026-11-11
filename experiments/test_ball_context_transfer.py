import unittest
import numpy as np
from ball_context_transfer import clean_ball,ball_windows


class TransferTests(unittest.TestCase):
    def test_clean_ball_is_normalized_and_missing_or_outside_points_masked(self):
        ball=np.tile([50.,30.],(150,1));players=np.ones((1,30,4),bool);ball[15]=np.nan;ball[20]=[110.,30.];players[:,5]=False
        x,m=clean_ball(ball,players,[100,60]);self.assertTrue(m[0,0]);np.testing.assert_array_equal(x[0,0],[0,0]);self.assertFalse(m[0,3:6].any());self.assertTrue(np.isfinite(x).all())

    def test_half_padding_cannot_borrow_intro_ball(self):
        ball=np.ones((1,50,2),np.float32);seen=np.ones((1,50),bool);half=np.r_[np.full(10,-1),np.zeros(40)].astype(int)
        a=ball_windows(ball,seen,half,[10]);self.assertFalse(a[1][0,:15].any());self.assertTrue(a[1][0,15:].all())
        ball[:,:10]=1000;b=ball_windows(ball,seen,half,[10])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
        with self.assertRaises(ValueError):ball_windows(ball,seen,half,[9])


if __name__=='__main__':unittest.main()
