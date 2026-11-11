import shutil
import subprocess
import unittest
import numpy as np
from wasb_decode import selection_expression


@unittest.skipUnless(shutil.which('ffmpeg'),'ffmpeg required')
class DecodeTests(unittest.TestCase):
    def test_768_selected_indices_do_not_exceed_parser_depth(self):
        source=np.repeat((np.arange(1000)%256).astype(np.uint8)[:,None,None,None],48,axis=1).reshape(1000,4,4,3)
        selected=np.unique(np.rint(np.linspace(0,999,768)).astype(int))
        result=subprocess.run(['ffmpeg','-nostdin','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','4x4','-r','25','-i','pipe:0','-vf','select='+selection_expression(selected),'-vsync','0','-pix_fmt','rgb24','-f','rawvideo','pipe:1'],input=source.tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
        decoded=np.frombuffer(result.stdout,np.uint8).reshape(-1,4,4,3);np.testing.assert_array_equal(decoded,source[selected])


if __name__=='__main__':unittest.main()
