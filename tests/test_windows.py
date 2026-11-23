
import numpy as np
from fall_detection.data.windows import make_windows

def test_window_shape():
    x = np.ones((10,4),dtype=np.float32); y = np.zeros(10,dtype=np.int64)
    wx,wy = make_windows(x,y,4,2)
    assert wx.shape == (4,4,4) and wy.shape == (4,)
