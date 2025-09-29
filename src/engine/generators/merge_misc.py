import logging

import numpy as np
import pandas as pd

from src.engine.generators.merge_frame import Frame
from src.engine.generators.merge_loopers import FromTo, Intersection

logging.basicConfig(level=logging.INFO)

k = FromTo(0, 2)
q = FromTo(0, k)
r = FromTo(q, k)

frame = Frame(k=k, q=q, r=r)

nu = np.array([[1,2,3]])

def vector(arr):
    return pd.Series([row for row in arr])

frame.evaluate(c2=lambda k: (k+1))
frame.evaluate(c3=lambda k,q: (k+1)*(q+1))
frame.evaluate(ar=lambda q,r: vector(nu-q-r))

# frame.evaluate(mul=lambda qq, kk: qq*kk)
full_frame = frame.construct_full_frame()
print((full_frame['c2'] * full_frame['c3'] * full_frame['ar']).sum())

frame.print_structure()
print('='*40)
print('='*40)
print('='*40)
frame.reduce('k')
frame.print_structure()

full_frame2 = frame.construct_full_frame()
print((full_frame2['c2_*_c3'] * full_frame2['ar']).sum())


a=1
