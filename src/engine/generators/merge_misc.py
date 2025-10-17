import logging
import time

import numpy as np
import pandas as pd
from yatools import logging_config

from src.engine.generators.merge_frame import Frame
from src.engine.generators.merge_loopers import FromTo, Intersection, Projection, Value, Triangular

logging_config.init(logging.INFO)

k = FromTo(0, 10)
s = Value(5.5)
q = Projection(k)
j = Triangular(k, s)
Jʹ = Triangular(k, j)
Jʹʹ = Triangular(k, Jʹ)

frame = Frame(k=k, s=s,q=q,j=j, Jʹ=Jʹ, Jʹʹ=Jʹʹ)

nu = np.array([[1,2,3]*100])

def vector(arr):
    return pd.Series([row for row in arr])

frame.set_factors(
    c2=lambda k: (k+1),
    c3=lambda k,q: (k+1)*(q+1),
    c4=lambda q,j: (j+1)*(q+1),
    ar=lambda q,s: vector(nu-q-s),
    c8=lambda k, Jʹ,Jʹʹ: (Jʹ+1)*(Jʹʹ+1)*(k+1),
)

frc=frame.copy()
t0 = time.perf_counter()
logging.warning(frc.debug_evaluate_legacy())
t1 = time.perf_counter()

logging.warning(frame.reduce(k, Jʹ, Jʹʹ, ..., q,s))
t2=time.perf_counter()

print(f"Old time: {t1-t0}, new time: {t2-t1}")

a=1
