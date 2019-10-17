# -*- encoding:utf8 -*-
import numpy as np
a = np.array([1,2,3,4,5])
b = np.array(a)*2
c = (a,b)

for i,j in zip(*c):
    print(i,j)

# import tensorflow as tf
from PIL import Image