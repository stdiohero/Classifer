#!/user/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Wang Haoyu'

import re
import numpy as np

a = [1, 0]
b = [0, 2]
a = np.array(a)
b = np.array(b)
c = []
c.append(a)
c.append(b)
d = sum(np.array(c))
print(c)
print(d)
print(d/len(c))
