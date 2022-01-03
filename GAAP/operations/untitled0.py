# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 06:01:24 2021

@author: Kyle
"""

import numpy as np

test_4d = np.full((4,4,3,20), 1)

# This works at reducing dimensionality fairly easily
test = test_4d[:3, :3, :1, :10]

# In this case, the first two accessors are based on filter dimension.
# The third is based on colour channel.
# The last is determined by the filter number, which then corresponds to
# the number of layer outputs

# Need to check tomorrow how much of a difference bias makes on model performance,
# if I can disregard bias that will significantly aid in increasing optimization speeds

# Maybe I should consider bias as an additional population?
# Rapidly optimize the bias upon finding a good individual
