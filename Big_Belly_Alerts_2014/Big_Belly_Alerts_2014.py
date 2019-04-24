# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:13 2019

@author: jmiller
"""

import numpy as np
import pandas as pd

df = pd.read_csv('big-belly-alerts-2014.csv', parse_dates = ['timestamp'])

# TODO this isn't working
df['fullness'] = df['fullness'].apply(lambda x: 0 if x == 'GREEN')

# see if you can predict when cans are actually getting collected
# then you can see if this system is actually working
# try a linear SVC