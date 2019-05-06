#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:05:44 2019

@author: Jake
"""

# https://data.boston.gov/dataset/food-establishment-inspections
# There is an API to query this if you want to figure it out

import pandas as pd
import numpy as np
from datetime import datetime

df = pd.read_csv('tmp_8g7ojgx.csv')

datetime_cols = ['issdttm', 'expdttm', 'resultdttm', 'violdttm']
str_cols = ['viollevel', 'violstatus', 'violdesc']

print(df.iloc[0])
print()
print(df.isnull().sum())

df.drop(columns = ['dbaname'], inplace = True)

# Example of missing datetime
print('\nMissing datetime: ' + df.iloc[4663]['issdttm'])

# Drop any rows that don't have issue dates - ok to not have expiration date
df.drop(df.loc[df['issdttm'] == ' '].index, inplace = True)

# Force string columns to actually be strings
df[str_cols] = df[str_cols].astype(str)

# Print violation descriptions
print('\nTypes of violations: \n' + str(np.unique(df['violdesc'])))
print('\nViolation levels: \n' + str(np.unique(df['viollevel'])))

#TODO can you infer any legalowners from business name?
#TODO can you infer business name from property id?
#TODO cast columns to string that need to be string
#TODO cast datetime columns to datetime