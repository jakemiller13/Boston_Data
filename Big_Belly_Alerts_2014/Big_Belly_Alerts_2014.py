# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:13 2019

@author: jmiller
"""

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt

# Load dataframe, parse datetimes
df = pd.read_csv('big-belly-alerts-2014.csv', parse_dates = ['timestamp'])

# Change string values to useful integer values
df['fullness'] = df['fullness'].apply(lambda x: 0 if x == 'GREEN'
                                      else 1 if x == 'YELLOW'
                                      else 2)
df['collection'] = df['collection'].astype(np.int)

# Get unique locations
description, counts = np.unique(df['description'], return_counts = True)
print('Number of unique compactors: {}'.format(len(description)))

str_coords = df['Location'].apply(lambda x: x.split())

# we have counts and description
# TODO need to correlate description with counts AND coordinates

coords = []
x = []
y = []
for row in str_coords:
    x.append(float(row[0].strip(string.punctuation)))
    y.append(float(row[1].strip(string.punctuation)))
    
    j = float(row[0].strip(string.punctuation))
    k = float(row[1].strip(string.punctuation))
    
    coords.append([j, k])

plt.scatter(x, y)
plt.title('Locations of Big Belly Compactors')
plt.show()

to_plot = {}

for i, (desc, num) in enumerate(zip(description, counts)):
#    to_plot[desc] = [counts, df.iloc[i][]]
    # TODO Working on this line
   print(i)
   print(desc)
   print(num)

# see if you can predict when cans are actually getting collected
# then you can see if this system is actually working
# try a linear SVC

# TODO change location to numbers, plot on axes
# TODO use collection numbers to indicate colors (size of bubbles?)
# TODO see if there's a trend in when/where cans are emptied