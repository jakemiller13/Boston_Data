# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 15:28:13 2019

@author: jmiller
"""

import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

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
locs, loc_counts = np.unique(df['Location'], return_counts = True)

str_coords = df['Location'].apply(lambda x: x.split())

# we have counts and description
# TODO need to correlate description with counts AND coordinates

coords = []
lat = []
lon = []

for row in locs:
    row_split = row.split()    
    lat.append(float(row_split[0].strip(string.punctuation)))
    lon.append(-float(row_split[1].strip(string.punctuation)))

'''
coords = []
x = []
y = []
for row in str_coords:
    x.append(float(row[1].strip(string.punctuation)))
    y.append(float(row[0].strip(string.punctuation)))
    
    j = float(row[0].strip(string.punctuation))
    k = float(row[1].strip(string.punctuation))
    
    coords.append([j, k])

plt.scatter(x, y)
plt.title('Locations of Big Belly Compactors')
plt.show()
'''

# see if you can predict when cans are actually getting collected
# then you can see if this system is actually working
# try a linear SVC

# TODO change location to numbers, plot on axes
# TODO use collection numbers to indicate colors (size of bubbles?)
# TODO see if there's a trend in when/where cans are emptied

lon_diff = 0.2 * abs(max(lon) - min(lon)) # add 20% of difference
lat_diff = 0.2 * abs(max(lat) - min(lat))

m = Basemap(projection = 'merc',
            llcrnrlon = min(lon) - lon_diff,
            llcrnrlat = min(lat) - lat_diff,
            urcrnrlon = max(lon) + lon_diff,
            urcrnrlat = max(lat) + lat_diff,
            area_thresh = 0.1,
            resolution = 'h')
m.drawcoastlines()
m.drawcountries()
m.fillcontinents(color = 'coral')
m.drawmapboundary()

x, y = m(lon, lat)
m.plot(x, y, 'bo', markersize = 10, alpha = 0.1)

plt.show()

'''
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', lat_0=42.3, lon_0=-71,
            width=22500, height=30000, resolution='l')
#m.fillcontinents(color="#FFDDCC", lake_color='#DDEEFF')
#m.drawmapboundary(fill_color="#DDEEFF")
#m.drawcoastlines()
#plt.title('Map of Boston')
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')
x, y = m(lon, lat)
#m.scatter(lon, lat, latlon = True,
#          c = loc_counts,
#          cmap = 'Reds')
m.plot(x, y, 'bo', markersize = 100)
plt.show()

#########
fig = plt.figure(figsize=(8, 8))
m = Basemap(projection='lcc', resolution='l', 
            lat_0=42.3, lon_0=-71,
            width=1E6, height=1.2E6)
m.shadedrelief()
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# 2. scatter city data, with color reflecting population
# and size reflecting area
m.scatter(lon, lat, latlon=True,
          s= loc_counts,
          cmap='Reds', alpha=0.5)
'''