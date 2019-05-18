#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:05:44 2019

@author: Jake
"""

# https://data.boston.gov/dataset/food-establishment-inspections

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.error import HTTPError
import matplotlib.pyplot as plt
import seaborn as sns

# API Link
link = 'https://data.boston.gov/datastore/odata3.0/'\
        + '4582bec6-2b4f-4f9e-bc55-cbaa73117f4c'

# Create empty dataframe
columns = ['id',
           'businessname',
           'dbaname',
           'legalowner',
           'namelast',
           'namefirst',
           'licenseno',
           'issdttm',
           'expdttm',
           'licstatus',
           'licensecat',
           'descript',
           'result',
           'resultdttm',
           'violation',
           'viollevel',
           'violdesc',
           'violdttm',
           'violstatus',
           'statusdate',
           'comments',
           'address',
           'city',
           'state',
           'zip',
           'property_id',
           'location']
df = pd.DataFrame(columns = columns)

#####################
# UTILITY FUNCTIONS #
#####################
def inc_address(link, inc):
    '''
    Retrieve 1000 rows at a time to avoid timing out
    '''
    increment = inc * 1000
    link = link + '?$top=1000&$skip=' + str(increment)
    return link

def get_data(df, link):
    '''
    Returns data from link as ElementTree.Element
    Occasionally runs into HTTPError mid-execution
    If that happens, returns df as-is. To continue running, re-execute this
        function with correct increment fed into inc_address to get
        starting link address
    '''
    count = 0
    while True: # uncomment when running full, otherwise = 25
        try:
            if count % 10 == 0:
                print('Count: ' + str(count) + ' | df shape: ' + str(df.shape))
            link = inc_address(link, count)
            response = urlopen(link).read().decode('utf-8')
            tree = ET.fromstring(response)
            if tree[-1].attrib == {}:
                temp_df = create_dataframe(tree, start = 4, columns = columns)
                df = pd.concat([df, temp_df], ignore_index = True)
                break
            else:
                temp_df = create_dataframe(tree, start = 4, columns = columns)
                df = pd.concat([df, temp_df], ignore_index = True)
                count += 1
        except HTTPError:
            return df
    return df

def create_dataframe(tree, start, columns):
    '''
    Used in get_data()
    Returns dataframe created from *tree*
    Must supply *start* which is root level at which info starts, e.g. "4"
    *columns* are column headers for dataframe
    '''
    info_list = []
    while True:
        entry = []
        try:
            for child in tree[start][-1][-1]:
                entry.append(child.text)
            info_list.append(entry)
            start += 1
        except IndexError:
            break
    return pd.DataFrame(info_list, columns = columns)

def split_datetime(df, dttm_col):
    '''
    Splits datetime column into 2 separate date & time columns
    Returns [date, time]
    '''    
    date, time = [], []
    dttm_split = df[dttm_col].apply(lambda x: x.split())
    for i in dttm_split:
        try:
            date.append(i[0])
            time.append(i[1])
        except IndexError:
            date.append(np.nan)
            time.append(np.nan)
    return date, time

def add_dttm_cols(df, dttm_col, date, time):
    '''
    Adds new columns in df for *date* and *time*
    *date* and *time* are from split_datetime()
    *dttm_col* is the string name of column being split
    Returns new column names, used when converting to pd.datetime
    '''
    idx = list(df.columns).index(dttm_col)
    col_name = dttm_col[:-4]
    date_name = col_name + '_date'
    time_name = col_name + '_time'
    df.insert(idx + 1, date_name, date)
    df.insert(idx + 2, time_name, time)
    return date_name, time_name

def date_counts(df, date_col):
    '''
    Returns dates and counts of *date_col*
    Used when counting violations given
    Temporarily drops rows that are NaT to avoid extra counting
    '''
    return np.unique(df.dropna(subset = [date_col])[date_col],
                     return_counts = True)

def find_top_violators(df, year, k_violators):
    '''
    Finds top *k-violators* based on *year*
    *df*: can be either dataframe, but should be "df_fail"
    returns: top_names, top_lic, top_counts
    '''
    rows = [row for row in range(len(df))
            if str(year) in str(df.iloc[row].viol_date)]
    df_year = df.iloc[rows]
    lic_no, lic_counts = np.unique(df_year['licenseno'], return_counts = True)
    top_index = lic_counts.argsort()[-k_violators:][::-1]
    top_counts = lic_counts[top_index]
    top_lic = lic_no[top_index]
    top_names = [df[df.licenseno == i]['businessname'].iloc[0].upper()
                 for i in top_lic]
    return top_names, top_lic, top_counts
    
def plot_top_violators(df, year, k_violators):
    '''
    Plots top violators by accessing "find_top_violators"
    *df*: can be either dataframe, but should be "df_fail"
    *year*: year to look at top violators
    *k_violators*: top k-violators for year
    '''
    top_names, top_lic, top_counts = find_top_violators(df, year, k_violators)
    plt.figure(figsize = (10, 10))
    for i, lic in enumerate(top_lic):
        temp_years, temp_counts = np.unique(pd.DatetimeIndex(
                                  df[df['licenseno'] == lic]
                                  ['viol_date']).year,
                                  return_counts = True)
        plt.plot(temp_years, temp_counts, label = top_names[i])
    plt.legend(loc = 'best')
    plt.xlim(2008, 2018)
    plt.xlabel('Year')
    plt.ylabel('Violations')
    plt.title('Violations Per Year for Top ' + str(year) + '-Violators')
    plt.show()

############################
# GATHER AND CLEAN UP DATA #
############################
df = get_data(df, link)
copy_df = df.copy() # this is here to help troubleshoot

# Split datetime columns into separate date and time columns
datetime_cols = ['issdttm', 'expdttm', 'resultdttm', 'violdttm']
date_cols = []
time_cols = []

for col in datetime_cols:
    date, time = split_datetime(df, col)
    date_name, time_name = add_dttm_cols(df, col, date, time)
    date_cols.append(date_name)
    time_cols.append(time_name)

# Convert strings to dates
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors = 'ignore')

# Ensure string columns are actually strings
str_cols = ['viollevel', 'violstatus', 'violdesc']
df[str_cols] = df[str_cols].astype(str)

############################
# PLOT DATES OF VIOLATIONS #
############################
print('\nResults of inspections: \n' + str(np.unique(df.result)))

# Remove "passing" inspections
df_fail = df[(df.result != 'HE_Pass') &
             (df.result != 'Pass') &
             (df.result != 'PassViol')]

viol_date, viol_counts = date_counts(df_fail, 'viol_date')

# Plot total violations found in dataset
plt.plot(viol_date, viol_counts)
plt.ylim(0, max(viol_counts) + 10)
plt.xlabel('Date Found')
plt.ylabel('Number of Violations Found')
plt.title('Total Violations Found')
plt.grid(b = True, axis = 'y', linestyle = '--')
plt.show()

# Line plot above not very useful, try a histogram
years = np.unique(pd.DatetimeIndex(viol_date).year)
n, bins, patches = plt.hist(pd.DatetimeIndex(viol_date).year,
                            bins = np.arange(min(years), max(years) + 2) - 0.5,
                            edgecolor = 'black',
                            linewidth = 1)
cm = plt.cm.YlOrRd
for i, p in enumerate(patches):
    plt.setp(p, 'facecolor', cm(i/len(years)))
plt.ylabel('Number of Violations Found')
plt.title('Violations Issued By Year')
plt.grid(b = True, axis = 'y', linestyle = '--', color = 'k', alpha = 0.2)
plt.show()

######################################
# CHECK DAYS WITH HIGHEST VIOLATIONS #
######################################
top_10_index = viol_counts.argsort()[-10:][::-1]
top_10_dates = viol_date[top_10_index]
top_10_counts = viol_counts[top_10_index]
print('\nTop 10 days violations were identified [number identified]:')
for i, j in zip(pd.DatetimeIndex(top_10_dates), top_10_counts):
    print(str(i.date()) + ' [' + str(j) + ']')

###################################
# MOST COMMON TYPES OF VIOLATIONS #
###################################
viol_types, viol_type_counts = np.unique(df_fail.violdesc,
                                         return_counts = True)
top_viol_index = viol_type_counts.argsort()[-20:][::-1]
top_viols = viol_types[top_viol_index]
top_viols_counts = viol_type_counts[top_viol_index]

ax = sns.barplot(x = top_viols_counts, y = top_viols)
ax.set_title('Top Violations')
ax.grid(True, axis = 'x')
plt.show()

#########################
# TOP VIOLATORS IN 2018 #
#########################
top_2018_names, top_2018_lic, top_2018_counts = \
                              find_top_violators(df_fail, 2018, 20)
ax = sns.barplot(x = top_2018_counts, y = top_2018_names)
ax.set_title('Violations Found in 2018')
ax.grid(True, axis = 'x')
plt.show()

########################################
# PLOT TOP VIOLATORS FOR VARIOUS YEARS #
########################################
plot_top_violators(df_fail, 2008, 20)
plot_top_violators(df_fail, 2013, 20)
plot_top_violators(df_fail, 2018, 20)
