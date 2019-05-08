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
import xml.etree.ElementTree as ET
from urllib.request import urlopen
from urllib.error import HTTPError

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
    while count < 25: #True: # uncomment this when running full script
        try:
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
    Returns dataframe created from "tree"
    Must supply "start" which is root level at which info starts, e.g. "4"
    "columns" are column headers for dataframe
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
    Adds new columns in df for "date" and "time"
    "date" and "time" are from split_datetime()
    "dttm_col" is the string name of column being split
    '''
    idx = list(df.columns).index(dttm_col)
    col_name = dttm_col[:-4]
    df.insert(idx + 1, col_name + '_date', date)
    df.insert(idx + 2, col_name + '_time', time)

############################
# GATHER AND CLEAN UP DATA #
############################
df = get_data(df, link)

datetime_cols = ['issdttm', 'expdttm', 'resultdttm', 'violdttm']
str_cols = ['viollevel', 'violstatus', 'violdesc']

for col in datetime_cols:
    date, time = split_datetime(df, col)
    add_dttm_cols(df, col, date, time)

df[str_cols] = df[str_cols].astype(str)

##########################
# DESIGNATE COLUMN TYPES #
##########################

######

'''
print(df.iloc[0])
print()
print(df.isnull().sum())

df.drop(columns = ['dbaname'], inplace = True)

# Example of missing datetime
#print('\nMissing datetime: ' + df.iloc[4663]['issdttm'])

# Drop any rows that don't have issue dates - ok to not have expiration date
df.drop(df.loc[df['issdttm'] == ' '].index, inplace = True)

# Force string columns to actually be strings


# Print violation descriptions
print('\nTypes of violations: \n' + str(np.unique(df['violdesc'])))
print('\nViolation levels: \n' + str(np.unique(df['viollevel'])))

#TODO can you infer any legalowners from business name?
#TODO can you infer business name from property id?
#TODO cast columns to string that need to be string
#TODO cast datetime columns to datetime
'''