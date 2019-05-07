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

# API Link
link = 'https://data.boston.gov/datastore/odata3.0/'\
        + '4582bec6-2b4f-4f9e-bc55-cbaa73117f4c'

# TODO this API only returns 500 rows at a time, need to loop through it

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
    link = link + '?$skip=' + str(increment) + '&$top=500'
    return link
    
'''
tree[-1].attrib
Out[106]: 
{'rel': 'next',
 'href': 'https://data.boston.gov/datastore/odata3.0/4582bec6-2b4f-4f9e-bc55-cbaa73117f4c?$skip=500&$top=500'}

temp_tree[-1].attrib
Out[107]: {}
''' 
    
    
    


def get_data(link, df):
    '''
    Returns data from link as ElementTree.Element
    '''
#    response = urlopen(link).read().decode('utf-8')
#    tree = ET.fromstring(response)
#    return tree
    count = 0
    while True:
        link = inc_address(link, count)
        response = urlopen(link).read().decode('utf-8')
        tree = ET.fromstring(response)
        if tree[-1].attrib == {}:
            temp_df = create_dataframe(tree, start = 4, columns = columns)
            df.append(temp_df)
            break
        else:
            temp_df = create_dataframe(tree, start = 4, columns = columns)
            df.append(temp_df)
            count += 1
    return df

def create_dataframe(tree, start, columns):
    '''
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

def split_datetime(dttm_col):
    '''
    Splits datetime column into 2 separate date & time columns
    Returns [date, time]
    '''
    dttm_split = df[dttm_col].apply(lambda x: x.split())

###############
# GATHER DATA #
###############
tree = get_data(link)
df = create_dataframe(tree, start = 4, columns = columns)


##########################
# DESIGNATE COLUMN TYPES #
##########################
datetime_cols = ['issdttm', 'expdttm', 'resultdttm', 'violdttm']
str_cols = ['viollevel', 'violstatus', 'violdesc']

df[str_cols] = df[str_cols].astype(str)

######


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