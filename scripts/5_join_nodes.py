#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Due to computational limitations and the large number of nodes from the high resolution domains the data has to be processed in slices.

Prepares slices of the dataframes and joins with the coastline dataframe in a one (obs per node file) to many (the entire coastline df) join.

Only keep the obs, land_cell and coords tuple columns.

Slice the nodes df in slices of 1000 node coords, retain only columns 0,3 and 4 for the merge.

These columns may vary depending on the structure of the previous files if any changes have been made.

Merge dfs on a key, resulting in a one to many merge (df_cat copies itself for each row of df_r, df_h and df_c).

In this case there were over 61000 nodes for the harbour domain, over 36,000 for the coastal domain.

Delete the loaded node variables after each step to prevent the kernel from crashing and restarting.

Note: the regional domain did not require a for loop.
"""

import pandas as pd

# read the pickles
df_h = pd.read_pickle('../pickles/nodes/coords/harbour_nodes_coords.pkl')
df_c = pd.read_pickle('../pickles/nodes/coords/coastal_nodes_coords.pkl')
df_r = pd.read_pickle('../pickles/nodes/coords/regional_nodes_coords.pkl')
df_cat = pd.read_pickle('../pickles/coords/coastline_polygon.pkl')

# drop lat and lon columns of the coastline polygon, only need the coords tuple for the comparison 
df_cat.drop('longitude', axis=1, inplace=True)
df_cat.drop('latitude', axis=1, inplace=True)



# coastal files
for i in range(36): # this range will depend on the number of nodes
    df_coastal = df_c.iloc[(i*1000):((i*1000)+1000),[0,3,4]]
    df_coastal = df_coastal.assign(key=1).merge(df_cat.assign(key=1), on='key').drop('key',axis=1)
    df_coastal.to_pickle('../pickles/nodes/merged_coords_slices/coastal_coords_combined_' + str(i) + '.pkl')        
    del(df_coastal)

# harbour files 
for j in range(61): # this range will depend on the number of nodes
    df_harbour = df_h.iloc[(j*1000):((j*1000)+1000),[0,3,4]]
    df_harbour = df_harbour.assign(key=1).merge(df_cat.assign(key=1), on='key').drop('key',axis=1)
    df_harbour.to_pickle('../pickles/nodes/merged_coords_slices/harbour_coords_combined_' + str(j) + '.pkl')        
    del(df_harbour)

# if the regional domain has a large number of nodes, do a loop as above
df_regional = df_r.assign(key=1).merge(df_cat.assign(key=1), on='key').drop('key',axis=1)
df_regional.to_pickle('../pickles/nodes/merged_coords_slices/regional_coords_combined.pkl')  