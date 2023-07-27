#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Read the pickle slices

Reduce the number of rows to one per node by groupìng by minimum distance.

Note: the regional domain did not require a for loop.
"""

import pandas as pd


# coastal
for i in range(36): # this range will depend on the number of nodes    
    df = pd.read_pickle('../pickles/nodes/distance/coastal_distance_' + str(i) + '.pkl')
    df_min = df.groupby(['obs'], as_index=False)['distance'].min()
    df_min.to_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_' + str(i) + '.pkl')
    del(df)
    del(df_min)
    
# harbour
for j in range(61): # this range will depend on the number of nodes    
    df = pd.read_pickle('../pickles/nodes/distance/harbour_distance_' + str(j) + '.pkl')
    df_min = df.groupby(['obs'], as_index=False)['distance'].min()
    df_min.to_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_' + str(j) + '.pkl')
    del(df)
    del(df_min)
    
# if the regional domain has a large number of nodes, do a loop as above
df = pd.read_pickle('../pickles/nodes/distance/regional_distance.pkl')
df_min = df.groupby(['obs'], as_index=False)['distance'].min()
df_min.to_pickle('../pickles/nodes/distance_min/regional_distance_min.pkl')


