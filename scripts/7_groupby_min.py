#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Read the pickle slices

Reduce the number of rows to one per node by groupìng by minimum distance and save to a pickle

To conserve memory and prevent the kernel from restarting after a few iterations the df is deleted at each iteration
"""

import pandas as pd
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/distance_min/slices'
os.makedirs(path_pickles, exist_ok=True)


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
        
# regional
for k in range(3): # this range will depend on the number of nodes    
    df = pd.read_pickle('../pickles/nodes/distance/regional_distance_' + str(k) + '.pkl')
    df_min = df.groupby(['obs'], as_index=False)['distance'].min()
    df_min.to_pickle('../pickles/nodes/distance_min/slices/regional_distance_min_' + str(k) + '.pkl')
    del(df)
    del(df_min)
