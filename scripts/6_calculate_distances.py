#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan

Read the pickles and calculate the distance from each node to each point on the coastline using a lamda function.

Use great circle calculation, about 20x faster than Vicenty calculation and has an accuracy difference of ~0.14%.

Each iteration can take some time (approx 20 minutes per generated file)

Resulting files can be several Gb in size.

To conserve memory and prevent the kernel from restarting after a few iterations the df is deleted at each iteration
"""

import pandas as pd
from geopy.distance import great_circle
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/distance/'
os.makedirs(path_pickles, exist_ok=True)


# coastal
for i in range(36): # this range will depend on the number of nodes  
    df = pd.read_pickle('../pickles/nodes/merged_coords_slices/coastal_coords_combined_' + str(i) + '.pkl')
    df['distance'] = df.apply(lambda x: great_circle((x['coords_pnt']),(x['coords'])).km, axis=1)
    df.to_pickle('../pickles/nodes/distance/coastal_distance_' + str(i) + '.pkl')
    del(df)

# harbour    
for j in range(61): # this range will depend on the number of nodes  
    df = pd.read_pickle('../pickles/nodes/merged_coords_slices/harbour_coords_combined_' + str(j) + '.pkl')
    df['distance'] = df.apply(lambda x: great_circle((x['coords_pnt']),(x['coords'])).km, axis=1)
    df.to_pickle('../pickles/nodes/distance/harbour_distance_' + str(j) + '.pkl')
    del(df)

# regional    
for k in range(3): # this range will depend on the number of nodes      
    df = pd.read_pickle('../pickles/nodes/merged_coords_slices/regional_coords_combined_' + str(k) + '.pkl')
    df['distance'] = df.apply(lambda x: great_circle((x['coords_pnt']),(x['coords'])).km, axis=1)
    df.to_pickle('../pickles/nodes/distance/regional_distance_' + str(k) + '.pkl')
    del(df)