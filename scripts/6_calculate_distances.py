#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan

Read the pickles and calculate the distance from each node to each point on the coastline using a lamda function.

Use great circle calculation, about 20x faster than Vicenty calculation and has an accuracy difference of ~0.14%.

Each iteration can take some time.

Resulting files can be several Gb in size.

Note: the regional domain did not require a for loop.
"""

import pandas as pd
from geopy.distance import great_circle



# save to a pickle
# delete the df variable -> very important to conserve memory and prevent the kernel from restarting after a few iterations

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
   
# if the regional domain has a large number of nodes, do a loop as above
df = pd.read_pickle('../pickles/nodes/merged_coords_slices/regional_coords_combined.pkl')
df['distance'] = df.apply(lambda x: great_circle((x['coords_pnt']),(x['coords'])).km, axis=1)
df.to_pickle('../pickles/nodes/distance/regional_distance.pkl')    
    