#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Prepares the nodes, creates tuples of lat, lon, and creates an obs column from the index for later steps
"""

import pandas as pd
import geopy.distance
from functools import reduce
import math
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/coords/'
os.makedirs(path_pickles, exist_ok=True)

# read the differentiated nodes (land=True , water=False)
df_h = pd.read_pickle('../pickles/nodes/land_water/harbour_nodes_by_type.pkl') 
df_c = pd.read_pickle('../pickles/nodes/land_water/coastal_nodes_by_type.pkl') 
df_r = pd.read_pickle('../pickles/nodes/land_water/regional_nodes_by_type.pkl') 

# creates tuples of coordinate points from lat, lon and puts them in a column
df_h['coords_pnt'] = list(zip(df_h.latitude, df_h.longitude))
df_c['coords_pnt'] = list(zip(df_c.latitude, df_c.longitude))
df_r['coords_pnt'] = list(zip(df_r.latitude, df_r.longitude))

# create col called obs which has the index value to do the merge later on. 
# This is added at the end of the df (as default)
df_h['obs'] = df_h.index
df_c['obs'] = df_c.index
df_r['obs'] = df_r.index

# to move the cobs column from the end to the beginning of the df to keep things neat
# pop out (remove) the obs column and copy to a new df
pop_h = df_h.pop("obs")
pop_c = df_c.pop("obs")
pop_r = df_r.pop("obs")

# insert obs column on the 1st position of the df
df_h.insert(0, "obs", pop_h)
df_c.insert(0, "obs", pop_c)
df_r.insert(0, "obs", pop_r)

# save data to a pickle
df_h.to_pickle('../pickles/nodes/coords/harbour_nodes_coords.pkl')
df_c.to_pickle('../pickles/nodes/coords/coastal_nodes_coords.pkl')
df_r.to_pickle('../pickles/nodes/coords/regional_nodes_coords.pkl')

