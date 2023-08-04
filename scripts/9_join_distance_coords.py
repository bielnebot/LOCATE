#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Join the dfs with the min distance and the corresponding node coordinates
"""

import pandas as pd
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/to_netcdf/'
os.makedirs(path_pickles, exist_ok=True)

# read the pickles with the coordinates
df_h = pd.read_pickle('../pickles/nodes/coords/harbour_nodes_coords.pkl')
df_c = pd.read_pickle('../pickles/nodes/coords/coastal_nodes_coords.pkl') 
df_r = pd.read_pickle('../pickles/nodes/coords/regional_nodes_coords.pkl')  

# read the pickles with the min distance
df_h_min = pd.read_pickle('../pickles/nodes/distance_min/harbour_distance_min.pkl')
df_c_min = pd.read_pickle('../pickles/nodes/distance_min/coastal_distance_min.pkl') 
df_r_min = pd.read_pickle('../pickles/nodes/distance_min/regional_distance_min.pkl') 


# copy only the distance column before joining databases
df_h_dist = df_h_min[['distance']].copy()
df_c_dist = df_c_min[['distance']].copy()
df_r_dist = df_r_min[['distance']].copy()

df_h_dist = df_h_min[['distance']].copy()
df_c_dist = df_c_min[['distance']].copy()
df_r_dist = df_r_min[['distance']].copy()

# join dtistance and node coords dataframes
df_h_merged = df_h.join(df_h_dist)
df_c_merged = df_c.join(df_c_dist)
df_r_merged = df_r.join(df_r_dist)

# if land_cell == True then convert distance to a negative number
df_h_merged.loc[df_h_merged['land_cell'] == True, 'distance_shore'] = 0 - df_h_merged['distance']
df_h_merged.loc[df_h_merged['land_cell'] == False, 'distance_shore'] = df_h_merged['distance'] 
df_c_merged.loc[df_c_merged['land_cell'] == True, 'distance_shore'] = 0 - df_c_merged['distance']
df_c_merged.loc[df_c_merged['land_cell'] == False, 'distance_shore'] = df_c_merged['distance'] 
df_r_merged.loc[df_r_merged['land_cell'] == True, 'distance_shore'] = 0 - df_r_merged['distance']
df_r_merged.loc[df_r_merged['land_cell'] == False, 'distance_shore'] = df_r_merged['distance'] 

# drop unecessary columns
df_h_merged.drop('obs', axis=1, inplace=True)
df_h_merged.drop('land_cell', axis=1, inplace=True)
df_h_merged.drop('coords_pnt', axis=1, inplace=True)
df_h_merged.drop('distance', axis=1, inplace=True)

df_c_merged.drop('obs', axis=1, inplace=True)
df_c_merged.drop('land_cell', axis=1, inplace=True)
df_c_merged.drop('coords_pnt', axis=1, inplace=True)
df_c_merged.drop('distance', axis=1, inplace=True)

df_r_merged.drop('obs', axis=1, inplace=True)
df_r_merged.drop('land_cell', axis=1, inplace=True)
df_r_merged.drop('coords_pnt', axis=1, inplace=True)
df_r_merged.drop('distance', axis=1, inplace=True)

# save pickles
df_h_merged.to_pickle('../pickles/nodes/to_netcdf/harbour_nodes.pkl')
df_c_merged.to_pickle('../pickles/nodes/to_netcdf/coastal_nodes.pkl')
df_r_merged.to_pickle('../pickles/nodes/to_netcdf/regional_nodes.pkl')

