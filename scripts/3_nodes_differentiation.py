#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez
Differentiates nodes between land and water cells
"""


import pandas as pd
from shapely.geometry import Point, Polygon
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/land_water/'
os.makedirs(path_pickles, exist_ok=True)

# load the node data
df_h = pd.read_pickle("../pickles/nodes/from_netcdf/harbour_nodes.pkl")
df_c = pd.read_pickle("../pickles/nodes/from_netcdf/coastal_nodes.pkl")
df_r = pd.read_pickle("../pickles/nodes/from_netcdf/regional_nodes.pkl")

# load the cat coastline
df_cat = pd.read_pickle("../pickles/coords/coastline_polygon.pkl")


# puts tuples of the coastline coordinates into a list
coords = df_cat['coords'].values.tolist()

# puts the list of coordinates in polygon format
polygon = Polygon(coords)

# checks if a point is within the polygon. True=land, False=water
df_h['land_cell'] = df_h.apply(lambda x: polygon.contains(Point(x['latitude'], x['longitude'])), axis=1)
df_c['land_cell'] = df_c.apply(lambda x: polygon.contains(Point(x['latitude'], x['longitude'])), axis=1)
df_r['land_cell'] = df_r.apply(lambda x: polygon.contains(Point(x['latitude'], x['longitude'])), axis=1)

# save data to pickles
df_h.to_pickle('../pickles/nodes/land_water/harbour_nodes_by_type.pkl') 
df_c.to_pickle('../pickles/nodes/land_water/coastal_nodes_by_type.pkl') 
df_r.to_pickle('../pickles/nodes/land_water/regional_nodes_by_type.pkl') 
