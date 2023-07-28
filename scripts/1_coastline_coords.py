#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez
Previous processing and conversion from high resolution coastline data as shapefile to csv format is assumed, using QGIS sofwtware or alternative, editing to the required study domain coordinates. 


If the original data is in the form of a linestring, a polygon should be created in QGIS by adding an extra point(s), exporting data as a polygon in csv format.


If the original data is in the form of a linestring, a polygon should be created in QGIS by adding an extra point(s), exporting data as a polygon in csv format. These data are not included in GitHub. The original shapefiles can be downloaded from https://agricultura.gencat.cat/ca/serveis/cartografia-sig/bases-cartografiques/cartografia-referencia/linia-costa/ in .shp format.

Csv data are processed and saved as a pickle for later use.
"""

import pandas as pd
import numpy as np
import os

# import file from another folder
import sys
sys.path.append('../')
from config import sample_data as cfg

# creates folder for pickles if it does not exist
path_pickles = '../pickles/coords/'
os.makedirs(path_pickles, exist_ok=True)


# these data are not included in GitHub
df_poly = pd.read_csv ('../coastline/coastline_polygon.csv')


df_poly = pd.read_csv ('../coastline/cat_coastline_poly.csv')


# data is the csv is in one cell. This may vary depending on how the csv file is created
# select the only cell with data
poly_coords = df_poly.iloc[0]['WKT']

# strip the leading and trailing characters from the coordinates
poly_coords = poly_coords.strip('MULTIPOLYGON (((')
poly_coords = poly_coords.strip(')))')

# puts the comma separated coordinates into a list
coastline = poly_coords.split(",")

# create a dataframe from the list
df = pd.DataFrame(coastline)

# rename column 0 to coords for ease
# this may vary depending on the structure of list
df.rename(columns = {0:'coords'}, inplace = True)

# create arrays for lat and lon
lat = []
lon = []
# splits coords into lon and lat. Create two lists for the loop results to be placed
for row in df['coords']:
    # Try to,
    try:
        # Split the row by comma and append everything before the comma to lat
        lon.append(row.split(' ')[0])
        # Split the row by comma and append everything after the comma to lon
        lat.append(row.split(' ')[1])
    # But if you get an error
    except:
        # append a missing value to lat
        lon.append(np.NaN)
        # append a missing value to lon
        lat.append(np.NaN)

# Create two new columns from lat and lon
df['longitude'] = lon
df['latitude'] = lat

# assign data type
df = df.astype({'longitude':'float'})
df = df.astype({'latitude':'float'})

# drop original coords column
df.drop('coords', axis=1, inplace=True)

"""
# if the study area is smaller than that of the data in the csv file
# assumes northern hemisphere
# drop rows outside the study area
df = df.drop(df[(df['longitude'] < cfg.domain_W)].index)
df = df.drop(df[(df['longitude'] > cfg.domain_E)].index)
df = df.drop(df[(df['latitude'] > cfg.domain_N)].index)
df = df.drop(df[(df['latitude'] < cfg.domain_S)].index)
"""

# convert lat and lon to tuples in new coords column to be used later 
df['coords'] = list(zip(df.latitude, df.longitude))

# save the points within study area to a pickle
df.to_pickle('../pickles/coords/coastline_polygon.pkl') 
print (df)