#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez
Extracts nodes from the harbour, coastal, and regional netcdf files
Assumes these files are already downloaded and have undergone the resampling step, or are in regular grids
"""

import xarray as xr
import pandas as pd
import os

# creates folder for pickles if it does not exist
path_pickles = '../pickles/nodes/from_netcdf/'
os.makedirs(path_pickles, exist_ok=True)

# opens datasets and puts data into arrays
data_harbour = xr.open_dataset('../hydrodynamic_data/currents/harbour_resampled/harbour_coastal_2023-03-05_resampled.nc')
data_coastal = xr.open_dataset('../hydrodynamic_data/currents/coastal_resampled/coastal_IBI_2023-03-05_resampled.nc')
data_regional = xr.open_dataset('../hydrodynamic_data/currents/regional_resampled/IBI_2023-03-05_resampled.nc')

# puts arrays into dataframes
df_h = data_harbour.to_dataframe()
df_c = data_coastal.to_dataframe()
df_r = data_regional.to_dataframe()

# resets indices and flattens multidimensional data
df_h = df_h.reset_index()
df_c = df_c.reset_index()
df_r = df_r.reset_index()

# drop unecessary columns in harbour. Customise depending on data
df_h.drop('ssh', axis=1, inplace=True)
df_h.drop('salinity', axis=1, inplace=True)
df_h.drop('temperature', axis=1, inplace=True)
df_h.drop('ubar', axis=1, inplace=True)
df_h.drop('vbar', axis=1, inplace=True)
df_h.drop('depth', axis=1, inplace=True)
df_h.drop('time', axis=1, inplace=True)
df_h.drop('u', axis=1, inplace=True)
df_h.drop('v', axis=1, inplace=True)
df_h = df_h.reset_index()

# drop unecessary columns in coastal. Customise depending on data
df_c.drop('ssh', axis=1, inplace=True)
df_c.drop('salinity', axis=1, inplace=True)
df_c.drop('temperature', axis=1, inplace=True)
df_c.drop('ubar', axis=1, inplace=True)
df_c.drop('vbar', axis=1, inplace=True)
df_c.drop('depth', axis=1, inplace=True)
df_c.drop('time', axis=1, inplace=True)
df_c.drop('u', axis=1, inplace=True)
df_c.drop('v', axis=1, inplace=True)
df_c = df_c.reset_index()

# drop unecessary columns in regional. Customise depending on data
df_r.drop('zos', axis=1, inplace=True)
df_r.drop('thetao', axis=1, inplace=True)
df_r.drop('depth', axis=1, inplace=True)
df_r.drop('time', axis=1, inplace=True)
df_r.drop('u', axis=1, inplace=True)
df_r.drop('v', axis=1, inplace=True)
df_r = df_r.reset_index()


# drop unecessary index columns
df_h.drop('index', axis=1, inplace=True)
df_c.drop('index', axis=1, inplace=True)
df_r.drop('index', axis=1, inplace=True)


# drop duplicates from flattened multidimensional data and reindex
df_h.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
df_h = df_h.reset_index()
# drop again newly created index column
df_h.drop('index', axis=1, inplace=True)

df_c.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
df_c = df_c.reset_index()
# drop again newly created index column
df_c.drop('index', axis=1, inplace=True)

df_r.drop_duplicates(subset=['longitude', 'latitude'], keep='first', inplace=True)
df_r = df_r.reset_index()
# drop again newly created index column
df_r.drop('index', axis=1, inplace=True)

# save to pickles
df_h.to_pickle('../pickles/nodes/from_netcdf/harbour_nodes.pkl')
df_c.to_pickle('../pickles/nodes/from_netcdf/coastal_nodes.pkl') 
df_r.to_pickle('../pickles/nodes/from_netcdf/regional_nodes.pkl') 


