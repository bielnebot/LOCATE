#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ivan Hernandez

Concatenate df's for coastal and harbour 

These df's only have the obs and distance column

The number of dfs per domain will depend on the size of the domain and how many nodes there were / files were generated in previous steps
"""

import pandas as pd

# harbour files
df_h_0 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_0.pkl') 
df_h_1 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_1.pkl') 
df_h_2 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_2.pkl')
df_h_3 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_3.pkl')
df_h_4 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_4.pkl')
df_h_5 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_5.pkl')
df_h_6 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_6.pkl')
df_h_7 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_7.pkl')
df_h_8 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_8.pkl')
df_h_9 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_9.pkl')
df_h_10 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_10.pkl')

df_h_11 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_11.pkl') 
df_h_12 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_12.pkl')
df_h_13 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_13.pkl')
df_h_14 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_14.pkl')
df_h_15 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_15.pkl')
df_h_16 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_16.pkl')
df_h_17 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_17.pkl')
df_h_18 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_18.pkl')
df_h_19 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_19.pkl')
df_h_20 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_20.pkl') 

df_h_21 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_21.pkl') 
df_h_22 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_22.pkl')
df_h_23 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_23.pkl')
df_h_24 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_24.pkl')
df_h_25 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_25.pkl')
df_h_26 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_26.pkl')
df_h_27 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_27.pkl')
df_h_28 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_28.pkl')
df_h_29 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_29.pkl')
df_h_30 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_30.pkl')  

df_h_31 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_31.pkl') 
df_h_32 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_32.pkl')
df_h_33 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_33.pkl')
df_h_34 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_34.pkl')
df_h_35 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_35.pkl')
df_h_36 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_36.pkl') 
df_h_37 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_37.pkl')
df_h_38 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_38.pkl')
df_h_39 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_39.pkl')
df_h_40 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_40.pkl')

df_h_41 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_41.pkl') 
df_h_42 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_42.pkl')
df_h_43 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_43.pkl')
df_h_44 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_44.pkl')
df_h_45 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_45.pkl')
df_h_46 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_46.pkl')
df_h_47 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_47.pkl')
df_h_48 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_48.pkl')
df_h_49 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_49.pkl')
df_h_50 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_50.pkl') 

df_h_51 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_51.pkl') 
df_h_52 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_52.pkl')
df_h_53 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_53.pkl')
df_h_54 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_54.pkl')
df_h_55 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_55.pkl')
df_h_56 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_56.pkl')
df_h_57 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_57.pkl')
df_h_58 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_58.pkl')
df_h_59 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_59.pkl')
df_h_60 = pd.read_pickle('../pickles/nodes/distance_min/slices/harbour_distance_min_60.pkl') 



# coastal files
df_c_0 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_0.pkl') 
df_c_1 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_1.pkl') 
df_c_2 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_2.pkl')
df_c_3 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_3.pkl')
df_c_4 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_4.pkl')
df_c_5 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_5.pkl')
df_c_6 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_6.pkl')
df_c_7 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_7.pkl')
df_c_8 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_8.pkl')
df_c_9 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_9.pkl')
df_c_10 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_10.pkl')

df_c_11 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_11.pkl') 
df_c_12 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_12.pkl')
df_c_13 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_13.pkl')
df_c_14 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_14.pkl')
df_c_15 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_15.pkl')
df_c_16 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_16.pkl')
df_c_17 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_17.pkl')
df_c_18 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_18.pkl')
df_c_19 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_19.pkl')
df_c_20 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_20.pkl') 

df_c_21 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_21.pkl') 
df_c_22 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_22.pkl')
df_c_23 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_23.pkl')
df_c_24 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_24.pkl')
df_c_25 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_25.pkl')
df_c_26 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_26.pkl')
df_c_27 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_27.pkl')
df_c_28 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_28.pkl')
df_c_29 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_29.pkl')
df_c_30 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_30.pkl')  

df_c_31 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_31.pkl') 
df_c_32 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_32.pkl')
df_c_33 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_33.pkl')
df_c_34 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_34.pkl')
df_c_35 = pd.read_pickle('../pickles/nodes/distance_min/slices/coastal_distance_min_35.pkl')

# regional files
df_r_0 = pd.read_pickle('../pickles/nodes/distance_min/slices/regional_distance_min_0.pkl') 
df_r_1 = pd.read_pickle('../pickles/nodes/distance_min/slices/regional_distance_min_1.pkl') 
df_r_2 = pd.read_pickle('../pickles/nodes/distance_min/slices/regional_distance_min_2.pkl')


# prepare dataframes for concatenation
data_harbour = [df_h_0, 
                df_h_1, 
                df_h_2, 
                df_h_3, 
                df_h_4, 
                df_h_5,
                df_h_6,
                df_h_7,
                df_h_8,
                df_h_9,
                df_h_10,
                df_h_11,
                df_h_12,
                df_h_13,
                df_h_14,
                df_h_15,
                df_h_16,
                df_h_17,
                df_h_18,
                df_h_19,
                df_h_20,
                df_h_21,
                df_h_22,
                df_h_23,
                df_h_24,
                df_h_25,
                df_h_26,
                df_h_27,
                df_h_28,
                df_h_29,
                df_h_30,
                df_h_31,
                df_h_32,
                df_h_33,
                df_h_34,
                df_h_35,
                df_h_36,
                df_h_37,
                df_h_38,
                df_h_39,
                df_h_40,
                df_h_41,
                df_h_42,
                df_h_43,
                df_h_44,
                df_h_45,
                df_h_46,
                df_h_47,
                df_h_48,
                df_h_49,
                df_h_50,
                df_h_51,
                df_h_52,
                df_h_53,
                df_h_54,
                df_h_55,
                df_h_56,
                df_h_57,
                df_h_58,
                df_h_59,
                df_h_60,
                ]

# prepare dataframes for concatenation
data_coastal = [df_c_0, 
                df_c_1, 
                df_c_2, 
                df_c_3, 
                df_c_4, 
                df_c_5,
                df_c_6,
                df_c_7,
                df_c_8,
                df_c_9,
                df_c_10,
                df_c_11,
                df_c_12,
                df_c_13,
                df_c_14,
                df_c_15,
                df_c_16,
                df_c_17,
                df_c_18,
                df_c_19,
                df_c_20,
                df_c_21,
                df_c_22,
                df_c_23,
                df_c_24,
                df_c_25,
                df_c_26,
                df_c_27,
                df_c_28,
                df_c_29,
                df_c_30,
                df_c_31,
                df_c_32,
                df_c_33,
                df_c_34,
                df_c_35
                ]

# prepare dataframes for concatenation
data_regional = [df_r_0, 
                df_r_1, 
                df_r_2
                ]


# concatenate the dataframes
df_h = pd.concat(data_harbour, ignore_index=True, sort=False)
df_c = pd.concat(data_coastal, ignore_index=True, sort=False)
df_r = pd.concat(data_regional, ignore_index=True, sort=False)

df_h.to_pickle('../pickles/nodes/distance_min/harbour_distance_min.pkl')
df_c.to_pickle('../pickles/nodes/distance_min/coastal_distance_min.pkl')
df_r.to_pickle('../pickles/nodes/distance_min/regional_distance_min.pkl')