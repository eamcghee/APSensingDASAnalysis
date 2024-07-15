'''
=============================================================================
PhasetoStrain_npy.py

Read-in AP Sensing DAS phase data, convert to nanostrain, save as .npy
Load .npy files and save as a list for indexing 
For one selected channel only
Took 45 min for 32 files (Input: 1 Hdf5 file, 2.5 GB; Output: 1 .npy file 1.2 MB)
=============================================================================
=============================================================================
Read-in AP Sensing DAS Phase Data, convert to nanostrain, save as .npy arrays
This code is for one given channel selection 
=============================================================================
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon 15 July 2024

@author: em
"""

import os, h5py, glob
import numpy as np
from numpy import asarray
from numpy import savetxt
from obspy import read

'''
=============================================================================
Section 1: Data Read-In & Convert from Phase to Nanostrain: Hdf5 files to .npy 
=============================================================================
'''

# Read-in sample AP Sensing files; for reference, 0630 was AN sunrise, but noise floor shift happened around 0700L:

file1 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000236_2023-09-24_12.46.31.73962.hdf5' #12:46 = 0546L
file2 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000237_2023-09-24_12.51.31.73841.hdf5' #12:51 = 0551L
file3 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000238_2023-09-24_12.56.31.73818.hdf5' #12:56 = 0556L
file4 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000239_2023-09-24_13.01.31.73795.hdf5' #13:01 = 0601L
file5 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000240_2023-09-24_13.06.31.73772.hdf5' #13:06 = 0606L
file6 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000241_2023-09-24_13.11.31.73749.hdf5' #13:11 = 0611L
file7 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000242_2023-09-24_13.16.31.73727.hdf5' #13:16 = 0616L
file8 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000243_2023-09-24_13.21.31.73704.hdf5' #13:21 = 0621L
file9 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000244_2023-09-24_13.26.31.73682.hdf5' #13:26 = 0626L
file10 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000245_2023-09-24_13.31.31.73660.hdf5' #13:31 = 0631L
file11 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000246_2023-09-24_13.36.31.73638.hdf5' #13:36 = 0636L
file12 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000247_2023-09-24_13.41.31.73616.hdf5' #13:41 = 0641L
file13 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000248_2023-09-24_13.46.31.73594.hdf5' #13:46 = 0646L
file14 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000249_2023-09-24_13.51.31.73572.hdf5' #13:51 = 0651L
file15 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000250_2023-09-24_13.56.31.73550.hdf5' #13:56 = 0656L
file16 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000251_2023-09-24_14.01.31.73528.hdf5' #14:01 = 0701L
file17 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000252_2023-09-24_14.06.31.73506.hdf5' #14:06 = 0706L
file18 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000253_2023-09-24_14.11.31.73483.hdf5' #14:11 = 0711L
file19 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000254_2023-09-24_14.16.31.73461.hdf5' #14:16 = 0716L
file20 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000255_2023-09-24_14.21.31.73439.hdf5' #14:21 = 0721L
file21 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000256_2023-09-24_14.26.31.73416.hdf5' #14:26 = 0726L
file22 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000257_2023-09-24_14.31.31.73393.hdf5' #14:31 = 0731L
file23 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000258_2023-09-24_14.36.31.73370.hdf5' #14:36 = 0736L
file24 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000259_2023-09-24_14.41.31.73346.hdf5' #14:41 = 0741L
file25 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000260_2023-09-24_14.46.31.73322.hdf5' #14:46 = 0746L
file26 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000261_2023-09-24_14.51.31.73298.hdf5' #14:51 = 0751L
file27 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000262_2023-09-24_14.56.31.73274.hdf5' #14:56 = 0756L 
file28 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000263_2023-09-24_15.01.31.73250.hdf5' #15:01 = 0801L 
file29 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000264_2023-09-24_15.06.31.73226.hdf5' #15:06 = 0806L 
file30 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000265_2023-09-24_15.11.31.73202.hdf5' #15:11 = 0811L 
file31 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000266_2023-09-24_15.16.31.73177.hdf5' #15:16 = 0816L 
file32 = '/Volumes/CATALOGDR00/OSIRIS REx/DAS All/APSensing/09-24-2023/0000000267_2023-09-24_15.21.31.73153.hdf5' #15:21 = 0821L

# Directory where you want to save the files
directory = '/Users/em/PROJECTS/OREX/DATA/'

## set channel; 'res' is the raw DAS das in phase units (rad / m/s?), then convert phase to strain
channel = 900 #Gauge Length: 4.9005713 m; channel 102 ~500m; channel 10 ~49m; channel 100 ~490m; channel 1000 ~4900m; 
res = []

# Loop over the indices
for i in range(1, 33):
    file_num = f'file{i}'
    file_path = f'{directory}res{i}_{channel}.npy'
    
    # Use eval to dynamically get the filename variable
    with h5py.File(eval(file_num), 'r') as f:
        results = f.get('DAS')
        res = results[:, channel]
        delta = f['ProcessingServer']['DataRate'][0]
        dx = f['ProcessingServer']['SpatialSampling'][0]
        GL = f['ProcessingServer']['GaugeLength'][0]
        FiberRefractiveIndex = f['Interrogator']['FiberRefractiveIndex'][0]
        phase_to_strain = (1550*10**-9)/(4*np.pi*FiberRefractiveIndex * GL * 0.78)
        res = np.array(res) * phase_to_strain
    
    # Save the array to the specified location
    np.save(file_path, res)

    # Show which file is complete 
    print(f'{file_num} complete')
    

# '''
# =============================================================================
# # Section 2: Data Load for Next Steps: Create a list to index of .npy arrays
# =============================================================================
# '''

# import os, h5py, glob
# import numpy as np
# from numpy import asarray
# from numpy import savetxt
# from obspy import read

# channel = 900 

# #res_arrays will be a list containing all the loaded numpy arrays from res1 to res32
# #access each array by indexing into the list, e.g., res_arrays[0] for res1, res_arrays[1] for res2, and so on.
    
# # Define the directory where the files are stored
# directory = '/Users/em/PROJECTS/OREX/DATA/'

# # Initialize a list to store the loaded arrays
# res_arrays = []

# # Loop over the range of numbers from 1 to 32
# for i in range(1, 33):
#     # Construct the file path
#     file_path = f'{directory}res{i}_{channel}.npy'
    
#     # Load the array
#     res = np.load(file_path)
    
#     # Append the loaded array to the list
#     res_arrays.append(res)

# # Now, res_arrays contains all the loaded arrays from res1 to res32    

# '''
# =============================================================================
# # Section 3: Table of Indices to Times. Create a list to index of .npy arrays
# =============================================================================
# '''
# import pandas as pd

# # file path to csv with array number in col 1; local time in col 2
# file_path = '/Users/em/PROJECTS/OREX/DATA/res_time.csv'

# # Read the CSV file into a Pandas DataFrame
# df = pd.read_csv(file_path)

# print(df)

# '''
# =============================================================================
# # The next .py scripts create PSDs or Spectrograms 



