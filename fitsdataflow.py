import numpy as np
import pandas as pd
import astropy.io.fits as fits
from astropy.table import Table
import glob
import os

folder_path = '/home/aswin/vs code/Summer-Project-2024/Coarse_grained_merger_history/Data/BH'

def id_to_data_puller_fits(search_value, parameter, folder_path):
    fits_files = sorted(glob.glob(folder_path + '/*.fits'))
    redshifts = [file.split('z_')[-1].split('.fits')[0] for file in fits_files]
    final_data = np.zeros(len(fits_files))  # Default to 0.0
    
    for idx, file in enumerate(fits_files):
        with fits.open(file) as hdulist:
            data = Table(hdulist[1].data)
            df = data.to_pandas()
            match = df[df['ID'] == search_value]
            if not match.empty:
                final_data[idx] = match[parameter].values[0]
    
    return final_data.tolist(), redshifts

def extract_mass_filtered_ids(fits_file, m_lower, m_upper):
    with fits.open(fits_file) as hdulist:
        data = Table(hdulist[1].data).to_pandas()
        
        # Filter masses within range
        mass_filtered = data[(data['Mass'] > m_lower) & (data['Mass'] < m_upper)]
        
        # Extract corresponding IDs
        id_list = mass_filtered['ID'].tolist()
    
    return id_list