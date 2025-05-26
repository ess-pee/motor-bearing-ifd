import os
import os.path

import numpy as np
import pandas as pd
import scipy.io.matlab as mat

from utility_functions import downsampler

def extract_cwru_data(path='data/cwru', downsample: bool = False, factor: int = 1):
    """
    Args:
        path (str): The path to the directory containing the data files.
        downsample (bool): Whether to downsample the data or not.
        factor (int): The downsampling factor. Default is 1 (no downsampling).
    Returns:
        tuple: A tuple containing the extracted data and the class labels.
    """
    # This is a very specific function to extract the CWRU data from the files in the cwru directory.
    # It only works for this specific directory structure and file names.
    # It is was engineered after enduring a lot of pain and suffering for this directory.

    # CWRU data is stored in a specific directory structure and file names. # Sample Rate: 12000 Hz
    cwru_classes = ['N', 'B', 'IR', 'OR']
    pull_files = ['12K', 'N']
    SAMPLE_RATE = 12000 # CWRU default

    cwru_files = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files]
    cwru_12k_files = [file for file in cwru_files for name in pull_files if name in file ]
    data = {clss : [values for file in cwru_12k_files if clss in file for key, values in mat.loadmat(file).items() if 'DE' in key] for clss in cwru_classes}

    if downsample:
        # Optional Downsampling Feature, factor 1 for orignal 
        cwru_data = {key : [downsampler(value, factor) for value in values] for key, values in data.items()} 
        sample_rate = SAMPLE_RATE // factor
    else:
        cwru_data = data
        sample_rate = SAMPLE_RATE
    
    return cwru_data, cwru_classes, sample_rate

def extract_mfd_data(path='data/mfd', downsample: bool = True, factor: int = 5):
    """
    Args:
        path (str): The path to the directory containing the data files.
        downsample (bool): Whether to downsample the data or not.
        factor (int): The downsampling factor. Default is 5 (recommended to reduce overhead) 1 to retain the original.
    Returns:
        tuple: A tuple containing the extracted data and the class labels.
    """

    # This is a very specific function to extract the mafaulda data from the files in the mfd directory.
    # It only works for this specific directory structure and file names.
    # It was engineered after enduring a lot of pain and suffering for this directory.
    
    # MFD data is stored in a specific directory structure and file names. Sample Rate: 50000 Hz
    mfd_classes = ['normal', 'ball', 'cage', 'outer']
    SAMPLE_RATE = 50000 # MFD default

    mfd_files = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files]
    data = {clss : [np.array(pd.read_csv(file).iloc[:, 1]) for file in mfd_files if clss in file] for clss in mfd_classes}

    if downsample:
        # Optional Downsampling Feature, factor 1 for orignal 
        mfd_data = {key : [downsampler(value, factor) for value in values] for key, values in data.items()}
        sample_rate = SAMPLE_RATE // factor
    else:
        mfd_data = data
        sample_rate = SAMPLE_RATE

    return mfd_data, mfd_classes, sample_rate

def extract_tri_data(path='data/triaxial', downsample: bool = False, factor: int = 1):
    """
    Args:
        path (str): The path to the directory containing the data files.
    Returns:
        tuple: A tuple containing the extracted data and the class labels.
    """
    # This is a very specific function to extract the mafaulda data from the files in the mfd directory.
    # It only works for this specific directory structure and file names.
    # It is was engineered after enduring a lot of pain and suffering for this directory.

    # This block is for the triaxial bearing vibration dataset. Sample Rate: 10000 Hz
    tri_classes = ['Healthy', 'Inner', 'Outer']
    SAMPLE_RATE = 10000 # TRI default

    tri_files = [os.path.join(root, name) for root, dirs, files in os.walk(path) for name in files]
    data = {clss : [np.array(pd.read_csv(file)[' X-axis']) for file in tri_files if clss in file] for clss in tri_classes}

    if downsample:
        # Optional Downsampling Feature, factor 1 for orignal 
        tri_data = {key : [downsampler(value, factor) for value in values] for key, values in data.items()} 
        sample_rate = SAMPLE_RATE // factor
    else:
        tri_data = data
        sample_rate = SAMPLE_RATE

    return tri_data, tri_classes, sample_rate
