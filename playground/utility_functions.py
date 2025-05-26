import numpy as np

# Optional Downsampling Feature so that GPU does not burn down
def downsampler(array: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsamples the input array by the specified factor.
    
    Args:
        array (np.ndarray): The input array to be downsampled.
        factor (int): The downsampling factor.

    Returns:
        np.ndarray: The downsampled array.
    """
    return array[::factor]

# Cleans up the data by removing the trash starting machine data sample points
def snipper(array: np.ndarray) -> np.ndarray:
    """
    Removes the initial unwanted data points from the input array.
    
    Args:
        array (np.ndarray): The input array to be cleaned.

    Returns:
        np.ndarray: The cleaned array with initial trash data removed.
    """
    x = len(array)
    r = x % int(np.floor((x / 10000)) * 10000)
    array = array[r:]
    return array

def extractor(dataframe) -> tuple:
    """
    Splits the dataframe into features and labels.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe containing features and labels.

    Returns:
        tuple: A tuple containing the features (x) and labels (y).
    """
    y = dataframe['label']
    x = dataframe.drop('label', axis=1)
    return x, y 

def asinput(dataframe) -> np.ndarray:
    """
    Reshapes the dataframe into a 3D array for model input.
    
    Args:
        dataframe (pd.DataFrame): The input dataframe to be reshaped.

    Returns:
        np.ndarray: The reshaped 3D array.
    """
    return np.reshape(np.array(dataframe), (len(np.array(dataframe)), -1, 1))