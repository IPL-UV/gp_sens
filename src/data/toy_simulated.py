import pandas as pd
import numpy as np
from typing import Tuple


def get_demo_data(dataset: str='prosail500') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Imports some demo datasets from Gustau and Jochem.
    
    Parameters
    ----------
    dataset : str (default='prosail500')
        The dataset to be extracted.
        Options: {'prosail1000', 'prosail500', 'prospect4'}

    Returns
    -------
    X : (n_samples x d_features)
        The samples and features to be used for predictions
    y : (n_samples x n_channels)
        The samples and channels to be predicted from the features.
    
    Example
    -------
    >> X, y = get_demo_data(dataset='prosail500')
    """
    
    # Check if dataset is str
    assert(isinstance(dataset, str))

    # Default data directory
    toy_data_directory = '/media/disk/erc/papers/2019_GP_SA/data'
    
    if dataset.lower() == 'prosail1000':
        
        filename = 'Directional_reflectance_PROSAIL1000_S2.txt'
        
        df = pd.read_csv(toy_data_directory + '/' + filename)
        
        X = df.loc[1:, :'0.00000.6']
        y = df.loc[1:, '443.00000':]
        
        features = [
            'Cab', 
            'Cw', 
            'Cm', 
            'LAI', 
            'angle', 
            'tts', 
            'psi'
        ]
        
        X.columns = features
        
    elif dataset.lower() == 'prosail500':
        
        filename = 'Directional_reflectance_PROSAIL.txt'
        
        df = pd.read_csv(toy_data_directory + '/' + filename)
        
        X = df.loc[:, :'0.00000.7']
        y = df.loc[:, '443.00000':]
        
        # Column names
        X.columns = [
            'N',
            'Cab', 
            'Cw',
            'Cm',
            'lai',
            'angle',
            'psoil',
            'tts',
        ]
        
    elif dataset.lower() == 'prospect4':
        
        filename = 'Reflectance_of_the_leaf.txt'
        
        df = pd.read_csv(toy_data_directory + '/' + filename)
        
        X = df.loc[:, :'0.00000.3']
        y = df.loc[:, '443.00000':]
        
        # Column names
        X.columns = [
            'N',
            'Cab', 
            'Cw',
            'Cm',
        ]
        
    else:
        raise ValueError('Unrecognized dataset.')

    # Clean up the channels
    channels = y.columns.values
    channels = [ichannel.split('.')[0] for ichannel in channels]
    y.columns = channels

    return X, y