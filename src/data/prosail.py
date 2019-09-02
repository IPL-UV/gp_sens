import numpy as np
import pandas as pd
from pyDOE2 import lhs
import prosail


# def prosail()
def prosail8d(N, CAB, CW, CM, LAI, LAD, PSOIL, SZA):
    """Generates Prosail Data
    
    Parameters
    ----------
    N : float
        Leaf Structure Parameter
    
    CAB : float 
        Leaf Chlorophyll Concentration
    
    CW : float 
        Equivalent Leaf Water
    
    CM : float
        Leaf Dry Matter
    
    LAI : float
        Leaf Area Index
    
    LAD : float
        Leaf angle distribution
    
    PSOIL : float 
        Solar Scalar 2 (Moisture)
    
    SZA : float
        Solar Zenith Angle
    
    Returns
    -------
    
    spectrum : np.ndarray (2101, )
        The generated spectrum for prosail
    """

    return prosail.run_prosail(
        n=N,  # Leaf Structure Parameter
        cab=CAB,  # Chlorophyll a + b
        car=0,  # Leaf Cartenoid Concentration
        cbrown=0,  # Senescent Pigment
        cw=CW,  # Equivalent Leaf Water
        cm=CM,  # Leaf Dry Matter
        lai=LAI,  # Leaf Area Index
        lidfa=LAD,  # Leaf Angle Distribution (a)
        hspot=0.01,  # HotSpot
        tts=SZA,  # Sun Zenith Angle
        tto=0,  # Sensor zenith angle
        psi=0.0,  # Relative Sensor-Solar Azimuth angle
        ant=0,  # leaf antho. concen.
        alpha=40,  #
        psoil=PSOIL,
        rsoil=1.0,
    )


def get_limits(limits="prosail8d"):

    if limits is "prosail8d":
        lims = np.array(
            [
                [1, 2.6],
                [0, 80],
                [1e-3, 0.08],
                [1e-3, 0.02],
                [0, 10],
                [10, 90],
                [0, 1],
                [0, 60],
            ]
        )

        rows = ["N", "CAB", "CW", "CM", "LAI", "LAD", "PSOIL", "SZA"]
        columns = ["min", "max"]
        limits_df = pd.DataFrame(lims, columns=columns, index=rows)
    else:
        raise ValueError(f"Unrecognized limits: {limits}")
    return limits_df


def latin_hypercube(data, n_samples=10, seed=None):
    np.random.RandomState(seed)
    n_features = data.shape[0]
    term1 = lhs(n_features, n_samples).T
    term2 = np.diff(data.values)
    term3 = data.values[:, 0].reshape(n_features, 1)
    return (term1 * term2 + term3).T


def prosail_generator(n_samples=1000):

    # Get Limits
    limits_df = get_limits("prosail8d")

    # Generate Samples (Latin)
    samples = latin_hypercube(limits_df, n_samples)

    spectrum = list()

    for n in range(n_samples):
        isample = samples[n, :]

        spectrum.append(
            prosail8d(
                N=isample[0],
                CAB=isample[1],
                CW=isample[2],
                CM=isample[3],
                LAI=isample[4],
                LAD=isample[5],
                PSOIL=isample[6],
                SZA=isample[7],
            )
        )

    columns = ["N", "Cab", "Cw", "Cm", "lai", "angle", "psoil", "tts"]
    X = pd.DataFrame(samples, columns=columns)
    return X, np.array(spectrum)
