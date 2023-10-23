"""
Contains function related to reading the CSV files containing the raw data and generating pandas dataframe.
"""

import pandas as pd
import numpy as np
#from params import Params

def scale_z(z,minz,maxz,nbins):
    """
    bin the z axis
    """
    minz = np.abs(minz)
    maxz = np.abs(maxz)
    z = np.add(minz,z)
    return (z * float(nbins-1) / (minz+maxz)).astype(np.int)

def scale_phi(phi, phi_bins):
    """
    bin phi
    """
    return (phi * float(phi_bins-1) / (2*np.pi)).astype(np.int)


#calc_r, theta, eta, phi, draw_scatter from Steve's notebook
def calc_r(x, y):
    """Cylindrical radius"""
    return np.sqrt(x**2 + y**2)

def calc_theta(r, z):
    """Zenith angle ranging from [0, pi]"""
    return np.arctan2(r, z)

def calc_eta(theta):
    """Pseudo-rapidity"""
    return -1. * np.log(np.tan(theta / 2.))

def calc_phi(x, y):
    """Azimuthal angle in the x-y plane. Results are in the interval [0;2*pi]"""
    phi = np.arctan2(y, x)
    phi_idx = np.where(phi<0)
    phi.iloc[phi_idx] += 2*np.pi
    return phi

def calc_tranverse_momentum(theta, p):
    """
    Transverse momentum (momentum in the x-y plane)
    """
    return np.sin(theta) * p
