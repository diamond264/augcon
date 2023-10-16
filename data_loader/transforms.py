"Vectorized transformation functions for mobile sensor time series"
import itertools
import numpy as np
import scipy.interpolate
from transforms3d.axangles import axangle2mat

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

An re-implemention of
T. T. Um et al., “Data augmentation of wearable sensor data for parkinson’s disease monitoring using convolutional neural networks,” in Proceedings of the 19th ACM International Conference on Multimodal Interaction, ser. ICMI 2017. New York, NY, USA: ACM, 2017, pp. 216–220.

https://dl.acm.org/citation.cfm?id=3136817

https://arxiv.org/abs/1706.00527

@inproceedings{TerryUm_ICMI2017, author = {Um, Terry T. and Pfister, Franz M. J. and Pichler, Daniel and Endo, Satoshi and Lang, Muriel and Hirche, Sandra and Fietzek, Urban and Kuli\'{c}, Dana}, title = {Data Augmentation of Wearable Sensor Data for Parkinson's Disease Monitoring Using Convolutional Neural Networks}, booktitle = {Proceedings of the 19th ACM International Conference on Multimodal Interaction}, series = {ICMI 2017}, year = {2017}, isbn = {978-1-4503-5543-8}, location = {Glasgow, UK}, pages = {216--220}, numpages = {5}, doi = {10.1145/3136755.3136817}, acmid = {3136817}, publisher = {ACM}, address = {New York, NY, USA}, keywords = {Parkinson\&#39;s disease, convolutional neural networks, data augmentation, health monitoring, motor state detection, wearable sensor}, }

"""

def noise_transform_vectorized(X, sigma=0.05):
    """
    Adding random Gaussian noise with mean 0
    """
    noise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X + noise

def scaling_transform_vectorized(X, sigma=0.1):
    """
    Scaling by a random factor
    """
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(X.shape[0], 1))
    return X * scaling_factor

def rotation_transform_vectorized(X):
    """
    Applying a random 3D rotation
    """
    axis = np.random.uniform(low=-1, high=1, size=X.shape[0])
    angle = np.random.uniform(low=-np.pi, high=np.pi)

    # return np.matmul(X.transpose(), axangle2mat(axis,angle)).transpose()
    return np.matmul(axangle2mat(axis,angle), X)

def negate_transform_vectorized(X, prob=0.5):
    """
    Inverting the signals independently for each element in the batch with a specified probability
    """
    if np.random.random() < prob:
        return -X
    else:
        return X+0

def time_flip_transform_vectorized(X, prob=0.5):
    """
    Reversing the direction of time independently for each element in the batch with a specified probability
    """
    if np.random.random() < prob:
        return np.flip(X, axis=1)
    else:
        return X+0

def channel_shuffle_transform_vectorized(X):
    """
    Shuffling the different channels
    
    Note: it might consume a lot of memory if the number of channels is high
    """
    channels = X.shape[0]
    channel_indices = np.arange(channels)
    np.random.shuffle(channel_indices)

    X_transformed = X[channel_indices]
    return X_transformed

def time_segment_permutation_transform_vectorized(X, num_segments=4, minSegLength=10):
    """
    Randomly scrambling sections of the signal
    """
    X_transformed = np.zeros(X.shape)
    idx = np.random.permutation(num_segments)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(num_segments+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[1]-minSegLength, num_segments-1))
        segs[-1] = X.shape[1]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(num_segments):
        x_temp = X[:,segs[idx[ii]]:segs[idx[ii]+1]]
        X_transformed[:,pp:pp+x_temp.shape[1]] = x_temp
        pp += x_temp.shape[1]
    return X_transformed

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[0],1))*(np.arange(0,X.shape[1], (X.shape[1]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[0]))
    x_range = np.arange(X.shape[1])
    cs_x = scipy.interpolate.CubicSpline(xx[:,0], yy[:,0])
    cs_y = scipy.interpolate.CubicSpline(xx[:,1], yy[:,1])
    cs_z = scipy.interpolate.CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[1]-1)/tt_cum[-1,0],(X.shape[1]-1)/tt_cum[-1,1],(X.shape[1]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum.transpose()

def time_warp_transform_vectorized(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[1])
    X_new[0,:] = np.interp(x_range, tt_new[0,:], X[0,:])
    X_new[1,:] = np.interp(x_range, tt_new[1,:], X[1,:])
    X_new[2,:] = np.interp(x_range, tt_new[2,:], X[2,:])
    return X_new

def is_scaling_factor_invalid(scaling_factor, min_scale_sigma):
    """
    Ensure each of the abs values of the scaling
    factors are greater than the min
    """
    for i in range(len(scaling_factor)):
        if abs(scaling_factor[i] - 1) < min_scale_sigma:
            return True
    return False

def tpn_noise(X, choice):
    if choice == 1:
        return noise_transform_vectorized(X, 0.1)
    else:
        return X+0

def tpn_rotate(X, choice):
    if choice == 1:
        return rotation_transform_vectorized(X)
    else:
        return X+0

def tpn_negate(X, choice):
    if choice == 1:
        return negate_transform_vectorized(X)
    else:
        return X+0

def tpn_flip(X, choice):
    if choice == 1:
        return np.flip(X, axis=1)
    else: return X+0

def tpn_scale(X, choice, scale_range=0.5, min_scale_diff=0.15):
    if choice == 1:
        low = 1 - scale_range
        high = 1 + scale_range
        scaling_factor = np.random.uniform(
            low=low, high=high, size=(X.shape[0])
        )
        while is_scaling_factor_invalid(scaling_factor, min_scale_diff):
            scaling_factor = np.random.uniform(
                low=low, high=high, size=(X.shape[0])
            )
        X_new = np.zeros(X.shape)
        for i in range(3):
            X_new[i, :] = X[i, :] * scaling_factor[i]
        return X_new
    else:
        return X+0

def tpn_permute(X, choice):
    if choice == 1:
        return time_segment_permutation_transform_vectorized(X)
    else:
        return X+0

def tpn_time_warp(X, choice):
    if choice == 1:
        return time_warp_transform_vectorized(X)
    else:
        return X+0

def tpn_channel_shuffle(X, choice):
    if choice == 1:
        return channel_shuffle_transform_vectorized(X)
    else:
        return X+0