"""Explicit temporal-change features for UrbanSARFloods tiles."""

import numpy as np
PRE_COHERENCE_VH = 0
PRE_COHERENCE_VV = 1
CO_COHERENCE_VH = 2
CO_COHERENCE_VV = 3
PRE_INTENSITY_VH = 4
PRE_INTENSITY_VV = 5
POST_INTENSITY_VH = 6
POST_INTENSITY_VV = 7
DIFFERENCE_VV = 8
DIFFERENCE_VH = 9


def add_intensity_differences(image):
    if image.ndim != 3 or image.shape[0] != 8:
        raise ValueError(f"Expected an (8, height, width) array, got {image.shape}")
    difference_vv = image[POST_INTENSITY_VV] - image[PRE_INTENSITY_VV]
    difference_vh = image[POST_INTENSITY_VH] - image[PRE_INTENSITY_VH]
    return np.concatenate([image, difference_vv[np.newaxis], difference_vh[np.newaxis]])


def add_coherence_differences(image):
    if image.ndim != 3 or image.shape[0] != 8:
        raise ValueError(f"Expected an (8, height, width) array, got {image.shape}")
    difference_vv = image[PRE_COHERENCE_VV] - image[CO_COHERENCE_VV]
    difference_vh = image[PRE_COHERENCE_VH] - image[CO_COHERENCE_VH]
    return np.concatenate([image, difference_vv[np.newaxis], difference_vh[np.newaxis]])


def add_all_differences(image):
    return np.concatenate([add_intensity_differences(image), add_coherence_differences(image)[DIFFERENCE_VV:]])
