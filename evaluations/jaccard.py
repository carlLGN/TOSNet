""" Intersection-over-Union (IoU) metric.
Reference: https://github.com/scaelles/DEXTR-PyTorch/blob/master/evaluation/evaluation.py
"""
import numpy as np


def jaccard(annotation, segmentation, void_pixels=None):

    assert(annotation.shape == segmentation.shape)

    if void_pixels is None:
        void_pixels = np.zeros_like(annotation)
    assert(void_pixels.shape == annotation.shape)

    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)
    void_pixels = void_pixels.astype(bool)
    if np.isclose(np.sum(annotation & np.logical_not(void_pixels)), 0) and np.isclose(np.sum(segmentation & np.logical_not(void_pixels)), 0):
        return 1
    else:
        return np.sum(((annotation & segmentation) & np.logical_not(void_pixels))) / \
               np.sum(((annotation | segmentation) & np.logical_not(void_pixels)), dtype=np.float32)