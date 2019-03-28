"""
Created on Apr 28, 2017
@author: xiagai
"""
import numpy as np
from scipy.ndimage import _ni_support
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, \
    distance_transform_edt


def dice(pred, label):
    """Compute the dice coeffcient, which is 2 * |(A ∩ B)| / |A| + |B|
       Range: [0, 1], and 0 is no overlap and 1 is perfect overlap
    Args:
        pred: The prediction array
        label: The label array
    Return:
        The dice coeffcient
    """
    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    intersection = np.count_nonzero(pred & label)
    pred_size = np.count_nonzero(pred)
    label_size = np.count_nonzero(label)

    try:
        dc = float(2 * intersection) / float(pred_size + label_size)
    except ZeroDivisionError:
        print("ZeroDivisionError, the inputs have all zero objects")
        dc = 0.0

    return dc


def jaccard(pred, label):
    """Compute the jaccard coeffcient which is |A ∩ B| / |A ∪ B|
       Range: [0, 1], and 0 is no overlap and 1 is perfect overlap
    Args:
        pred: The prediction array
        label: The label array
    Returns:
        The jaccard coeffcient
    """
    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    intersection = np.count_nonzero(pred & label)
    union = np.count_nonzero(pred | label)
    try:
        jc = float(intersection) / float(union)
    except ZeroDivisionError:
        print("ZeroDivisionError, the inputs have all zero objects")
        jc = 0.0

    return jc


def ravd(pred, label):
    """Compute the relative absolute volume difference, which is |A| - |B| / |B|
       Range: [-1, +inf), and 0 is perfect
    Args:
        pred: The prediction array
        label: The label array
    Returns:
        The relative absolute volume difference
    """
    pred = np.atleast_1d(pred.astype(np.bool))
    label = np.atleast_1d(label.astype(np.bool))

    pred_size = np.count_nonzero(pred)
    label_size = np.count_nonzero(label)

    try:
        ravd = float(pred_size - label_size) / float(label_size)
    except ZeroDivisionError:
        print("ZeroDivisionError, the label object have all zero objects")

    return ravd


def hausdorff(pred, label, voxelspacing=None, connectivity=1):
    """compute Hausdorff Distance.
    Args:
        pred: The prediction array
        label: The label array
        voxelspacing: float or sequence of floats, optional
            The voxelspacing in a distance unit i.e. spacing of elements
            along each dimension. If a sequence, must be of length equal to
            the input rank; if a single number, this is used for all axes.
        connectivity: int
            The neighbourhood/connectivity considered when determining the surface
            of the binary objects. This value is passed to 
            scipy.ndimage.morphology.generate_binary_structure and should usually be '>1'.
            Presumably does not influence the result in the case of the Hausdorff distance.
    Returns:
        hd: The Hausdorff Distance between inputs. The distance unit is the same as for the
            spacing of elements along each dimension, which is usually given in mm.
    """
    hd1 = _surface_distances(pred, label, voxelspacing, connectivity).max()
    hd2 = _surface_distances(label, pred, voxelspacing, connectivity).max()
    hd = max(hd1, hd2)
    return hd


def assd(pred, label, voxelspacing=None, connectivity=1):
    """compute the average symmetric surface distance
       It's not the mean value of asd1(average surface distance) and asd2,
       which is calculated in medpy's assd.
        Args:
            pred: prediction array
            label: label array
            voxelspacing: optional
            connectivity: It seems that 1 represents 4-neighbourhood
            and 2 represents 8-neightbourhood. 1 by default. Better just
            leave it there.
        Returns:
            The average symmetric surface distance
    """
    sds1 = _surface_distances(pred, label, voxelspacing, connectivity)
    sds2 = _surface_distances(label, pred, voxelspacing, connectivity)
    assd = (sds1.sum() + sds2.sum()) / (len(sds1) + len(sds2))
    return assd


def _surface_distances(input1, input2, voxelspacing=None, connectivity=1):
    input1 = np.atleast_1d(input1.astype(bool))
    input2 = np.atleast_1d(input2.astype(bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, input1.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    if 0 == np.count_nonzero(input1):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(input2):
        raise RuntimeError('The second supplied array does not contain any binary object.')

    # binary structure, used by binary_erosion()
    footprint = generate_binary_structure(input1.ndim, connectivity)
    # extract only 1-pixel border line of objects
    input1_border = input1 - binary_erosion(input1, structure=footprint, iterations=1)
    input2_border = input2 - binary_erosion(input2, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipy's distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~input2_border, sampling=voxelspacing)
    sds = dt[input1_border]

    return sds


# test pary
'''
x = np.array([[0, 1, 0],
              [1, 0, 1],
              [0, 1, 0]])
y = np.array([[1, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
print(_surface_distances(x, y))
print(_surface_distances(y, x))
print(assd(x, y)
'''