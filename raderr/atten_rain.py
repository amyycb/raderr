#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2011-2019, wradlib developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

"""
Attenuation Correction
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :nosignatures:
   :toctree: generated/

    {}
"""
__all__ = ['constraint_pia',
           'correct_attenuation_constrained',
           'correct_radome_attenuation_empirical']
__doc__ = __doc__.format('\n   '.join(__all__))


import numpy as np
from scipy import ndimage, interpolate
from wradlib import trafo, zr

class AttenuationOverflowError(Exception):
    pass

def _sector_filter(mask, min_sector_size):
    """Calculate an array of same shape as mask, which is set to 1 in case of \
    at least min_sector_size adjacent values, otherwise it is set to 0.
    """

    kernela = np.ones([1] * (mask.ndim - 1) + [min_sector_size])
    kernelb = np.ones((min_sector_size,))
    forward_origin = (-(min_sector_size - (min_sector_size // 2)) +
                      min_sector_size % 2)
    backward_origin = (min_sector_size - (min_sector_size // 2)) - 1
    forward_sum = ndimage.correlate1d(mask.astype(np.int), kernelb,
                                      axis=-1, mode='wrap',
                                      origin=forward_origin)
    backward_sum = ndimage.correlate1d(mask.astype(np.int), kernelb,
                                       axis=-1, mode='wrap',
                                       origin=backward_origin)
    forward_corners = (forward_sum == min_sector_size)
    backward_corners = (backward_sum == min_sector_size)
    forward_large_sectors = np.zeros_like(mask)
    backward_large_sectors = np.zeros_like(mask)
    for iii in range(mask.shape[0]):
        forward_large_sectors[iii] = ndimage.morphology.binary_dilation(
            forward_corners[iii], kernela[0], origin=forward_origin).astype(
            int)
        backward_large_sectors[iii] = ndimage.morphology.binary_dilation(
            backward_corners[iii], kernela[0],
            origin=backward_origin).astype(int)

    return (forward_large_sectors | backward_large_sectors)

def bisect_reference_attenuation(gateset, pia_ref,
                                 a_max=0.0022*1.2, a_min=0.0022*1.2,
                                 b_start=1.17, gate_length=1.0,
                                 mode='difference', thrs=0.25,
                                 max_iterations=10):
    """Find the optimal attenuation coefficients for a gateset to achieve a \
    given reference attenuation using a the forward correction algorithm in \
    combination with the bisection method.

    Parameters
    ----------
    gateset : :class:`numpy:numpy.ndarray`
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the last array-dimension.

        Data has to be provided in rainfall [mm/h].
    pia_ref : :class:`numpy:numpy.ndarray`
        Array of the same number of dimensions as ``gateset``, but the size of
        the last dimension is 1, as it constitutes the reference pia [mm/h] of
        the last range gate of every beam.
    a_max : float
        Upper bound of the bisection interval within the linear coefficient a
        of the k-R relation has to be. ( :math:`k=a \\cdot R^{b}` ).

        Per default set to0.0022*1.2.
    a_min : float
        Lower bound of the bisection interval within the linear coefficient a
        of the k-R relation has to be. ( :math:`k=a \\cdot R^{b}` ).

        Per default set to 0.0022*0.8.
    b_start : float
        Initial value for exponential coefficient of the k-R relation
        ( :math:`k=a \\cdot R^{b}` ). This value will be lowered incremental
        by 0.01 if no solution was found within the bisection interval of
        ``a_max`` and ``a_min`` within the number of given iterations
        ``max_iterations``.

        Per default set to 1.17.
    gate_length : float
        Radial length of a range gate [km].

        Per default set to 1.0.
    mode : string
        {?atio?or ?ifference?
        Kind of tolerance of calculated pia in relation to reference pia.

        Per default set to 'difference'.
    thrs : float
        Value of the tolerance to stop bisection iteration successful. It is
        recommended to choose 0.05 for ratio ``mode`` and 0.25 for difference
        ``mode``, which means a deviation tolerance of 5% or 0.25 mm/h,
        respectively.

        Per default set to 0.25.
    max_iterations : int
        Number of bisection iteration before the exponential coefficient b of
        the k-R relation will be decreased and the bisection starts again.

        Per default set to 10.

    Returns
    -------
    pia : :class:`numpy:numpy.ndarray`
        Array with the same shape as ``gateset`` containing the calculated path
        integrated attenuation [mm/h] for each range gate.
    a_mid : :class:`numpy:numpy.ndarray`
        Array with the same shape as ``pia_ref`` containing the finally used
        linear k-Z relation coefficient a for successful pia calculation.
    b : :class:`numpy:numpy.ndarray`
        Array with the same shape as ``pia_ref`` containing the finally used
        exponential k-R relation coefficient b for successful pia calculation.
    """
    # Prepare arrays of initial k-Z relation coefficients for each beam.
    a_hi = np.ones(pia_ref.shape)*a_max  # np.repeat(a_max, pia_ref.shape)
    a_lo = np.ones(pia_ref.shape)*a_min  # np.repeat(a_min, pia_ref.shape)
    b = np.ones(pia_ref.shape)*b_start  # np.repeat(b_start, pia_ref.shape)
    pia = np.empty_like(gateset)
    iteration_count = 0

    # Iterate until upper and lower bounds of linear k-Z relation coefficients
    # for pia calculation are the same.
    while not np.all(a_hi == a_lo):
        a_mid = (a_hi + a_lo) / 2
        pia = calc_attenuation_forward(gateset, a_mid, b, gate_length)
        # Find indices where calculated and reference pia sufficiently match
        if mode == 'difference':
            overshoot = (pia[..., -1] - pia_ref) > thrs
            undershoot = (pia[..., -1] - pia_ref) < -thrs
            hit = (np.abs(pia[..., -1] - pia_ref)) < thrs
        elif mode == 'ratio':
            overshoot = ((pia[..., -1] - pia_ref) / pia_ref) > thrs
            undershoot = ((pia[..., -1] - pia_ref) / pia_ref) < -thrs
            hit = (np.abs(pia[..., -1] - pia_ref) / pia_ref) < thrs
        else:
            raise Exception('Unknown mode type ' + mode + '.')
        # Define new bounds of linear k-Z relation coefficient for over- and
        # undershooting pia calculations.
        a_hi[overshoot] = a_mid[overshoot]
        a_lo[undershoot] = a_mid[undershoot]
        a_hi[hit] = a_mid[hit]
        a_lo[hit] = a_mid[hit]
        iteration_count += 1
        # Change exponential k-Z relation coefficient in case of maximum
        # iterations for linear k-Z relation coefficient are reached.
        if iteration_count > max_iterations:
            b[overshoot] -= 0.01
            b[undershoot] += 0.01
    return pia, a_mid, b


def constraint_pia(gateset, pia, thrs_pia):
    """Constraint callback function for correct_attenuation_constrained.

    Selects beams, in which the path integrated attenuation exceeds
    ``thrs_pia``.
    """
    return np.max(pia, axis=-1) > thrs_pia


def calc_attenuation_forward(gateset, a=0.0022, b=1.17, gate_length=1.):
    """Gate-by-Gate forward correction as described in :cite:`Kraemer2008` 
    modified to estimate attenuation from actual rainfall field.

    Parameters
    ----------
    gateset : :class:`numpy:numpy.ndarray`
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the last array-dimension.

        Data has to be provided in rainfall [mm/h].
    a : float
        proportionality factor of the k-R relation (:math:`k=a \\cdot R^{b}`).
        Per default set to 0.0022.
    b : float
        exponent of the k-R relation ( :math:`k=a \\cdot R^{b}` ). Per default
        set to 1.17.
    gate_length : float
        length of a range gate [km]. Per default set to 1.0.

    Returns
    -------
    pia : :class:`numpy:numpy.ndarray`
        Array with the same shape as ``gateset`` containing the calculated path
        integrated attenuation [mm/h] for each range gate.
    """
    pia = np.zeros(gateset.shape)
    for gate in range(gateset.shape[-1] - 1):
        #k = a * trafo.idecibel(gateset[..., gate] + pia[..., gate]) ** b \
        #    * 2.0 * gate_length
        #k = a * trafo.idecibel(gateset[..., gate]) ** b \
        #    * 2.0 * gate_length
        k = 2 * gate_length * a * gateset[..., gate] ** b
        pia[..., gate + 1] = pia[..., gate] + k
    return pia

def correct_attenuation_constrained(gateset, a_max=0.0022*1.2, a_min=0.0022*0.8,
                                    n_a=4, b_max=1.17+0.15, b_min=1.17-0.15, n_b=6,
                                    gate_length=1., constraints=None,
                                    constraint_args=None, sector_thr=10):
    """Gate-by-Gate attenuation correction based on the iterative approach of \
    :cite:`Kraemer2008` and :cite:`Jacobi2016` with a generalized and \
    scalable number of constraints, modified to estimate attenuation on rainfall.

    Differing from the original approach, the method for addressing
    small sectors which conflict with the constraints is based on a bisection
    forward calculating method, and not on backwards attenuation calculation.

    Parameters
    ----------
    gateset : :class:`numpy:numpy.ndarray`
        Multidimensional array, where the range gates (over which iteration has
        to be performed) are supposed to vary along the last array-dimension
        and the azimuths are supposed to vary along the next to last
        array-dimension.

        Data has to be provided in decibel representation of rainfall [mm/h].
    a_max : float
        Initial value for linear coefficient of the k-R relation
        ( :math:`k=a \\cdot R^{b}` ).

        Per default set to 0.0022*1.2 - Gunn and East (1954) claim estimates 
        could be out by 20%.
    a_min : float
        Minimal allowed linear coefficient of the k-R relation
        ( :math:`k=a \\cdot R^{b}` ) in the downwards iteration of 'a' in case
        of breaching one of thresholds ``constr_args`` of the optional
        conditions ``constraints``.

        Per default set to 0.0022*0.8 - Gunn and East (1954) claim estimates 
        could be out by 20%.
    n_a : int
        Number of iterations from ``a_max`` to ``a_min``.

        Per default set to 4.
    b_max : float
        Initial value for exponential coefficient of the k-R relation
        ( :math:`k=a \\cdot R^{b}` ).

        Per default set to 1.17+0.15.
    b_min : float
        Minimal allowed exponential coefficient of the k-R relation
        ( :math:`k=a \\cdot R^{b}` ) in the downwards iteration of 'b' in case
        of breaching one of thresholds ``constr_args`` of the optional
        conditions ``constraints`` and the linear coefficient 'a' has already
        reached the lower limit ``a_min``.

        Per default set to 1.17-0.15.
    n_b : int
        Number of iterations from ``b_max`` to ``b_min``.

        Per default set to 6.
    gate_length : float
        Radial length of a range gate [km].

        Per default set to 1.0.
    constraints : list
        List of constraint functions. The signature of these functions has to
        be constraint_function(`gateset`, `k`, `*constr_args`). Their return
        value must be a boolean array of shape gateset.shape[:-1] set to True
        for beams, which do not fulfill the constraint.
    constraint_args : list
        List of lists, which are to be passed to the individual constraint
        functions using the `*args` mechanism
        (len(constr_args) == len(constraints)).
    sector_thr : int
        Number of adjacent beams, for which in case of breaching the
        constraints the attenuation with downward iterated ``a`` and ``b`` -
        parameters is recalculated. For more narrow sectors the integrated
        attenuation of the last gate is interpolated and used as reference
        for the recalculation.

    Returns
    -------
    pia : :class:`numpy:numpy.ndarray`
        Array with the same shape as ``gateset`` containing the calculated path
        integrated attenuation [mm/h] for each range gate.
                              [[59.0],[20.0]])
    """
    if constraints is None:
        constraints = []
    if constraint_args is None:
        constraint_args = []
    n_az = gateset.shape[-2]
    n_rng = gateset.shape[-1]
    tmp_gateset = gateset.reshape((-1, n_az, n_rng))

    pia = np.zeros_like(tmp_gateset)

    a_used = np.empty(tmp_gateset.shape[:-1])
    b_used = np.empty(tmp_gateset.shape[:-1])

    # Calculate attenuation forward.
    # Indexing all rows of last dimension (radarbeams)
    beams2correct = np.where(np.ones(tmp_gateset.shape[:-1], dtype=np.bool))
    small_sectors = np.zeros(tmp_gateset.shape[:-1], dtype=np.bool)

    if n_a != 1:
        delta_a = (a_max - a_min) / (n_a - 1)
    else:
        delta_a = 0.
    if n_b != 1:
        delta_b = (b_max - b_min) / (n_b - 1)
    else:
        delta_b = 0.

    # Iterate over possible b-parameters
    for j in range(n_b):
        b = b_max - delta_b * j
        # Iterate over possible a-parameters
        for i in range(n_a):
            a = a_max - delta_a * i
            # Generate subset of beams that have to be corrected
            sub_gateset = tmp_gateset[beams2correct]
            sub_pia = calc_attenuation_forward(sub_gateset, a, b, gate_length)
            pia[beams2correct] = sub_pia
            a_used[beams2correct] = a
            b_used[beams2correct] = b
            # Indexing threshold exceeding beams
            incorrectbeams = np.zeros(tmp_gateset.shape[:-1], dtype=np.bool)
            for constraint, constraint_arg in zip(constraints,
                                                  constraint_args):
                incorrectbeams = np.logical_or(incorrectbeams,
                                               constraint(tmp_gateset, pia,
                                                          *constraint_arg))
            # Determine incorrect sectors larger than sector_thr
            large_sectors = _sector_filter(incorrectbeams, sector_thr)
            # Determine incorrect sectors smaller than sector_thr
            small_sectors = np.logical_or(small_sectors,
                                          (incorrectbeams & ~large_sectors))
            beams2correct = np.where(large_sectors)
            if len(pia[beams2correct]) == 0:
                break
        if len(pia[beams2correct]) == 0:
            break
    if np.any(small_sectors):
        # Interpolate reference pia of most distant
        # rangebin of invalid sectors.
        _interp_atten(pia, small_sectors)
        # Calculate attenuation forward by achieving reference
        # attenuation based on bisection-method.
        tmp_pia, tmp_a, tmp_b = bisect_reference_attenuation(
            tmp_gateset[small_sectors, :],
            pia[small_sectors, -1],
            a_max=a_max,
            a_min=a_min,
            b_start=b_max,
            gate_length=gate_length,
            mode='difference',
            thrs=0.25,
            max_iterations=10)
        pia[small_sectors, :] = tmp_pia
        a_used[small_sectors] = tmp_a
        b_used[small_sectors] = tmp_b

    return pia.reshape(gateset.shape)