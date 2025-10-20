import CIexpansion, GivensRotationsUtil, jax
import numpy as np
import jax.numpy as jnp
from itertools import combinations
from jax import vmap
from jax import config
from matplotlib import pyplot as plt
from numba import njit, prange
from jax import vmap, jit, numpy as jnp, scipy as jsp, random, lax, scipy as jsp
from functools import partial


config.update("jax_enable_x64", True)

def makePDEmatrix_up(factora, factorb, shapeTuple, idx, dp1, deltap, Ndiscrete, Alldets, comboToTheta, n_sites, n_elec, combos, ref):
    nidx = idx.shape[0]
    DPsiDp_idx   = np.zeros((dp1.shape[0], 2*nidx+1), dtype=np.int32)
    DPsiDp_Coeff = np.zeros((dp1.shape[0], 2*nidx+1))


    for I in range(dp1.shape[0]):
        Index = list(np.unravel_index(I, shapeTuple))
        
        Index2 = Index.copy()
        for k in range(nidx):
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(I, k, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            DPsiDp_idx  [I, 2*k  ]   = np.ravel_multi_index(Index2, shapeTuple)
            DPsiDp_Coeff[I, 2*k  ]   = sign * factora * dp1[I, k+1]/deltap/2.

            #Index2 = Index.copy()
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(I, k, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            DPsiDp_idx  [I, 2*k+1]   = np.ravel_multi_index(Index2, shapeTuple)
            # DPsiDp_Coeff[I, 2*k+1]   = factora * -dp1[I, k+1]/deltap
            DPsiDp_Coeff[I, 2*k+1  ]   = sign * factora * -dp1[I, k+1]/(deltap)/2.


            Index2[k] = (Index[k])

        DPsiDp_idx  [I, 2*nidx] = I
        DPsiDp_Coeff[I, 2*nidx] = factorb * dp1[I, 0]

    return DPsiDp_idx, DPsiDp_Coeff


def makePDEmatrix_down(factora, factorb, shapeTuple, idx, dp1, deltap, Ndiscrete, Alldets, comboToTheta, n_sites, n_elec, combos, ref):
    nidx = idx.shape[0]
    DPsiDp_idx   = np.zeros((dp1.shape[0], 2*nidx+1), dtype=np.int32)
    DPsiDp_Coeff = np.zeros((dp1.shape[0], 2*nidx+1))


    for I in range(dp1.shape[0]):
        Index = list(np.unravel_index(I, shapeTuple))
        
        Index2 = Index.copy()
        for k in range(nidx):
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(I, k, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            DPsiDp_idx  [I, 2*k  ]   = np.ravel_multi_index(Index2, shapeTuple)
            DPsiDp_Coeff[I, 2*k  ]   = sign * factora * dp1[I, k+1]/deltap/2.

            #Index2 = Index.copy()
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(I, k, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            DPsiDp_idx  [I, 2*k+1]   = np.ravel_multi_index(Index2, shapeTuple)
            # DPsiDp_Coeff[I, 2*k+1]   = factora * -dp1[I, k+1]/deltap
            DPsiDp_Coeff[I, 2*k+1  ]   = sign * factora * -dp1[I, k+1]/(deltap)/2.


            Index2[k] = (Index[k])

        DPsiDp_idx  [I, 2*nidx] = I
        DPsiDp_Coeff[I, 2*nidx] = factorb * dp1[I, 0]

    return DPsiDp_idx, DPsiDp_Coeff



def makeExtendedPDEmatrix(
    factora, factorb,
    shapeTuple,   # full (p+q) grid shape
    idx_up, idx_down,
    dp1, dq1,     # shape (size_p, n_p) and (size_q, n_q)
    deltap, Ndiscrete,
    Alldets_up, Alldets_down,
    comboToTheta, n_sites, n_elec, 
    combos_up, combos_down,
    ref_up, ref_down,
):
    n_p = idx_up.shape[0]
    n_q = idx_down.shape[0]
    nidx = n_p + n_q

    shape_p = shapeTuple[:n_p]
    shape_q = shapeTuple[n_p:]

    size_p = np.prod(shape_p)
    size_q = np.prod(shape_q)
    n_states = size_p * size_q

    DPsiDpq_idx   = np.zeros((n_states, 2*nidx+1), dtype=np.int32)
    DPsiDpq_coeff = np.zeros((n_states, 2*nidx+1))

    # for I in range(n_states):
    for I in range(n_states):
        # Unravel full index to get p and q indices
        full_idx = np.unravel_index(I, shapeTuple)
        p_idx = full_idx[:n_p]
        q_idx = full_idx[n_p:]


        # Convert to linear indices for sub-grids
        I_p = np.ravel_multi_index(p_idx, shape_p)
        I_q = np.ravel_multi_index(q_idx, shape_q)

        # --- p‐loop uses up‐sector data ---
        for k in range(n_p):
            # forward in p
            p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(
                I_p, k, +1, deltap, Ndiscrete,
                Alldets_up, shape_p, idx_up, 
                comboToTheta, n_sites, n_elec, combos_up, ref_up
            )
            full_index2 = list(p_index2) + list(q_idx)
            DPsiDpq_idx  [I, 2*k  ] = np.ravel_multi_index(full_index2, shapeTuple)
            DPsiDpq_coeff[I, 2*k  ] = sign * factora * dp1[I_p, k+1]/(2*deltap)

            # backward in p
            p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(
                I_p, k, -1, deltap, Ndiscrete,
                Alldets_up, shape_p, idx_up,
                comboToTheta, n_sites, n_elec, combos_up, ref_up
            )
            full_index2 = list(p_index2) + list(q_idx)
            DPsiDpq_idx  [I, 2*k+1] = np.ravel_multi_index(full_index2, shapeTuple)
            DPsiDpq_coeff[I, 2*k+1] = sign * factora * -dp1[I_p, k+1]/(2*deltap)

        # --- q‐loop uses down‐sector data ---
        for k in range(n_q):
            # forward in q
            q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(
                I_q, k, +1, deltap, Ndiscrete,
                Alldets_down, shape_q, idx_down,
                comboToTheta, n_sites, n_elec, combos_down, ref_down
            )
            full_index2 = list(p_idx) + list(q_index2)
            DPsiDpq_idx  [I, 2*(n_p+k)  ] = np.ravel_multi_index(full_index2, shapeTuple)
            DPsiDpq_coeff[I, 2*(n_p+k)  ] = sign * factora * dq1[I_q, k+1]/(2*deltap)

            # backward in q
            q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(
                I_q, k, -1, deltap, Ndiscrete,
                Alldets_down, shape_q, idx_down,
                comboToTheta, n_sites, n_elec, combos_down, ref_down
            )
            full_index2 = list(p_idx) + list(q_index2)
            DPsiDpq_idx  [I, 2*(n_p+k)+1] = np.ravel_multi_index(full_index2, shapeTuple)
            DPsiDpq_coeff[I, 2*(n_p+k)+1] = sign * factora * -dq1[I_q, k+1]/(2*deltap)

        # --- diagonal weight term ---
        DPsiDpq_idx  [I, 2*nidx] = I
        DPsiDpq_coeff[I, 2*nidx] = factorb * (dp1[I_p,0] + dq1[I_q,0])

    return DPsiDpq_idx, DPsiDpq_coeff

