import CIexpansion, GivensRotationsUtil, jax
import numpy as np
import jax.numpy as jnp
from itertools import combinations
from jax import vmap
from jax import config

config.update("jax_enable_x64", True)


##the factor for the diffusion, drift and braching terms
def makePDEmatrix_up(factora, factorb, factorc, Alldets, shapeTuple, idx, H, dpdx, d2pdx2, deltap, Ndiscrete, n_sites, n_elec, combos, comboToTheta, ref):
    nidx = idx.shape[0]
    D2PsiDp2_idx   = np.zeros((Alldets.shape[0], nidx*3 + 4 * (nidx)*(nidx-1)//2 + 1), dtype=np.int32)
    D2PsiDp2_Coeff = np.zeros((Alldets.shape[0], nidx*3 + 4 * (nidx)*(nidx-1)//2 + 1))

    for I in range(Alldets.shape[0]):
        Index = list(np.unravel_index(I, shapeTuple))

        #print(Index)
        Index2 = Index.copy()
        for k in range(nidx):
            #Index2[k] = (Index[k]+1)%Ndiscrete
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(I, k, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            D2PsiDp2_idx  [I, 3*k  ]   = np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k  ]   = sign * (factora * H[I, k, k]/deltap**2 + factorb * dpdx[I, k]/deltap/2.)
            #print(Index2, sign)

            #Index2[k] = (Index[k]-1)%Ndiscrete
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(I, k, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            D2PsiDp2_idx  [I, 3*k+1]   = np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+1]   = sign * (factora * H[I, k, k]/deltap**2 - factorb * dpdx[I, k]/deltap/2.)
            #print(Index2, sign)

            #Index2[k] = (Index[k])
            D2PsiDp2_idx  [I, 3*k+2]   = I #np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+2]   = -factora * 2*H[I, k, k]/deltap**2
            #print(Index2, sign)

        IDX = 0
        for k in range(nidx):
            for l in range(k+1, nidx):

                Index2 = Index.copy()
                #Index2[k] = (Index[k]+1)%Ndiscrete
                #Index2[l] = (Index[l]+1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(I, k, l, 1, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4    ]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4    ]   = sign * factora *  H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                #Index2[k] = (Index[k]+1)%Ndiscrete
                #Index2[l] = (Index[l]-1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(I, k, l, 1, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +1]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +1]   = sign * factora * -H[I, k, l]/deltap**2/2. 
                #print(Index2, sign)

                #Index2[k] = (Index[k]-1)%Ndiscrete
                #Index2[l] = (Index[l]+1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(I, k, l, -1, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +2]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +2]   = sign * factora * -H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                #Index2[k] = (Index[k]-1)%Ndiscrete
                #Index2[l] = (Index[l]-1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(I, k, l, -1, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +3]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +3]   = sign * factora *  H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                Index2[k] = (Index[k])
                Index2[l] = (Index[l])
                IDX += 1

        ##the diagonal term
        D2PsiDp2_idx  [I, 3*nidx + IDX*4 ] = I 
        D2PsiDp2_Coeff[I, 3*nidx + IDX*4 ] = factorc * d2pdx2[I, 0]
        #exit(0)
    return D2PsiDp2_idx, D2PsiDp2_Coeff



##the factor for the diffusion, drift and braching terms
def makePDEmatrix_down(factora, factorb, factorc, Alldets, shapeTuple, idx, H, dpdx, d2pdx2, deltap, Ndiscrete, n_sites, n_elec, combos, comboToTheta, ref):
    nidx = idx.shape[0]
    D2PsiDp2_idx   = np.zeros((Alldets.shape[0], nidx*3 + 4 * (nidx)*(nidx-1)//2 + 1), dtype=np.int32)
    D2PsiDp2_Coeff = np.zeros((Alldets.shape[0], nidx*3 + 4 * (nidx)*(nidx-1)//2 + 1))

    for I in range(Alldets.shape[0]):
        Index = list(np.unravel_index(I, shapeTuple))

        #print(Index)
        Index2 = Index.copy()
        for k in range(nidx):
            #Index2[k] = (Index[k]+1)%Ndiscrete
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(I, k, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            D2PsiDp2_idx  [I, 3*k  ]   = np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k  ]   = sign * (factora * H[I, k, k]/deltap**2 + factorb * dpdx[I, k]/deltap/2.)
            #print(Index2, sign)

            #Index2[k] = (Index[k]-1)%Ndiscrete
            Index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(I, k, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
            D2PsiDp2_idx  [I, 3*k+1]   = np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+1]   = sign * (factora * H[I, k, k]/deltap**2 - factorb * dpdx[I, k]/deltap/2.)
            #print(Index2, sign)

            #Index2[k] = (Index[k])
            D2PsiDp2_idx  [I, 3*k+2]   = I #np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+2]   = -factora * 2*H[I, k, k]/deltap**2
            #print(Index2, sign)

        IDX = 0
        for k in range(nidx):
            for l in range(k+1, nidx):

                Index2 = Index.copy()
                #Index2[k] = (Index[k]+1)%Ndiscrete
                #Index2[l] = (Index[l]+1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(I, k, l, 1, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4    ]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4    ]   = sign * factora *  H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                #Index2[k] = (Index[k]+1)%Ndiscrete
                #Index2[l] = (Index[l]-1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(I, k, l, 1, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +1]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +1]   = sign * factora * -H[I, k, l]/deltap**2/2. 
                #print(Index2, sign)

                #Index2[k] = (Index[k]-1)%Ndiscrete
                #Index2[l] = (Index[l]+1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(I, k, l, -1, 1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +2]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +2]   = sign * factora * -H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                #Index2[k] = (Index[k]-1)%Ndiscrete
                #Index2[l] = (Index[l]-1)%Ndiscrete
                Index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(I, k, l, -1, -1, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref)
                D2PsiDp2_idx  [I, 3*nidx + IDX*4  +3]   = np.ravel_multi_index(Index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*nidx + IDX*4  +3]   = sign * factora *  H[I, k, l]/deltap**2/2.
                #print(Index2, sign)

                Index2[k] = (Index[k])
                Index2[l] = (Index[l])
                IDX += 1

        ##the diagonal term
        D2PsiDp2_idx  [I, 3*nidx + IDX*4 ] = I 
        D2PsiDp2_Coeff[I, 3*nidx + IDX*4 ] = factorc * d2pdx2[I, 0]
        #exit(0)
    return D2PsiDp2_idx, D2PsiDp2_Coeff




def makeExtendedPDEmatrix(
        factora, factorb, factorc, factord, factore, factorf, factorg, factorh, factori, factorj,
        Alldets_up, Alldets_down,
        shapeTuple, # full (p+q) grid shape
        idx_up, idx_down,
        H_up, H_down,
        dpdx_up, dpdx_down,
        d2pdx2_up, d2pdx2_down,
        deltap, Ndiscrete, 
        n_sites, n_elec, 
        combos_up, combos_down,
        comboToTheta,
        ref_up, ref_down,
        dp1up_dwdown, dp1down_dwup, dp1up_dp1down, dwup_dwdown
    ):
    n_p = idx_up.shape[0]
    n_q = idx_down.shape[0]
    nidx = n_p + n_q

    shape_p = shapeTuple[:n_p]
    shape_q = shapeTuple[n_p:]

    size_p = np.prod(shape_p)
    size_q = np.prod(shape_q)
    n_states = size_p * size_q

    D2PsiDp2_idx   = np.zeros((n_states, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + 4 * (n_p*n_q) + 1), dtype=np.int32)
    D2PsiDp2_Coeff = np.zeros((n_states, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + 4 * (n_p*n_q) + 1))

    for I in range(n_states):
        # Unravel full index to get p and q indices
        full_idx = np.unravel_index(I, shapeTuple)
        p_idx = full_idx[:n_p]
        q_idx = full_idx[n_p:]

        # Convert to linear indices for sub-grids
        I_p = np.ravel_multi_index(p_idx, shape_p)
        I_q = np.ravel_multi_index(q_idx, shape_q)

        # --- p‐loop for alpha spin ---
        for k in range(n_p):
            p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(
                I_p, k, +1, deltap, Ndiscrete,
                Alldets_up, shape_p, idx_up, 
                comboToTheta, n_sites, n_elec, combos_up, ref_up
            )
            full_index2 = list(p_index2) + list(q_idx)
            D2PsiDp2_idx  [I, 3*k  ]   = np.ravel_multi_index(full_index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k  ]   = sign * (factora * H_up[I_p, k, k]/deltap**2 + factorb * (dpdx_up[I_p, k] + factori*dp1up_dwdown[I_p, I_q, k])/deltap/2.)


            p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_up(
                I_p, k, -1, deltap, Ndiscrete, 
                Alldets_up, shape_p, idx_up,
                comboToTheta, n_sites, n_elec, combos_up, ref_up
            )
            full_index2 = list(p_index2) + list(q_idx)
            D2PsiDp2_idx  [I, 3*k+1]   = np.ravel_multi_index(full_index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+1]   = sign * (factora * H_up[I_p, k, k]/deltap**2 - factorb * (dpdx_up[I_p, k] + factori*dp1up_dwdown[I_p, I_q, k])/deltap/2.)


            D2PsiDp2_idx  [I, 3*k+2]   = I #np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*k+2]   = -factora * 2*H_up[I_p, k, k]/deltap**2
            #print(Index2, sign)

        IDX = 0
        for k in range(n_p):
            for l in range(k+1, n_p):

                p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(
                    I_p, k, l, 1, 1, deltap, Ndiscrete, 
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                full_index2 = list(p_index2) + list(q_idx)
                D2PsiDp2_idx  [I, 3*n_p + IDX*4    ]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*n_p + IDX*4    ]   = sign * factora *  H_up[I_p, k, l]/deltap**2/2.


                p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(
                    I_p, k, l, 1, -1, deltap, Ndiscrete, 
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                full_index2 = list(p_index2) + list(q_idx)
                D2PsiDp2_idx  [I, 3*n_p + IDX*4  +1]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*n_p + IDX*4  +1]   = sign * factora * -H_up[I_p, k, l]/deltap**2/2. 

            
                p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(
                    I_p, k, l, -1, 1, deltap, Ndiscrete,
                    Alldets_up, shape_p, idx_up,
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                full_index2 = list(p_index2) + list(q_idx)
                D2PsiDp2_idx  [I, 3*n_p + IDX*4  +2]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*n_p + IDX*4  +2]   = sign * factora * -H_up[I_p, k, l]/deltap**2/2.

                
                p_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_up(
                    I_p, k, l, -1, -1, deltap, Ndiscrete, 
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                full_index2 = list(p_index2) + list(q_idx)
                D2PsiDp2_idx  [I, 3*n_p + IDX*4  + 3]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, 3*n_p + IDX*4  + 3]   = sign * factora *  H_up[I_p, k, l]/deltap**2/2.

                IDX += 1

        # --- q‐loop for beta spin ---
        for k in range(n_q):
            q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(
                I_q, k, +1, deltap, Ndiscrete,
                Alldets_down, shape_q, idx_down, 
                comboToTheta, n_sites, n_elec, combos_down, ref_down
            )
            full_index2 = list(p_idx) + list(q_index2)
            D2PsiDp2_idx  [I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2]   = np.ravel_multi_index(full_index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2 ]   = sign * (factord * H_down[I_q, k, k]/deltap**2 + factore * (dpdx_down[I_q, k] + factorj*dp1down_dwup[I_q, I_p, k])/deltap/2.)


            q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_down(
                I_q, k, -1, deltap, Ndiscrete, 
                Alldets_down, shape_q, idx_down,
                comboToTheta, n_sites, n_elec, combos_down, ref_down
            )
            full_index2 = list(p_idx) + list(q_index2)
            D2PsiDp2_idx  [I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2 + 1]   = np.ravel_multi_index(full_index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2 + 1]   = sign * (factord * H_down[I_q, k, k]/deltap**2 - factore * (dpdx_down[I_q, k] + factorj*dp1down_dwup[I_q, I_p, k])/deltap/2.)


            D2PsiDp2_idx  [I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2 + 2]   = I #np.ravel_multi_index(Index2, shapeTuple)
            D2PsiDp2_Coeff[I, 3*(n_p + k) + 4 * (n_p)*(n_p-1)//2 + 2]   = -factord * 2*H_down[I_q, k, k]/deltap**2
            #print(Index2, sign)

        IDX = 0
        for k in range(n_q):
            for l in range(k+1, n_q):

                q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(
                    I_q, k, l, 1, 1, deltap, Ndiscrete, 
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                full_index2 = list(p_idx) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4    ]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4    ]   = sign * factord *  H_down[I_q, k, l]/deltap**2/2.


                q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(
                    I_q, k, l, 1, -1, deltap, Ndiscrete, 
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                full_index2 = list(p_idx) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4 + 1]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4 + 1]   = sign * factord * -H_down[I_q, k, l]/deltap**2/2. 

            
                q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(
                    I_q, k, l, -1, 1, deltap, Ndiscrete,
                    Alldets_down, shape_q, idx_down,
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                full_index2 = list(p_idx) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4  +2]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4  +2]   = sign * factord * -H_down[I_q, k, l]/deltap**2/2.

                
                q_index2, sign = GivensRotationsUtil.nextMappedDeterminant_2_down(
                    I_q, k, l, -1, -1, deltap, Ndiscrete, 
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                full_index2 = list(p_idx) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4  +3]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + IDX*4  +3]   = sign * factord *  H_down[I_q, k, l]/deltap**2/2.

                IDX += 1

        # --- MIXED SECOND DERIVATIVE TERM ---
        IDX = 0
        for k in range(n_p):
            for l in range(n_q):

                p_index2, sign_p = GivensRotationsUtil.nextMappedDeterminant_up(
                    I_p, k, +1, deltap, Ndiscrete,
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                q_index2, sign_q = GivensRotationsUtil.nextMappedDeterminant_down(
                    I_q, l, +1, deltap, Ndiscrete,
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                sign = sign_p * sign_q
                full_index2 = list(p_index2) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4    ]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4    ]   = sign * factorg *  dp1up_dp1down[I_p, I_q, k, l]/deltap**2/4.



                p_index2, sign_p = GivensRotationsUtil.nextMappedDeterminant_up(
                    I_p, k, +1, deltap, Ndiscrete,
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                q_index2, sign_q = GivensRotationsUtil.nextMappedDeterminant_down(
                    I_q, l, -1, deltap, Ndiscrete,
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                sign = sign_p * sign_q
                full_index2 = list(p_index2) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 1]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 1]   = sign * factorg * -dp1up_dp1down[I_p, I_q, k, l]/deltap**2/4. 

            
                p_index2, sign_p = GivensRotationsUtil.nextMappedDeterminant_up(
                    I_p, k, -1, deltap, Ndiscrete,
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                q_index2, sign_q = GivensRotationsUtil.nextMappedDeterminant_down(
                    I_q, l, +1, deltap, Ndiscrete,
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                sign = sign_p * sign_q
                full_index2 = list(p_index2) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 2]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 2]   = sign * factorg * -dp1up_dp1down[I_p, I_q, k, l]/deltap**2/4.

                
                p_index2, sign_p = GivensRotationsUtil.nextMappedDeterminant_up(
                    I_p, k, -1, deltap, Ndiscrete,
                    Alldets_up, shape_p, idx_up, 
                    comboToTheta, n_sites, n_elec, combos_up, ref_up
                )
                q_index2, sign_q = GivensRotationsUtil.nextMappedDeterminant_down(
                    I_q, l, -1, deltap, Ndiscrete,
                    Alldets_down, shape_q, idx_down, 
                    comboToTheta, n_sites, n_elec, combos_down, ref_down
                )
                sign = sign_p * sign_q
                full_index2 = list(p_index2) + list(q_index2)
                D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 3]   = np.ravel_multi_index(full_index2, shapeTuple)
                D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 + 3]   = sign * factorg * dp1up_dp1down[I_p, I_q, k, l]/deltap**2/4.

                IDX += 1

        ##the diagonal term
        D2PsiDp2_idx  [I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 ] = I 
        D2PsiDp2_Coeff[I, nidx*3 + 4 * (n_p)*(n_p-1)//2 + 4 * (n_q)*(n_q-1)//2 + IDX*4 ] = factorh * dwup_dwdown[I_p,I_q] + factorc * d2pdx2_up[I_p, 0] + factorf * d2pdx2_down[I_q, 0]
        # exit(0)
    return D2PsiDp2_idx, D2PsiDp2_Coeff
