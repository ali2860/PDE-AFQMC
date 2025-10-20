import Op_1e, Op_2e, GivensRotationsUtil, CIexpansion
import jax.numpy as jnp
from importlib import reload
from itertools import combinations
import jax
from jax import vmap
from jax import config
import numpy as np
from jax import vmap, jit, numpy as jnp, scipy as jsp, random, lax, scipy as jsp
import random
import matplotlib.pyplot as plt
from functools import partial
from numba import njit
import os

reload(GivensRotationsUtil)
reload(CIexpansion)
reload(Op_1e)
reload(Op_2e)

config.update("jax_enable_x64", True)

@njit
def ActH(idx, coeffs, Psi):
    Psiout = 0. * Psi
    for i in range(idx.shape[0]):
        for j in range(idx.shape[1]):
            Psiout[idx[i, j]] += coeffs[i, j] * Psi[i]
    return Psiout




n_sites = 3
n_elec = (0, 1)
epsilon = 1.e-6
ci_basis = jnp.array(CIexpansion.make_basis(n_sites, n_elec), dtype=jnp.int32)

##make the hamiltonian which is u and hopping term
u = 3
np.random.seed(0) # THIS IS T_MATRIX1
# h = np.random.normal(size=(n_sites, n_sites))
# t_matrix = h + h.T 

t_matrix = jnp.array([[-0.6680644, 0.9605467, 0.520678],

[ 0.9605467, -0.1900202, 0.3938467],

[ 0.520678 , 0.3938467 , 0.0340779]]) # THIS IS T_MATRIX2

h1 = t_matrix

# h1 = jnp.array([[-0.6680644, 0.9605467, 0.520678, -0.1044906],

# [ 0.9605467, -0.1900202, 0.3938467, -1.0511902],

# [ 0.520678 , 0.3938467 , 0.0340779, -0.4899371],

# [-0.1044906, -1.0511902, -0.4899371, 0.0882198]])

# h1 = jnp.array([[0,-5],[-5,-5]])

# t_matrix = h1

ham_element = partial(CIexpansion.ham_element_hubbard_general, u=u, t_matrix=h1)
ham_mat = vmap(vmap(ham_element, (None, 0)), (0, None))(ci_basis, ci_basis)
print("built hamiltonian")
ene_diag, evec_diag = jnp.linalg.eigh(ham_mat)
evec_diag *= np.sign(evec_diag[0, 0])
print(f"Exact energies: ", ene_diag)

ham_element_h1 = partial(CIexpansion.ham_element_hubbard_general, u=0.0, t_matrix=h1)
h1_mat = vmap(vmap(ham_element_h1, (None, 0)), (0, None))(ci_basis, ci_basis)
ham_element_u = partial(CIexpansion.ham_element_hubbard_general, u=u, t_matrix=0.0 * h1)
u_mat = vmap(vmap(ham_element_u, (None, 0)), (0, None))(ci_basis, ci_basis)
# HF States
h1_ene, h1_evec = jnp.linalg.eigh(h1)
hf_u = h1_evec[:, :n_elec[0]]
hf_d = h1_evec[:, :n_elec[1]]
hf = [hf_u, hf_d]


AllCIdets_up = np.zeros((ci_basis.shape[0], n_sites, n_elec[0]))
AllCIdets_down = np.zeros((ci_basis.shape[0], n_sites, n_elec[1]))
for i in range(ci_basis.shape[0]):
    # — Up spin —
    e_up = 0
    for o in range(ci_basis[i].shape[1]):
        if ci_basis[i, 0, o] > 1e-5:
            AllCIdets_up[i, o, e_up] = 1.0
            e_up += 1

    # — Down spin (exactly the same, but using index 1) —
    e_dn = 0
    for o in range(ci_basis[i].shape[1]):
        if ci_basis[i, 1, o] > 1e-5:
            AllCIdets_down[i, o, e_dn] = 1.0
            e_dn += 1

idx_init = list(combinations(range(n_sites), 2))
nonRedundant_up = n_sites * (n_sites - 1) // 2 - (n_sites - n_elec[0]) * (n_sites - n_elec[0] - 1) // 2
idx_up = jnp.array(idx_init[:nonRedundant_up])
nonRedundant_down = n_sites * (n_sites - 1) // 2 - (n_sites - n_elec[1]) * (n_sites - n_elec[1] - 1) // 2
idx_down = jnp.array(idx_init[:nonRedundant_down])

if n_elec[0] != 0:
    R_hf_up, theta_hf_up =GivensRotationsUtil.givens_qr_decomposition_jax(hf_u,idx_up,np.eye(n_sites))
    Q_hf_up = GivensRotationsUtil.given_to_Q(n_sites,theta_hf_up,idx_up,np.eye(n_sites))
    Q_hf_up = Q_hf_up.at[:, :n_elec[0]].multiply(np.sign(np.diag(R_hf_up)))
    ref_state_up = Q_hf_up
    idx_up = jnp.array([[x,y] for x in range(n_elec[0]) for y in range(n_elec[0],n_sites)])



if n_elec[1] != 0:
    R_hf_down, theta_hf_down =GivensRotationsUtil.givens_qr_decomposition_jax(hf_d,idx_down,np.eye(n_sites))
    Q_hf_down = GivensRotationsUtil.given_to_Q(n_sites,theta_hf_down,idx_down,np.eye(n_sites))
    Q_hf_down = Q_hf_down.at[:, :n_elec[1]].multiply(np.sign(np.diag(R_hf_down)))
    ref_state_down = Q_hf_down
    idx_down = jnp.array([[x,y] for x in range(n_elec[1]) for y in range(n_elec[1],n_sites)])


def comboToTheta(combos, deltap):
    return combos * deltap - np.pi/2. + deltap/2.


# overlap and energy functions for HF and exact ground state
np.random.seed(123)
h1_ene, h1_evec = jnp.linalg.eigh(h1)
hf_u_noisy = h1_evec[:, :n_elec[0]]
hf_d_noisy = h1_evec[:, :n_elec[1]]
# noise_up = np.random.uniform(-0.5, 0.5, hf_u.shape)
# noise_up = np.random.uniform(-1.9, 1.9, hf_u_noisy.shape)
# noise_up = np.random.uniform(-1.3, 1.3, hf_u_noisy.shape)

if n_elec[0] != 0 and n_elec[1] == 0:
    noise = np.random.uniform(-3, 3, hf_u.shape)
    hf_u_noisy += noise
elif n_elec[0] == 0 and n_elec[1] != 0:
    noise = np.random.uniform(-3, 3, hf_d.shape)
    hf_d_noisy += noise
hf_noisy = [hf_u_noisy, hf_d_noisy]


ene_0 = ene_diag[0]


@jit
def _calc_overlap(lu, ld, ru, rd):
    return jnp.linalg.det(lu.conj().T @ ru) * jnp.linalg.det(ld.conj().T @ rd)


@jit
def _calc_green(lu, ld, ru, rd):
    gu = ru @ jnp.linalg.inv(lu.conj().T @ ru) @ lu.conj().T
    gd = rd @ jnp.linalg.inv(ld.conj().T @ rd) @ ld.conj().T
    return gu, gd


@jit
def _calc_energy(lu, ld, ru, rd, h1, u):
    gu, gd = _calc_green(lu, ld, ru, rd)
    energy_1 = jnp.sum(gu * h1) + jnp.sum(gd * h1)
    energy_2 = u * jnp.sum(gu.diagonal() * gd.diagonal())
    return energy_1 + energy_2


@jit
def calc_overlap_hf(ru, rd):
    return _calc_overlap(hf_u_noisy, hf_d_noisy, ru, rd)


@jit
def calc_energy_hf(ru, rd, u):
    return _calc_energy(hf_u_noisy, hf_d_noisy, ru, rd, h1, u)

    
@jit
def calc_hilbert_state(ru, rd):
    def _calc_coeff(occ_nums):
        occ_u, occ_d = occ_nums
        occ_up_ind = jnp.where(occ_u, size=n_elec[0])[0]
        occ_dn_ind = jnp.where(occ_d, size=n_elec[1])[0]
        det_up = jnp.linalg.det(ru[occ_up_ind, :])
        det_dn = jnp.linalg.det(rd[occ_dn_ind, :])
        return det_up * det_dn

    return vmap(_calc_coeff)(ci_basis)


hf_vec = calc_hilbert_state(hf_u_noisy, hf_d_noisy)
hf_vec = hf_vec/jnp.linalg.norm(hf_vec)


@jit
def calc_hilbert_state_det_sum(walkers, weights):
    walkers_u, walkers_d = walkers
    return jnp.sum(
        weights[:, None] * vmap(calc_hilbert_state)(walkers_u, walkers_d), axis=0
    )


exact_ground_state = evec_diag[:, 0]


@jit
def calc_overlap_exact(ru, rd):
    slater_hilbert = calc_hilbert_state(ru, rd)
    # slater_hilbert = slater_hilbert/jnp.linalg.norm(slater_hilbert)
    overlap = jnp.dot(exact_ground_state, slater_hilbert)
    return overlap

##discretized space
Ndiscrete = 15 ##by increasing this we can get closer to the exact result

if n_elec[0] != 0 and n_elec[1] == 0:
    shapeTuple_up = np.asarray((Ndiscrete,)*idx_up.shape[0])
    combos_up = jnp.array(vmap(jnp.unravel_index, (0, None))(jnp.arange(Ndiscrete**idx_up.shape[0]), shapeTuple_up)).T
    deltap = np.pi / (Ndiscrete)
    G_mats_up = comboToTheta(combos_up, deltap)
    
    ##make all determinants in the space
    Alldets = vmap(GivensRotationsUtil.given_to_Q, (None, 0, None, None))(n_sites, G_mats_up, idx_up, ref_state_up)
    Alldets_up = Alldets[:,:,:n_elec[0]]
    Alldets_down = Alldets[:,:,:n_elec[1]]
    is_boundary_coord = (combos_up == 0) | (combos_up == Ndiscrete - 1)
    is_boundary_point = jnp.any(is_boundary_coord, axis=1)
    boundary_combos = combos_up[is_boundary_point]
    interior_combos = combos_up[~is_boundary_point]

    boundary_linear = np.ravel_multi_index(boundary_combos.T, shapeTuple_up)
    interior_linear = np.ravel_multi_index(interior_combos.T, shapeTuple_up)

    dp1_up = vmap(jax.jacfwd(lambda tau, det : GivensRotationsUtil.applyh_1e_ov(tau, t_matrix, det, idx_up, ref_state_up, n_elec[0]), argnums=0), (None, 0))(epsilon, Alldets_up)

    DPsiDp_idx_up, DPsiDp_Coeff_up = Op_1e.makePDEmatrix_up(1.0, 1.0, shapeTuple_up, idx_up, dp1_up,deltap, Ndiscrete, Alldets_up, comboToTheta, n_sites, n_elec, combos_up, ref_state_up)

    fields_up = 1*np.eye(n_sites)
    dp1_up = vmap(vmap(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_up, ref_state_up, n_elec[0]), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_up, fields_up)
    dp2_up = vmap(vmap(jax.jacfwd(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_up, ref_state_up, n_elec[0]), argnums=0), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_up, fields_up)

    d2pdx2_up = jnp.einsum('ixj->ij', dp2_up) 
    dpdx_up   = d2pdx2_up[:,1:]/2. + jnp.einsum('ixj,ix->ij', dp1_up[:,:,1:], dp1_up[:,:,0]) 
    H_up = jnp.einsum('ixj, ixk->ijk', dp1_up[:,:,1:], dp1_up[:,:,1:])

    factora, factorb, factorc = -0.5, -1., -0.5
    DPsiDp_idx_2e_up, DPsiDp_Coeff_2e_up = Op_2e.makePDEmatrix_up(factora, factorb, factorc, Alldets_up, shapeTuple_up, idx_up, H_up, dpdx_up, d2pdx2_up, deltap, Ndiscrete, n_sites, n_elec, combos_up,  comboToTheta, ref_state_up)
    acth = lambda Psi: ActH(DPsiDp_idx_up, DPsiDp_Coeff_up, Psi) + ActH(DPsiDp_idx_2e_up, DPsiDp_Coeff_2e_up, Psi)

    hf_sr_cp_hs_energies = jax.vmap(calc_energy_hf, in_axes=(0, 0, None))(
    Alldets_up, Alldets_down, u
    )

    hf_sr_cp_hs_overlaps = jax.vmap(calc_overlap_hf)(Alldets_up, Alldets_down)


if n_elec[1] != 0 and n_elec[0] == 0:
    shapeTuple_down = np.asarray((Ndiscrete,)*idx_down.shape[0])
    combos_down = jnp.array(vmap(jnp.unravel_index, (0, None))(jnp.arange(Ndiscrete**idx_down.shape[0]), shapeTuple_down)).T
    deltap = np.pi / (Ndiscrete)
    G_mats_down = comboToTheta(combos_down, deltap)

    ##make all determinants in the space
    Alldets = vmap(GivensRotationsUtil.given_to_Q, (None, 0, None, None))(n_sites, G_mats_down, idx_down, ref_state_down)
    Alldets_down = Alldets[:,:,:n_elec[1]]
    Alldets_up = Alldets[:,:,:n_elec[0]]
    is_boundary_coord = (combos_down == 0) | (combos_down == Ndiscrete - 1)
    is_boundary_point = jnp.any(is_boundary_coord, axis=1)
    boundary_combos = combos_down[is_boundary_point]
    interior_combos = combos_down[~is_boundary_point]
    
    boundary_linear = np.ravel_multi_index(boundary_combos.T, shapeTuple_down)
    interior_linear = np.ravel_multi_index(interior_combos.T, shapeTuple_down)

    dp1_down = vmap(jax.jacfwd(lambda tau, det : GivensRotationsUtil.applyh_1e_ov(tau, t_matrix, det, idx_down, ref_state_down, n_elec[1]), argnums=0), (None, 0))(epsilon, Alldets_down)

    DPsiDp_idx_down, DPsiDp_Coeff_down = Op_1e.makePDEmatrix_down(1.0, 1.0, shapeTuple_down, idx_down, dp1_down,deltap, Ndiscrete, Alldets_down, comboToTheta, n_sites, n_elec, combos_down, ref_state_down)

    fields_down = 1*np.eye(n_sites)
    dp1_down = vmap(vmap(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_down, ref_state_down, n_elec[1]), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_down, fields_down)
    dp2_down = vmap(vmap(jax.jacfwd(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_down, ref_state_down, n_elec[1]), argnums=0), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_down, fields_down)

    d2pdx2_down = jnp.einsum('ixj->ij', dp2_down) 
    dpdx_down   = d2pdx2_down[:,1:]/2. + jnp.einsum('ixj,ix->ij', dp1_down[:,:,1:], dp1_down[:,:,0]) 
    H_down = jnp.einsum('ixj, ixk->ijk', dp1_down[:,:,1:], dp1_down[:,:,1:])

    factora, factorb, factorc = -0.5, -1., -0.5
    DPsiDp_idx_2e_down, DPsiDp_Coeff_2e_down = Op_2e.makePDEmatrix_down(factora, factorb, factorc, Alldets_down, shapeTuple_down, idx_down, H_down, dpdx_down, d2pdx2_down, deltap, Ndiscrete, n_sites, n_elec, combos_down,  comboToTheta, ref_state_down)
    acth = lambda Psi: ActH(DPsiDp_idx_down, DPsiDp_Coeff_down, Psi) + ActH(DPsiDp_idx_2e_down, DPsiDp_Coeff_2e_down, Psi)

    hf_sr_cp_hs_energies = jax.vmap(calc_energy_hf, in_axes=(0, 0, None))(
    Alldets_up, Alldets_down, u
    )

    hf_sr_cp_hs_overlaps = jax.vmap(calc_overlap_hf)(Alldets_up, Alldets_down)

if n_elec[0] != 0 and n_elec[1] != 0:
    shapeTuple_up = np.asarray((Ndiscrete,)*idx_up.shape[0])
    combos_up = jnp.array(vmap(jnp.unravel_index, (0, None))(jnp.arange(Ndiscrete**idx_up.shape[0]), shapeTuple_up)).T
    deltap = np.pi / (Ndiscrete)
    G_mats_up = comboToTheta(combos_up, deltap)
    
    ##make all determinants in the space
    Alldets_up = vmap(GivensRotationsUtil.given_to_Q, (None, 0, None, None))(n_sites, G_mats_up, idx_up, ref_state_up)
    Alldets_up = Alldets_up[:,:,:n_elec[0]]
    shapeTuple_down = np.asarray((Ndiscrete,)*idx_down.shape[0])
    combos_down = jnp.array(vmap(jnp.unravel_index, (0, None))(jnp.arange(Ndiscrete**idx_down.shape[0]), shapeTuple_down)).T
    # deltap = np.pi / (Ndiscrete-1)
    deltap = np.pi / (Ndiscrete)
    G_mats_down = comboToTheta(combos_down, deltap)

    ##make all determinants in the space
    Alldets_down = vmap(GivensRotationsUtil.given_to_Q, (None, 0, None, None))(n_sites, G_mats_down, idx_down, ref_state_down)
    Alldets_down = Alldets_down[:,:,:n_elec[1]]
    shapeTuple = np.array(list(shapeTuple_up) + list(shapeTuple_down))
    combos_full = np.vstack([
        np.concatenate([u, d])
        for u in combos_up
        for d in combos_down
    ])
    G_mats_full = np.vstack([
        np.concatenate([u, d])
        for u in G_mats_up
        for d in G_mats_down
    ])
    is_boundary_coord = (combos_full == 0) | (combos_full == Ndiscrete - 1)
    is_boundary_point = jnp.any(is_boundary_coord, axis=1)
    boundary_combos = combos_full[is_boundary_point]
    interior_combos = combos_full[~is_boundary_point]

    boundary_linear = np.ravel_multi_index(boundary_combos.T, shapeTuple)
    interior_linear = np.ravel_multi_index(interior_combos.T, shapeTuple)

    dp1_up = vmap(jax.jacfwd(lambda tau, det : GivensRotationsUtil.applyh_1e_ov(tau, t_matrix, det, idx_up, ref_state_up, n_elec[0]), argnums=0), (None, 0))(epsilon, Alldets_up)
    dp1_down = vmap(jax.jacfwd(lambda tau, det : GivensRotationsUtil.applyh_1e_ov(tau, t_matrix, det, idx_down, ref_state_down, n_elec[1]), argnums=0), (None, 0))(epsilon, Alldets_down)

    DPsiDpq_idx, DPsiDpq_coeff = Op_1e.makeExtendedPDEmatrix(1.0, 1.0, shapeTuple, idx_up, idx_down, dp1_up, dp1_down, deltap, Ndiscrete, Alldets_up, Alldets_down, comboToTheta, n_sites, n_elec, combos_up, combos_down, ref_state_up, ref_state_down)

    fields_up = 1*np.eye(n_sites)
    dp1_up = vmap(vmap(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_up, ref_state_up,n_elec[0]), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_up, fields_up)
    dp2_up = vmap(vmap(jax.jacfwd(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_up, ref_state_up,n_elec[0]), argnums=0), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_up, fields_up)

    d2pdx2_up = jnp.einsum('ixj->ij', dp2_up) 
    dpdx_up   = d2pdx2_up[:,1:]/2. + jnp.einsum('ixj,ix->ij', dp1_up[:,:,1:], dp1_up[:,:,0]) 
    H_up = jnp.einsum('ixj, ixk->ijk', dp1_up[:,:,1:], dp1_up[:,:,1:])

    fields_down = 1*np.eye(n_sites)
    dp1_down = vmap(vmap(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_down, ref_state_down,n_elec[1]), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_down, fields_down)
    dp2_down = vmap(vmap(jax.jacfwd(jax.jacfwd(lambda tau, det, flds : GivensRotationsUtil.applyh_2e_ov(tau, u, det, flds, idx_down, ref_state_down,n_elec[1]), argnums=0), argnums=0), (None, None, 0)), (None, 0, None))(epsilon, Alldets_down, fields_down)

    d2pdx2_down = jnp.einsum('ixj->ij', dp2_down) 
    dpdx_down   = d2pdx2_down[:,1:]/2. + jnp.einsum('ixj,ix->ij', dp1_down[:,:,1:], dp1_down[:,:,0]) 
    H_down = jnp.einsum('ixj, ixk->ijk', dp1_down[:,:,1:], dp1_down[:,:,1:])


    dp1up_dwdown = jnp.einsum('pxj,qx->pqj', dp1_up[:, :, 1:], dp1_down[:, :, 0])
    dp1down_dwup = jnp.einsum('qxj,px->qpj', dp1_down[:, :, 1:], dp1_up[:, :, 0])
    dp1up_dp1down = jnp.einsum('ixj, lxk->iljk', dp1_up[:,:,1:], dp1_down[:,:,1:])
    dwup_dwdown = jnp.einsum('ik,jk->ij', dp1_up[:, :, 0], dp1_down[:, :, 0])

    factora, factorb, factorc, factord, factore, factorf, factorg, factorh, factori, factorj = -0.5, -1., -0.5, -0.5, -1., -0.5, +1., +1., -1., -1.
    DPsiDpq_idx_2e, DPsiDpq_coeff_2e = Op_2e.makeExtendedPDEmatrix(factora, factorb, factorc, factord, factore, factorf, factorg, factorh, factori, factorj,                                                                  Alldets_up, Alldets_down,
                                                               shapeTuple,
                                                               idx_up, idx_down,
                                                               H_up, H_down,
                                                               dpdx_up, dpdx_down,
                                                               d2pdx2_up, d2pdx2_down,
                                                               deltap, Ndiscrete,
                                                               n_sites, n_elec,
                                                               combos_up, combos_down,
                                                               comboToTheta,
                                                               ref_state_up, ref_state_down,
                                                               dp1up_dwdown, dp1down_dwup, dp1up_dp1down, dwup_dwdown)
    acth = lambda Psi: ActH(DPsiDpq_idx, DPsiDpq_coeff, Psi) + ActH(DPsiDpq_idx_2e, DPsiDpq_coeff_2e, Psi)

    N_up   = Alldets_up.shape[0]
    N_down = Alldets_down.shape[0]
    up_pairs = jnp.repeat(Alldets_up, repeats=N_down, axis=0)
    down_pairs = jnp.tile(Alldets_down, (N_up, 1, 1))

    hf_sr_cp_hs_energies = jax.vmap(calc_energy_hf, in_axes=(0, 0, None))(
    up_pairs, down_pairs, u
    )

    hf_sr_cp_hs_overlaps = jax.vmap(calc_overlap_hf)(up_pairs, down_pairs)

def propogate(Psi_i, actH, dt):
    return Psi_i + actH(Psi_i)*dt



dt = -1.e-5
n_steps = int(5*1.e5)
is_cp = False
n_hilbert = hf_sr_cp_hs_energies.shape[0]

detId = n_hilbert//2

Psi_i = np.zeros((n_hilbert, ))
Psi_i[detId] = 1.

energy_array = np.zeros(n_steps + 1)
energy = jnp.sum(
hf_sr_cp_hs_energies * hf_sr_cp_hs_overlaps * Psi_i
) / jnp.sum(hf_sr_cp_hs_overlaps * Psi_i)

energy_array[0] = energy

for i in range(n_steps):
    Psi_i =  propogate(Psi_i, acth, dt)

    if is_cp:
        Psi_i[boundary_linear] = 0

    Psi_i = Psi_i/np.linalg.norm(Psi_i)

    energy = jnp.sum(
    hf_sr_cp_hs_energies * hf_sr_cp_hs_overlaps * Psi_i
    ) / jnp.sum(hf_sr_cp_hs_overlaps * Psi_i)

    energy_array[i+1] = energy  

plt.plot(energy_array)
plt.axhline(y=ene_0, color='g', linestyle='--',zorder=3)
plt.show()