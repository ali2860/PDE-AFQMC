from jax import config
import Op_1e, Op_2e, GivensRotationsUtil, CIexpansion
from importlib import reload


config.update("jax_enable_x64", True)

from jax import vmap, jit, numpy as jnp, scipy as jsp, random, lax, scipy as jsp
import jax
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
import itertools

import os
reload(GivensRotationsUtil)
reload(CIexpansion)
reload(Op_1e)
reload(Op_2e)

np.set_printoptions(precision=7, suppress=True)


@jit
def ham_element_hubbard_general(x, y, u, t_matrix):
    # coulomb
    diff = jnp.array((jnp.bitwise_xor(x[0], y[0]), jnp.bitwise_xor(x[1], y[1])))
    diff_count = jnp.array((jnp.sum(diff[0]), jnp.sum(diff[1])))
    on_site = (jnp.sum(diff_count) == 0) * u * jnp.bitwise_and(x[0], x[1]).sum()
    mu = jnp.diag(t_matrix)
    n_occ = x[0] + x[1]
    total_mu = jnp.dot(n_occ, mu)
    on_site = on_site + (jnp.sum(diff_count) == 0) * total_mu

    # hopping
    is_hopping = jnp.sum(diff_count) == 2
    diff_pos = jnp.nonzero(diff, size=2)
    spin_idx = diff_pos[0][0]
    site_1 = diff_pos[1][0]
    site_2 = diff_pos[1][1]
    t_element = t_matrix[site_1, site_2]
    # parity
    min_site = jnp.minimum(site_1, site_2)
    max_site = jnp.maximum(site_1, site_2)
    sites = jnp.arange(len(x[0]))
    mask = (sites > min_site) & (sites < max_site)
    electrons_between = jnp.sum(y[spin_idx] * mask)
    parity_factor = jnp.power(-1.0, electrons_between)
    hopping = is_hopping * t_element * parity_factor

    return on_site + hopping


def make_basis_spin(n_sites, n_elec):
    # generate permutations using lexicographic order
    basis = []
    elec = np.zeros(n_sites)
    for i in range(n_elec):
        elec[-i - 1] = 1
    basis.append(elec.copy())
    # find next permutation
    while True:
        k = -1
        for i in range(n_sites - 1):
            if elec[i] < elec[i + 1]:
                k = i
        if k == -1:
            break
        l = k
        for i in range(k + 1, n_sites):
            if elec[k] < elec[i]:
                l = i
        elec[k], elec[l] = elec[l], elec[k]
        elec[k + 1 :] = np.flip(elec[k + 1 :])
        basis.append(elec.copy())
    return np.array(basis, dtype=int)


def make_basis(n_sites, n_elec: tuple):
    basis_up = make_basis_spin(n_sites, n_elec[0])
    if n_elec[0] == n_elec[1]:
        basis_down = basis_up
    else:
        basis_down = make_basis_spin(n_sites, n_elec[1])
    basis = itertools.product(basis_up, basis_down)
    return np.array(list(basis))





n_sites = 3
n_elec = (1,0)
# n_sites = 2
# n_elec = (1,1)
u = 10
ci_basis = jnp.array(make_basis(n_sites, n_elec), dtype=jnp.int32)
print(f"built basis, length: {len(ci_basis)}")
h1 = np.zeros((n_sites,n_sites))

# h1 = jnp.array([[-0.6680644, 0.9605467, 0.520678, -0.1044906],

# [ 0.9605467, -0.1900202, 0.3938467, -1.0511902],

# [ 0.520678 , 0.3938467 , 0.0340779, -0.4899371],

# [-0.1044906, -1.0511902, -0.4899371, 0.0882198]]) # for n = 4

h1 = jnp.array([[-0.6680644, 0.9605467, 0.520678],

[ 0.9605467, -0.1900202, 0.3938467],

[ 0.520678 , 0.3938467 , 0.0340779]]) # for n = 3



ham_element = partial(ham_element_hubbard_general, u=u, t_matrix=h1)
ham_mat = vmap(vmap(ham_element, (None, 0)), (0, None))(ci_basis, ci_basis)
print("built hamiltonian")
ene_diag, evec_diag = jnp.linalg.eigh(ham_mat)
evec_diag *= np.sign(evec_diag[0, 0])
print(f"Exact energies: ", ene_diag)

ham_element_h1 = partial(ham_element_hubbard_general, u=0.0, t_matrix=h1)
h1_mat = vmap(vmap(ham_element_h1, (None, 0)), (0, None))(ci_basis, ci_basis)
ham_element_u = partial(ham_element_hubbard_general, u=u, t_matrix=0.0 * h1)
u_mat = vmap(vmap(ham_element_u, (None, 0)), (0, None))(ci_basis, ci_basis)

h1_ene, h1_evec = jnp.linalg.eigh(h1)
hf_u = h1_evec[:, :n_elec[0]]
hf_d = h1_evec[:, :n_elec[1]]
hf = [hf_u, hf_d]

np.random.seed(123)

if n_elec[0] == 0 or n_elec[1] == 0:
    # noise_up = np.random.uniform(-0.5, 0.5, hf_u.shape) # For n_sites = 4, n_elec = (2,0) t_matrix2
    # noise_up = np.random.uniform(-1.9, 1.9, hf_u.shape)
    noise_up = np.random.uniform(-3, 3, hf_u.shape) # For n_sites = 3, n_elec = (1,0) t_matrix2
    # noise_up = np.random.uniform(-1, 1, hf_u.shape)
    hf_u += noise_up
    hf = [hf_u, hf_d]



# overlap and energy functions for HF and exact ground state
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
    return _calc_overlap(hf_u, hf_d, ru, rd)


@jit
def calc_energy_hf(ru, rd, u):
    return _calc_energy(hf_u, hf_d, ru, rd, h1, u)

    
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


hf_vec = calc_hilbert_state(hf_u, hf_d)
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


total_time = 10
n_steps = 10000
dt = total_time / n_steps
ene_0 = ene_diag[0]



@jit
def stochastic_reconfiguration(walkers, weights, zeta):
    nwalkers = walkers[0].shape[0]
    cumulative_weights = jnp.cumsum(jnp.abs(weights))
    total_weight = cumulative_weights[-1]
    average_weight = total_weight / nwalkers
    weights = jnp.ones(nwalkers) * average_weight
    z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
    indices = vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
    walkers[0] = walkers[0][indices]
    walkers[1] = walkers[1][indices]
    return walkers, weights

@partial
def random_walk_sr_uniform(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas):
    # one walker, one step
    def prop_one_step(det_u, det_d, weight, fields_t):
        det_u = exp_h1 @ det_u
        det_d = exp_h1 @ det_d
        gamma_diag_u = jnp.exp(gamma * fields_t)
        gamma_diag_d = jnp.exp(-gamma * fields_t)
        det_u = det_u * gamma_diag_u[:, None]
        det_d = det_d * gamma_diag_d[:, None]
        det_u = exp_e0 * exp_h1 @ det_u
        det_d = exp_e0 * exp_h1 @ det_d
        det_u, r_u = jnp.linalg.qr(det_u)
        det_d, r_d = jnp.linalg.qr(det_d)
        norm_u = jnp.prod(jnp.diagonal(r_u))
        det_u = det_u.at[:, 0].mul(jnp.sign(norm_u))
        norm_d = jnp.prod(jnp.diagonal(r_d))
        det_d = det_d.at[:, 0].mul(jnp.sign(norm_d))
        return det_u, det_d, jnp.abs(norm_u) * jnp.abs(norm_d) * weight

    walkers_u, walkers_d, weights = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t)
    
    return [walkers_u, walkers_d], weights

@jit
def random_walk_sr_uniform_up_only(walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas):
    # one walker, one step
    def _prop_one_step(det_u, det_d, weight, fields_t):
        det_u = exp_h1 @ det_u
        det_d = det_d 
        gamma_diag_u = jnp.exp(gamma * fields_t)
        det_u = det_u * gamma_diag_u[:, None]
        det_d = det_d 
        det_u = exp_e0 * exp_h1 @ det_u
        det_d = det_d 
        det_u, r_u = jnp.linalg.qr(det_u)
        det_d, r_d = jnp.linalg.qr(det_d)
        norm_u = jnp.prod(jnp.diagonal(r_u))
        det_u = det_u.at[:, 0].mul(jnp.sign(norm_u))
        norm_d = jnp.prod(jnp.diagonal(r_d))
        det_d = det_d 
        return det_u, det_d, jnp.abs(norm_u) * jnp.abs(norm_d) * weight

    walkers_u, walkers_d, weights = jax.vmap(_prop_one_step)(walkers[0], walkers[1], weights, fields_t)
    
    return [walkers_u, walkers_d], weights



@partial(jit, static_argnums=7)
def random_walk_sr_uniform_cp(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas, overlap_fun, numb_flips_array
):
    # one walker, one step
    def prop_one_step(det_u, det_d, weight, fields_t, numb_flips):
        overlap = overlap_fun(det_u, det_d)
        det_u = exp_h1 @ det_u
        det_d = exp_h1 @ det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_1 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        gamma_diag_u = jnp.exp(gamma * fields_t)
        gamma_diag_d = jnp.exp(-gamma * fields_t)
        det_u = det_u * gamma_diag_u[:, None]
        det_d = det_d * gamma_diag_d[:, None]
        overlap_new = overlap_fun(det_u, det_d)
        sign_2 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        det_u = exp_e0 * exp_h1 @ det_u
        det_d = exp_e0 * exp_h1 @ det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_3 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        is_neg1 = sign_1 < 0
        is_neg2 = sign_2 < 0
        is_neg3 = sign_3 < 0
        numb_flips = numb_flips + is_neg1.astype(int) + is_neg2.astype(int) + is_neg3.astype(int)
        sign = is_neg1 + is_neg2 + is_neg3
        det_u, r_u = jnp.linalg.qr(det_u)
        det_d, r_d = jnp.linalg.qr(det_d)
        norm_u = jnp.prod(jnp.diagonal(r_u))
        det_u = det_u.at[:, 0].mul(jnp.sign(norm_u))
        norm_d = jnp.prod(jnp.diagonal(r_d))
        det_d = det_d.at[:, 0].mul(jnp.sign(norm_d))
        return det_u, det_d, jnp.abs(norm_u) * jnp.abs(norm_d) * weight * (sign == 0), numb_flips

    walkers_u, walkers_d, weights, numb_flips_array = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t, numb_flips_array)
    
    return [walkers_u, walkers_d], weights, numb_flips_array


@partial(jit, static_argnums=7)
def random_walk_sr_uniform_cp_up_only(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas, overlap_fun, numb_flips_array
):
    # one walker, one step
    def prop_one_step(det_u, det_d, weight, fields_t, numb_flips):
        overlap = overlap_fun(det_u, det_d)
        det_u = exp_h1 @ det_u
        det_d = det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_1 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        gamma_diag_u = jnp.exp(gamma * fields_t)
        det_u = det_u * gamma_diag_u[:, None]
        det_d = det_d 
        overlap_new = overlap_fun(det_u, det_d)
        sign_2 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        det_u = exp_e0 * exp_h1 @ det_u
        det_d = det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_3 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        is_neg1 = sign_1 < 0
        is_neg2 = sign_2 < 0
        is_neg3 = sign_3 < 0
        numb_flips = numb_flips + is_neg1.astype(int) + is_neg2.astype(int) + is_neg3.astype(int)
        sign = is_neg1 + is_neg2 + is_neg3
        det_u, r_u = jnp.linalg.qr(det_u)
        det_d, r_d = jnp.linalg.qr(det_d)
        norm_u = jnp.prod(jnp.diagonal(r_u))
        det_u = det_u.at[:, 0].mul(jnp.sign(norm_u))
        norm_d = jnp.prod(jnp.diagonal(r_d))
        det_d = det_d
        return det_u, det_d, jnp.abs(norm_u) * jnp.abs(norm_d) * weight * (sign == 0), numb_flips

    walkers_u, walkers_d, weights, numb_flips_array = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t, numb_flips_array)
    
    return [walkers_u, walkers_d], weights, numb_flips_array


@partial(jit, static_argnums=7)
def random_walk_sr_uniform_cp_up_only_modified_swap(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas, overlap_fun, numb_flips_array
):
    # one walker, one step
    def prop_one_step(det_u, det_d, weight, fields_t, numb_flips):
        overlap = overlap_fun(det_u, det_d)
        det_u = exp_h1 @ det_u
        det_d = det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_1 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        gamma_diag_u = jnp.exp(gamma * fields_t)
        det_u = det_u * gamma_diag_u[:, None]
        det_d = det_d 
        overlap_new = overlap_fun(det_u, det_d)
        sign_2 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        det_u = exp_e0 * exp_h1 @ det_u
        det_d = det_d
        overlap_new = overlap_fun(det_u, det_d)
        sign_3 = jnp.sign(overlap_new / overlap)
        overlap = overlap_new
        is_neg1 = sign_1 < 0
        is_neg2 = sign_2 < 0
        is_neg3 = sign_3 < 0
        numb_flips = numb_flips + is_neg1.astype(int) + is_neg2.astype(int) + is_neg3.astype(int)
        det_u = lax.cond(
            numb_flips > 0,
            lambda d: d.at[:, [0, 1]].set(d[:, [1, 0]]), 
            lambda d: d,                                 
            det_u
        )
        det_u, r_u = jnp.linalg.qr(det_u)
        det_d, r_d = jnp.linalg.qr(det_d)
        norm_u = jnp.prod(jnp.diagonal(r_u))
        det_u = det_u.at[:, 0].mul(jnp.sign(norm_u))
        norm_d = jnp.prod(jnp.diagonal(r_d))
        det_d = det_d
        return det_u, det_d, jnp.abs(norm_u) * jnp.abs(norm_d) * weight, numb_flips

    walkers_u, walkers_d, weights, numb_flips_array = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t, numb_flips_array)
    
    return [walkers_u, walkers_d], weights, numb_flips_array


@partial(jit, static_argnums=7)
def random_walk_sr_uniform_cp_modified(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas, overlap_fun, numb_flips_array
):
    def prop_one_step(det_u, det_d, weight, fields_t, nf):
        overlap0 = overlap_fun(det_u, det_d)

        det1_u = exp_h1 @ det_u
        det1_d = exp_h1 @ det_d
        overlap1 = overlap_fun(det1_u, det1_d)
        flip1 = (jnp.sign(overlap1/overlap0) < 0).astype(jnp.int32)

        gamma_u = jnp.exp(gamma * fields_t)[:,None]
        gamma_d = jnp.exp(-gamma * fields_t)[:,None]
        det2_u = det1_u * gamma_u
        det2_d = det1_d * gamma_d
        overlap2 = overlap_fun(det2_u, det2_d)
        flip2 = (jnp.sign(overlap2/overlap1) < 0).astype(jnp.int32)

        det3_u = exp_e0 * exp_h1 @ det2_u
        det3_d = exp_e0 * exp_h1 @ det2_d
        overlap3 = overlap_fun(det3_u, det3_d)
        flip3 = (jnp.sign(overlap3/overlap2) < 0).astype(jnp.int32)

        total_flips = flip1 + flip2 + flip3
        nf_new = nf + total_flips

        q_u, r_u = jnp.linalg.qr(det3_u)
        q_d, r_d = jnp.linalg.qr(det3_d)
        nu = jnp.prod(jnp.diagonal(r_u))  
        nd = jnp.prod(jnp.diagonal(r_d))
        sign_u = jnp.sign(nu)
        sign_d = jnp.sign(nd)
        q_u = q_u.at[:,0].mul(sign_u)
        q_d = q_d.at[:,0].mul(sign_d)
        w_new = weight * jnp.abs(nu) * jnp.abs(nd)

        no_flip = (total_flips == 0)
        # no_flip = True
        det_u_out = lax.select(no_flip, q_u, det_u)
        det_d_out = lax.select(no_flip, q_d, det_d)
        w_out = lax.select(no_flip, w_new, weight)

        return det_u_out, det_d_out, w_out, nf_new


    walkers_u, walkers_d, weights, numb_flips_array = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t, numb_flips_array)
    
    return [walkers_u, walkers_d], weights, numb_flips_array


@partial(jit, static_argnums=7)
def random_walk_sr_uniform_cp_modified_up_only(
    walkers, weights, exp_h1, exp_e0, gamma, fields_t, zetas, overlap_fun, numb_flips_array
):
    def prop_one_step(det_u, det_d, weight, fields_t, nf):
        overlap0 = overlap_fun(det_u, det_d)

        det1_u = exp_h1 @ det_u
        det1_d = det_d
        overlap1 = overlap_fun(det1_u, det1_d)
        flip1 = (jnp.sign(overlap1/overlap0) < 0).astype(jnp.int32)

        gamma_u = jnp.exp(gamma * fields_t)[:,None]
        det2_u = det1_u * gamma_u
        det2_d = det_d
        overlap2 = overlap_fun(det2_u, det2_d)
        flip2 = (jnp.sign(overlap2/overlap1) < 0).astype(jnp.int32)

        det3_u = exp_e0 * exp_h1 @ det2_u
        det3_d = det_d
        overlap3 = overlap_fun(det3_u, det3_d)
        flip3 = (jnp.sign(overlap3/overlap2) < 0).astype(jnp.int32)

        total_flips = flip1 + flip2 + flip3
        nf_new = nf + total_flips

        q_u, r_u = jnp.linalg.qr(det3_u)
        q_d, r_d = jnp.linalg.qr(det3_d)
        nu = jnp.prod(jnp.diagonal(r_u))  
        nd = jnp.prod(jnp.diagonal(r_d))
        sign_u = jnp.sign(nu)
        q_u = q_u.at[:,0].mul(sign_u)
        q_d = det_d
        w_new = weight * jnp.abs(nu) * jnp.abs(nd)

        no_flip = (total_flips == 0)
        # no_flip = True
        det_u_out = lax.select(no_flip, q_u, det_u)
        det_d_out = lax.select(no_flip, q_d, det_d)
        w_out = lax.select(no_flip, w_new, weight)

        return det_u_out, det_d_out, w_out, nf_new


    walkers_u, walkers_d, weights, numb_flips_array = jax.vmap(prop_one_step)(walkers[0], walkers[1], weights, fields_t, numb_flips_array)
    
    return [walkers_u, walkers_d], weights, numb_flips_array




cutoff = 1e-5
numb_of_experiments = 10
n_walkers = 10000

total_energy_array_sr = np.zeros((numb_of_experiments,n_steps))
numb_flips_array = np.zeros((numb_of_experiments,n_steps))
final_state_array = np.zeros((numb_of_experiments,len(ci_basis)))
final_walkers_array = np.zeros((numb_of_experiments,n_walkers,n_sites,n_elec[0]))
min_weights = np.zeros((numb_of_experiments,n_steps))

exp_h1 = jsp.linalg.expm(-h1 * dt / 2)
# gamma = jnp.arccosh(jnp.exp(dt * u / 2))
gamma = -jnp.sqrt(dt)*np.sqrt(u)

# exp_e0 = jnp.exp(
    # dt * (ene_0 - 2 * u) / n_elec[0]
# )  # including constant term in the hs that we ignored (FOR SPIN UP OR DOWN ONLY)

exp_e0 = jnp.exp(dt * ene_0 / 2 / n_elec[0])
# exp_e0 = jnp.exp(
    # dt * (ene_0 - u) / 2 / n_elec[0]
# )  # including constant term in the hs that we ignored


for tries in np.arange(numb_of_experiments):

    walkers = [jnp.array([h1_evec[:, :n_elec[0]]] * n_walkers), jnp.array([hf_d] * n_walkers)]
    weights = jnp.ones(n_walkers)

    hf_sr_cp_hs_energies = jax.vmap(calc_energy_hf, in_axes=(0, 0, None))(
    walkers[0], walkers[1], u
    )

    hf_sr_cp_hs_overlaps = jax.vmap(calc_overlap_hf)(walkers[0], walkers[1])
    
    energy = jnp.sum(
        hf_sr_cp_hs_energies * hf_sr_cp_hs_overlaps * weights
    ) / jnp.sum(hf_sr_cp_hs_overlaps * weights)

    seed = np.random.randint(1000)
    key = jax.random.PRNGKey(seed)

    if tries % 10 == 0:
        print("Experiment: ", tries)

        print("initial energy:",energy)

        print(f"Seed: {seed}")

    reconfig_interval = 1

    for time_step in range(n_steps):
        numb_flips = np.zeros(n_walkers)

        key, subkey = jax.random.split(key)

        # uniform_fields_t = jax.random.choice(
            # subkey, jnp.array([-1.0, 1.0]), shape=(n_walkers, n_sites)
        # )

        uniform_fields_t = jax.random.normal(
            subkey, 
            shape=(n_walkers, n_sites)
        )

        key, subkey = jax.random.split(key)
        zeta_t = jax.random.uniform(subkey, shape=())

        sr_uniform_cp_hs_dets, sr_uniform_cp_hs_weights, numb_flips_new = random_walk_sr_uniform_cp_up_only(
            walkers, weights, exp_h1, exp_e0, gamma, uniform_fields_t, zeta_t, calc_overlap_exact, numb_flips
        )

        # sr_uniform_cp_hs_dets, sr_uniform_cp_hs_weights = random_walk_sr_uniform_up_only(
        #     walkers, weights, exp_h1, exp_e0, gamma, uniform_fields_t, zeta_t
        # )

        min_weights[tries,time_step] = np.min(sr_uniform_cp_hs_weights)

        # numb_flips_array[tries,time_step] = np.sum(numb_flips_new)
        

        if time_step % reconfig_interval == 0:
            walkers, weights = stochastic_reconfiguration(
                [sr_uniform_cp_hs_dets[0], sr_uniform_cp_hs_dets[1]], 
                sr_uniform_cp_hs_weights, 
                zeta_t
            )
        else:
            walkers, weights = sr_uniform_cp_hs_dets, sr_uniform_cp_hs_weights

        hf_sr_cp_hs_energies = jax.vmap(calc_energy_hf, in_axes=(0, 0, None))(
        walkers[0], walkers[1], u
        )

        hf_sr_cp_hs_overlaps = jax.vmap(calc_overlap_hf)(walkers[0], walkers[1])

        energy = jnp.sum(
            hf_sr_cp_hs_energies * hf_sr_cp_hs_overlaps * weights
        ) / jnp.sum(hf_sr_cp_hs_overlaps * weights)

        total_energy_array_sr[tries,time_step] = energy

        # if time_step % 1000 == 0:   
            # print(f"Step {time_step}/{n_steps} completed in Experiment {tries}")
    
    final_walkers_array[tries,...] = walkers[0]
    state = calc_hilbert_state_det_sum(walkers,weights)
    final_state_array[tries,:] = state/jnp.linalg.norm(state)