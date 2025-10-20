import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
import itertools
from functools import partial

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


##make FCI basis from nsite and n_electrons
def make_basis(n_sites, n_elec: tuple):
    basis_up = make_basis_spin(n_sites, n_elec[0])
    if n_elec[0] == n_elec[1]:
        basis_down = basis_up
    else:
        basis_down = make_basis_spin(n_sites, n_elec[1])
    basis = itertools.product(basis_up, basis_down)
    return np.array(list(basis))


if __name__ == "__main__":
    n_sites = 3
    n_elec = (1, 0)

    ##make the hamiltonian which is u and hopping term
    u = 2.0
    np.random.seed(0)
    h = np.random.normal(size=(n_sites, n_sites))
    t_matrix = h + h.T

    ##make the FCI basis
    ci_basis = jnp.array(make_basis(n_sites, n_elec), dtype=jnp.int32)

    ##make the hamiltonian matrix
    hamiltonian_matrix = np.zeros((len(ci_basis), len(ci_basis)))
    h1_mat = vmap(vmap(ham_element_hubbard_general, (None, 0, None, None)), (0, None, None, None))(ci_basis, ci_basis, u, t_matrix)

    print(ci_basis)
    print("Eigenvalues: ", h1_mat)
    print(t_matrix)
    print(h1_mat)