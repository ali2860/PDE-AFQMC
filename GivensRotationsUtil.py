import numpy as np
import jax.numpy as jnp
from jax import jit
import jax
from itertools import combinations
import jax.scipy as jsp
from functools import partial


@jit
def givens_rotation_matrix(G, row, col, theta):
    c, s = jnp.cos(theta), jnp.sin(theta)
    G = G.at[row, row].set(c)
    G = G.at[col, col].set(c)
    G = G.at[row, col].set(-s)
    G = G.at[col, row].set(s)

    return G

@jit
def givens_rotation(a, b):
    """Compute c, s for a Givens rotation that zeroes out b."""
    r = ((a)**2 + (b)**2)**0.5
    c = jnp.abs(a) / r
    s = -b*jnp.sign(a) / r
    return c, s


def makeUpperTriangularWithColumnOps(A, n=None):
    if n is None:
        n = A.shape[1]
    
    # Precompute all masks for each i value
    masks = jnp.arange(n)[None, :] < jnp.arange(n)[:, None]
    
    def body_fn(i, A):
        # Use precomputed mask for this i
        mask = masks[i]
        
        # Compute factors for all columns
        factors = jnp.where(mask, A[i, :] / A[i, i], 0)
        
        # Update all columns simultaneously using outer product
        update = jnp.outer(A[:, i], factors)
        return A - update
    
    # Process rows in reverse order using fori_loop
    return jax.lax.fori_loop(0, n, lambda i, A: body_fn(n-1-i, A), A)


@jit
def qr_fun(carry, givensIdx):
    A = carry
    n = A.shape[0]

    i, j = givensIdx[0], givensIdx[1]
    a, b = A[i, i], A[j, i]
    c, s = givens_rotation(a, b)
    theta = jnp.atan2(s, c)

    G = givens_rotation_matrix(np.eye(n), i, j, theta)
    A = G @ A
    return (A), theta



@jit
def givens_qr_decomposition_jax_ref(Q, idx):
    return jax.lax.scan(qr_fun, (Q), idx)


def given_to_Q_ref(n, theta, idx):
    def fun(carry, givensIdx):
        i, j, theta = givensIdx[0].astype(jnp.int32), givensIdx[1].astype(jnp.int32), givensIdx[3]
        return givens_rotation_matrix(np.eye(n), i, j, -theta) @ carry, None
    return jax.lax.scan(fun, jnp.eye(n), jnp.hstack((idx[::-1, :], theta.reshape(-1,1)[::-1, :])))[0]
given_to_Q_ref = jit(given_to_Q_ref, static_argnums=0)


@jit
def givens_qr_decomposition_jax(Q, idx, ref):
    Q_eff = ref.T @ Q 
    return jax.lax.scan(qr_fun, (Q_eff), idx)


@partial(jit, static_argnums=(3,))
def givens_qr_decomposition_jax_ov(Q, idx, ref, n):
    Q_eff = ref.T @ Q 
    Q_eff = makeUpperTriangularWithColumnOps(Q_eff, n)
    return jax.lax.scan(qr_fun, (Q_eff), idx)


@partial(jit, static_argnums=(0,))
def given_to_Q_(n, theta, idx, ref):
    def fun(carry, givensIdx):
        i, j, theta = givensIdx[0].astype(jnp.int32), givensIdx[1].astype(jnp.int32), givensIdx[3]
        return givens_rotation_matrix(np.eye(n), i, j, -theta) @ carry, None
    args   = jnp.hstack((idx[::-1, :],
                        theta.reshape(-1,1)[::-1, :]))
    Q_rec, _ = jax.lax.scan(fun, jnp.eye(n), args)

    return (ref @ Q_rec)
given_to_Q = jit(given_to_Q_, static_argnums=0)



@jit
def applyh_1e(tau, h, det_u, idx, ref):
    ##make propogation
    prop = jsp.linalg.expm(h*tau)

    ##apply propogator
    detout = prop@det_u

    #final theta
    theta1 = givens_qr_decomposition_jax(detout, idx, ref)[1]

    #do qr decomposition
    norm = jnp.linalg.det(detout.T @ detout)**0.5

    return jnp.hstack((norm, theta1))


@partial(jit, static_argnums=(5,))
def applyh_1e_ov(tau, h, det_u, idx, ref, n_elecs):
    ##make propagation
    prop = jsp.linalg.expm(h * tau)

    ##apply propagator
    detout = prop @ det_u

    #final theta
    theta1 = givens_qr_decomposition_jax_ov(detout, idx, ref, n_elecs)[1]

    #do qr decomposition
    norm = jnp.linalg.det(detout.T @ detout)**0.5

    return jnp.hstack((norm, theta1))

def applyh_2e(x, u, det_u, fields_t, idx, ref):

    ##make propogation
    gamma_u = jnp.exp(x * u**0.5 * fields_t)[:,None]

    ##apply propogator
    detout = det_u * gamma_u

    #final theta
    r, theta1 = givens_qr_decomposition_jax(detout, idx, ref)

    #do qr decomposition
    norm = jnp.linalg.det(detout.T @ detout)**0.5

    return jnp.hstack((norm, theta1))



def applyh_2e_ov(x, u, det_u, fields_t, idx, ref, n_elecs):

    ##make propogation
    gamma_u = jnp.exp(x * u**0.5 * fields_t)[:,None]

    ##apply propogator
    detout = det_u * gamma_u

    #final theta
    r, theta1 = givens_qr_decomposition_jax_ov(detout, idx, ref, n_elecs)

    #do qr decomposition
    norm = jnp.linalg.det(detout.T @ detout)**0.5

    return jnp.hstack((norm, theta1))


def nextMappedDeterminant_2_ref(I, k1, k2, dk1, dk2, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos):
    Index = list(np.unravel_index(I, shapeTuple))
    Index2 = Index.copy()
    Index2[k1] = (Index[k1]+dk1)%Ndiscrete
    Index2[k2] = (Index[k2]+dk2)%Ndiscrete

    if (Index2[k1] != 0 and Index2[k1] != Ndiscrete-1) and (Index2[k2] != 0 and Index2[k2] != Ndiscrete-1):
        return Index2, 1.
    else:
        theta = comboToTheta(np.array(Index), deltap)
        theta2 = theta.copy()
        theta2[k1] = (theta[k1]+dk1*deltap)  ##move past the boundary
        theta2[k2] = (theta[k2]+dk2*deltap)  ##move past the boundary
        det = given_to_Q_ref(n_sites, theta2, idx)[:,:n_elec[0]]
        ovlps = jax.vmap(lambda i : 1. - abs(jnp.linalg.det(det.T @ Alldets[i])) )(jnp.arange(Alldets.shape[0]))
        id = jnp.where(ovlps < 1.e-14)[0]
        return np.array(combos[id][0]),  jnp.linalg.det(det.T @ Alldets[id])[0]    
    

def nextMappedDeterminant_2_up(I, k1, k2, dk1, dk2, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref):
    Index = list(np.unravel_index(I, shapeTuple))
    Index2 = Index.copy()
    Index2[k1] = (Index[k1]+dk1)%Ndiscrete
    Index2[k2] = (Index[k2]+dk2)%Ndiscrete

    if (Index2[k1] != 0 and Index2[k1] != Ndiscrete-1) and (Index2[k2] != 0 and Index2[k2] != Ndiscrete-1):
        return Index2, 1.
    else:
        theta = comboToTheta(np.array(Index), deltap)
        theta2 = theta.copy()
        theta2[k1] = (theta[k1]+dk1*deltap)  ##move past the boundary
        theta2[k2] = (theta[k2]+dk2*deltap)  ##move past the boundary
        det = given_to_Q(n_sites, theta2, idx, ref)[:,:n_elec[0]]
        ovlps = jax.vmap(lambda i : 1. - abs(jnp.linalg.det(det.T @ Alldets[i])) )(jnp.arange(Alldets.shape[0]))
        id = jnp.where(ovlps < 1.e-14)[0]
        return np.array(combos[id][0]),  jnp.linalg.det(det.T @ Alldets[id])[0]    


def nextMappedDeterminant_up(I, k, di, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref):
    Index = list(np.unravel_index(I, shapeTuple))
    Index2 = Index.copy()
    Index2[k] = (Index[k]+di)%Ndiscrete

    if (Index2[k] != 0 and Index2[k] != Ndiscrete-1):
        return Index2, 1.
    else:
        theta = comboToTheta(np.array(Index), deltap)
        theta2 = theta.copy()
        theta2[k] = (theta[k]+di*deltap)  ##move past the boundary
        det = given_to_Q(n_sites, theta2, idx, ref)[:,:n_elec[0]]
        ovlps = jax.vmap(lambda i : 1. - abs(jnp.linalg.det(det.T @ Alldets[i])) )(jnp.arange(Alldets.shape[0]))
        id = jnp.where(ovlps < 1.e-14)[0]
        #import pdb; pdb.set_trace()
        return np.array(combos[id][0]),  jnp.linalg.det(det.T @ Alldets[id])[0]    



def nextMappedDeterminant_2_down(I, k1, k2, dk1, dk2, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref):
    Index = list(np.unravel_index(I, shapeTuple))
    Index2 = Index.copy()
    Index2[k1] = (Index[k1]+dk1)%Ndiscrete
    Index2[k2] = (Index[k2]+dk2)%Ndiscrete

    if (Index2[k1] != 0 and Index2[k1] != Ndiscrete-1) and (Index2[k2] != 0 and Index2[k2] != Ndiscrete-1):
        return Index2, 1.
    else:
        theta = comboToTheta(np.array(Index), deltap)
        theta2 = theta.copy()
        theta2[k1] = (theta[k1]+dk1*deltap)  ##move past the boundary
        theta2[k2] = (theta[k2]+dk2*deltap)  ##move past the boundary
        det = given_to_Q(n_sites, theta2, idx, ref)[:,:n_elec[1]]
        ovlps = jax.vmap(lambda i : 1. - abs(jnp.linalg.det(det.T @ Alldets[i])) )(jnp.arange(Alldets.shape[0]))
        id = jnp.where(ovlps < 1.e-14)[0]
        return np.array(combos[id][0]),  jnp.linalg.det(det.T @ Alldets[id])[0]    


def nextMappedDeterminant_down(I, k, di, deltap, Ndiscrete, Alldets, shapeTuple, idx, comboToTheta, n_sites, n_elec, combos, ref):
    Index = list(np.unravel_index(I, shapeTuple))
    Index2 = Index.copy()
    Index2[k] = (Index[k]+di)%Ndiscrete

    if (Index2[k] != 0 and Index2[k] != Ndiscrete-1):
        return Index2, 1.
    else:
        theta = comboToTheta(np.array(Index), deltap)
        theta2 = theta.copy()
        theta2[k] = (theta[k]+di*deltap)  ##move past the boundary
        det = given_to_Q(n_sites, theta2, idx, ref)[:,:n_elec[1]]
        ovlps = jax.vmap(lambda i : 1. - abs(jnp.linalg.det(det.T @ Alldets[i])) )(jnp.arange(Alldets.shape[0]))
        id = jnp.where(ovlps < 1.e-14)[0]
        #import pdb; pdb.set_trace()
        return np.array(combos[id][0]),  jnp.linalg.det(det.T @ Alldets[id])[0]  