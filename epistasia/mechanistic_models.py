"""
Mechanistic models for epistasis based on Lotka–Volterra dynamics.

This module provides:
- Random generators for pairwise interaction matrices.
- Random generators for third-order mechanistic interaction tensors.
- Deterministic LV simulations.
- Extended LV simulations with explicit third-order terms.
- Exact fitness evaluation at pairwise LV equilibria for binary genotypes.
- Numerical community-function landscape construction for extended LV models.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

__all__ = [
    "gaussian_interaction_matrix",
    "gaussian_hoi_tensor",
    "simulate_lv",
    "simulate_lv_hoi",
    "evaluate_exact_F",
    "evaluate_numerical_F_hoi",
    "build_hoi_lv_landscape",
    "growth_rates",
    "is_lv_stable",
    "is_lv_feasible",
]


############################################
#       INTERACTION MATRIX FUNCTIONS       #
############################################

def gaussian_interaction_matrix(
    S: int,
    C: float,
    mu: float,
    sigma: float,
    diag_value: float = -1.0,
    symmetric: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random interaction matrix A for a Lotka–Volterra system.

    Off-diagonal entries are Gaussian N(mu, sigma^2), thinned by connectance C.
    Diagonal entries are fixed to diag_value (usually negative).

    Parameters
    ----------
    S : int
        Number of species.
    C : float
        Connectance (probability that an off-diagonal interaction is non-zero).
    mu : float
        Mean of the Gaussian distribution for off-diagonal entries.
    sigma : float
        Standard deviation of the Gaussian distribution for off-diagonal entries.
    diag_value : float, default -1.0
        Value set on the diagonal (self-interactions).
    symmetric : bool, default False
        If True, build a symmetric matrix (A_ij = A_ji).
        Note that the diagonal is still fixed to diag_value.
    rng : np.random.Generator, optional
        Random number generator. If None, use default_rng().

    Returns
    -------
    A : ndarray, shape (S, S)
        Interaction matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= C <= 1.0):
        raise ValueError("Connectance C must be in [0, 1].")

    # Start with a zero matrix
    A = np.zeros((S, S), dtype=float)

    # Generate upper triangle (excluding diagonal) if symmetric
    if symmetric:
        # random mask for upper triangle
        mask = rng.random((S, S))
        for i in range(S):
            mask[i, i] = 1.0  # avoid using mask on diagonal
        upper = (mask < C).astype(float)
        # draw Gaussian values
        vals = rng.normal(mu, sigma, size=(S, S))
        upper *= vals
        # symmetrize
        A = upper + upper.T
    else:
        # Full random mask (off-diagonal entries)
        mask = rng.random((S, S))
        # draw Gaussian values
        vals = rng.normal(mu, sigma, size=(S, S))
        A = (mask < C).astype(float) * vals

    # Set diagonal to fixed value (independent of connectance)
    np.fill_diagonal(A, diag_value)

    return A


############################################
#                LV SOLVER                 #
############################################

def simulate_lv(
    alpha: ArrayLike,
    r: ArrayLike,
    t_max: float = 100.0,
    N0: Optional[ArrayLike] = None,
    t_eval: Optional[ArrayLike] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    extinction_threshold: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate the Lotka–Volterra dynamics:

        dN_i/dt = N_i * ( r_i + sum_j alpha_ij N_j )

    Parameters
    ----------
    alpha : array_like, shape (S, S)
        Interaction matrix.
    r : array_like, shape (S,)
        Intrinsic growth rates.
    t_max : float, default 100.0
        Final time of integration.
    N0 : array_like, shape (S,), optional
        Initial abundances. If None, random positive values are used.
    t_eval : array_like, optional
        Time points at which to store the computed solution. If None,
        the solver chooses adaptively.
    rtol : float, default 1e-8
        Relative tolerance for the ODE solver.
    atol : float, default 1e-10
        Absolute tolerance for the ODE solver.
    extinction_threshold : float, default 0.0
        If > 0, abundances are floored at this value inside the RHS
        to avoid negative densities.

    Returns
    -------
    t : ndarray, shape (n_times,)
        Time points of the simulation.
    N : ndarray, shape (S, n_times)
        Abundance trajectories of each species over time.
    """
    alpha = np.asarray(alpha, dtype=float)
    r = np.asarray(r, dtype=float)

    if alpha.ndim != 2 or alpha.shape[0] != alpha.shape[1]:
        raise ValueError("alpha must be a square (S, S) array.")
    S = alpha.shape[0]
    if r.shape != (S,):
        raise ValueError("r must have shape (S,) compatible with alpha.")

    if N0 is None:
        # Small positive initial condition
        rng = np.random.default_rng()
        N0 = rng.normal(loc=0.1, scale=0.05, size=S)
        N0 = np.abs(N0)
    else:
        N0 = np.asarray(N0, dtype=float)
        if N0.shape != (S,):
            raise ValueError("N0 must have shape (S,) compatible with alpha.")

    def lv_rhs(_t: float, N: np.ndarray) -> np.ndarray:
        """Right-hand side of LV dynamics."""
        N = np.asarray(N, dtype=float).flatten()
        if extinction_threshold > 0.0:
            # Avoid negative densities for numerical stability
            N = np.maximum(N, extinction_threshold)
        return N * (r + alpha @ N)

    sol = solve_ivp(
        lv_rhs,
        (0.0, t_max),
        N0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )

    if not sol.success:
        raise RuntimeError(f"LV integration failed: {sol.message}")

    return sol.t, sol.y


############################################
#       EXACT FITNESS AT EQUILIBRIUM       #
############################################

def evaluate_exact_F(
    X: np.ndarray,
    r: ArrayLike,
    phi: ArrayLike,
    A: ArrayLike,
    K: Optional[ArrayLike] = None,
    clip_negative: bool = False,
) -> np.ndarray:
    """
    Evaluate F(x) = sum_i phi_i * y_i * x_i, where y is the LV equilibrium:

        A y = - (r ⊙ x)

    for each binary configuration x in X. Here ⊙ denotes element-wise product.

    Parameters
    ----------
    X : ndarray, shape (M, S)
        Matrix of binary configurations (M genotypes, S species/traits).
        Each row is a genotype x in {0,1}^S.
    r : array_like, shape (S,)
        Growth rates.
    phi : array_like, shape (S,)
        Weights used to compute the scalar fitness F(x) from the equilibrium y.
    A : array_like, shape (S, S)
        Interaction matrix (must be invertible).
    K : array_like, shape (S,), optional
        Optional carrying capacities or scaling of the equilibrium y.
        If provided, y is multiplied element-wise by K.
        (Currently a simple rescaling; can be adapted to a more
         mechanistic use if needed.)
    clip_negative : bool, default False
        If True, negative components of y are set to zero (interpreting
        them as extinctions).

    Returns
    -------
    F_exact : ndarray, shape (M,)
        Array of F(x) values, one per configuration.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (M, S).")

    M, S = X.shape
    if A.shape != (S, S):
        raise ValueError("A must have shape (S, S) compatible with X.")
    if r.shape != (S,):
        raise ValueError("r must have shape (S,).")
    if phi.shape != (S,):
        raise ValueError("phi must have shape (S,).")

    if K is not None:
        K = np.asarray(K, dtype=float)
        if K.shape != (S,):
            raise ValueError("K must have shape (S,) if provided.")

    # Right-hand side for all genotypes:
    # For each x: b(x) = - (r ⊙ x)
    # Arrange as matrix B of shape (S, M), each column is b(x)
    RX = (X * r[None, :])  # shape (M, S)
    B = -RX.T              # shape (S, M)

    # Solve A Y = B for Y (shape (S, M)), one column per genotype
    # This avoids computing the explicit inverse of A.
    try:
        Y = np.linalg.solve(A, B)  # shape (S, M)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "Interaction matrix A is singular or ill-conditioned; "
            "cannot compute equilibrium y."
        ) from exc

    # Optional scaling by K
    if K is not None:
        Y = Y * K[:, None]

    if clip_negative:
        Y = np.maximum(Y, 0.0)

    # Fitness: F(x) = sum_i phi_i * y_i(x) * x_i
    # Y: (S, M), X.T: (S, M)
    F_exact = np.sum(phi[:, None] * Y * X.T, axis=0)  # shape (M,)

    return F_exact


############################################
#      THIRD-ORDER MECHANISTIC HOI TENSOR   #
############################################

def gaussian_hoi_tensor(
    S: int,
    C: float,
    mu: float,
    sigma: float,
    exclude_self_mediated: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a random third-order interaction tensor B for an extended
    Lotka--Volterra system.

    The term B[i, j, l] represents the effect of the pair (j, l) on the
    per-capita growth rate of species i through B[i, j, l] * N_j * N_l.

    Only entries with j < l are sampled. All other entries are kept at zero,
    so the dynamical term corresponds to:

        sum_{j<l} B[i, j, l] * N_j * N_l.

    Parameters
    ----------
    S : int
        Number of species.
    C : float
        Connectance for third-order terms.
    mu : float
        Mean of the Gaussian distribution for non-zero third-order terms.
    sigma : float
        Standard deviation of the Gaussian distribution for non-zero
        third-order terms.
    exclude_self_mediated : bool, default True
        If True, remove terms where the focal species i is one of the
        mediating species j or l.
    rng : np.random.Generator, optional
        Random number generator. If None, use default_rng().

    Returns
    -------
    B : ndarray, shape (S, S, S)
        Third-order mechanistic interaction tensor.
    """
    if rng is None:
        rng = np.random.default_rng()

    if S <= 0:
        raise ValueError("S must be a positive integer.")
    if not (0.0 <= C <= 1.0):
        raise ValueError("Connectance C must be in [0, 1].")
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")

    B = np.zeros((S, S, S), dtype=float)

    for i in range(S):
        for j in range(S):
            for l in range(j + 1, S):
                if exclude_self_mediated and (i == j or i == l):
                    continue
                if rng.random() < C:
                    B[i, j, l] = rng.normal(mu, sigma)

    return B


############################################
#          EXTENDED LV WITH HOIs            #
############################################

def lv_hoi_rhs(
    _t: float,
    N: np.ndarray,
    r: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    extinction_threshold: float = 0.0,
) -> np.ndarray:
    """
    Right-hand side of an extended Lotka--Volterra model with explicit
    third-order mechanistic interactions:

        dN_i/dt = N_i * (r_i + sum_j A_ij N_j
                         + sum_{j<l} B_ijl N_j N_l).

    The tensor B is assumed to contain non-zero entries only for j < l.
    """
    N = np.asarray(N, dtype=float).flatten()

    if extinction_threshold > 0.0:
        # Avoid negative densities inside the RHS for numerical stability.
        N = np.maximum(N, extinction_threshold)

    pairwise = A @ N
    third_order = np.einsum("ijl,j,l->i", B, N, N)

    return N * (r + pairwise + third_order)


def simulate_lv_hoi(
    A: ArrayLike,
    B: ArrayLike,
    r: ArrayLike,
    t_max: float = 100.0,
    N0: Optional[ArrayLike] = None,
    t_eval: Optional[ArrayLike] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    extinction_threshold: float = 0.0,
    max_step=np.inf,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate extended Lotka--Volterra dynamics with explicit third-order
    mechanistic interactions.

    Dynamics
    --------
        dN_i/dt = N_i * (r_i + sum_j A_ij N_j
                         + sum_{j<l} B_ijl N_j N_l).

    Parameters
    ----------
    A : array_like, shape (S, S)
        Pairwise interaction matrix.
    B : array_like, shape (S, S, S)
        Third-order mechanistic interaction tensor.
    r : array_like, shape (S,)
        Intrinsic growth rates.
    t_max : float, default 100.0
        Final time of integration.
    N0 : array_like, shape (S,), optional
        Initial abundances. If None, random positive values are used.
    t_eval : array_like, optional
        Time points at which to store the computed solution. If None,
        the solver chooses adaptively.
    rtol, atol : float
        Relative and absolute tolerances for the ODE solver.
    extinction_threshold : float, default 0.0
        If > 0, abundances are floored at this value inside the RHS.
    max_step : float, default np.inf
        Maximum step size for the ODE solver.

    Returns
    -------
    t : ndarray, shape (n_times,)
        Time points of the simulation.
    N : ndarray, shape (S, n_times)
        Abundance trajectories.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    r = np.asarray(r, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square (S, S) array.")

    S = A.shape[0]

    if B.shape != (S, S, S):
        raise ValueError("B must have shape (S, S, S) compatible with A.")
    if r.shape != (S,):
        raise ValueError("r must have shape (S,) compatible with A.")

    if N0 is None:
        rng = np.random.default_rng()
        N0 = rng.normal(loc=0.1, scale=0.05, size=S)
        N0 = np.abs(N0)
    else:
        N0 = np.asarray(N0, dtype=float)
        if N0.shape != (S,):
            raise ValueError("N0 must have shape (S,) compatible with A.")

    def rhs(t: float, N: np.ndarray) -> np.ndarray:
        return lv_hoi_rhs(
            t,
            N,
            r,
            A,
            B,
            extinction_threshold=extinction_threshold,
        )

    sol = solve_ivp(
        rhs,
        (0.0, t_max),
        N0,
        method="RK45",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"Extended LV integration failed: {sol.message}")

    return sol.t, sol.y


############################################
#      NUMERICAL FITNESS WITH HOIs          #
############################################

def evaluate_numerical_F_hoi(
    X: np.ndarray,
    r: ArrayLike,
    phi: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    t_max: float = 200.0,
    N0_value: float = 0.1,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_step=np.inf,
    extinction_threshold: float = 0.0,
    clip_negative: bool = True,
    return_equilibria: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Evaluate a community-function landscape induced by an extended LV model
    with explicit third-order terms.

    For each binary composition x, the dynamics are restricted to the
    subcommunity of present species and integrated numerically. The function
    value is:

        F(x) = sum_i phi_i * y_i^*(x) * x_i,

    where y_i^*(x) is the final abundance after integration.

    Parameters
    ----------
    X : ndarray, shape (M, S)
        Binary configurations.
    r : array_like, shape (S,)
        Growth rates.
    phi : array_like, shape (S,)
        Species weights used to compute the scalar community function.
    A : array_like, shape (S, S)
        Pairwise interaction matrix.
    B : array_like, shape (S, S, S)
        Third-order mechanistic interaction tensor.
    t_max : float, default 200.0
        Final integration time for each subcommunity.
    N0_value : float, default 0.1
        Initial abundance assigned to each present species.
    rtol, atol : float
        Relative and absolute tolerances for the ODE solver.
    extinction_threshold : float, default 0.0
        If > 0, abundances are floored at this value inside the RHS.
    clip_negative : bool, default True
        If True, negative final abundances are clipped to zero.
    return_equilibria : bool, default False
        If True, also return the full equilibrium/final abundance matrix.

    Returns
    -------
    F : ndarray, shape (M,)
        Community-function landscape.
    Y_full : ndarray, shape (M, S), optional
        Final abundances for every configuration and species.
    """
    X = np.asarray(X, dtype=float)
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (M, S).")

    M, S = X.shape

    if A.shape != (S, S):
        raise ValueError("A must have shape (S, S) compatible with X.")
    if B.shape != (S, S, S):
        raise ValueError("B must have shape (S, S, S) compatible with X.")
    if r.shape != (S,):
        raise ValueError("r must have shape (S,).")
    if phi.shape != (S,):
        raise ValueError("phi must have shape (S,).")

    F = np.zeros(M, dtype=float)
    Y_full = np.zeros((M, S), dtype=float)

    for m, x in enumerate(X):
        present = x.astype(bool)

        if not np.any(present):
            F[m] = 0.0
            continue

        idx = np.where(present)[0]

        A_sub = A[np.ix_(idx, idx)]
        B_sub = B[np.ix_(idx, idx, idx)]
        r_sub = r[idx]
        phi_sub = phi[idx]

        N0 = np.full(len(idx), N0_value, dtype=float)

        _, traj = simulate_lv_hoi(
            A=A_sub,
            B=B_sub,
            r=r_sub,
            t_max=t_max,
            max_step=max_step,
            N0=N0,
            t_eval=None,
            rtol=rtol,
            atol=atol,
            extinction_threshold=extinction_threshold,
        )

        y_star = traj[:, -1]

        if clip_negative:
            y_star = np.maximum(y_star, 0.0)

        Y_full[m, idx] = y_star
        F[m] = np.sum(phi_sub * y_star)

    if return_equilibria:
        return F, Y_full

    return F


def build_hoi_lv_landscape(
    X: np.ndarray,
    r: ArrayLike,
    phi: ArrayLike,
    A: ArrayLike,
    B: ArrayLike,
    R: int = 5,
    noise_sd: float = 0.0,
    t_max: float = 200.0,
    N0_value: float = 0.1,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    extinction_threshold: float = 0.0,
    clip_negative: bool = True,
    rng: Optional[np.random.Generator] = None,
    feature_names: Optional[list[str]] = None,
    return_meta: bool = True,
    max_step=np.inf,
):
    """
    Build a replicated Landscape from an extended Lotka--Volterra model with
    explicit third-order mechanistic HOIs.

    The true landscape is:

        F_true(x) = sum_i phi_i * y_i^*(x) * x_i.

    Replicates are generated as:

        F_r(x) = F_true(x) + xi_r(x),
        xi_r(x) ~ Normal(0, noise_sd^2).

    Parameters
    ----------
    X : ndarray, shape (M, S)
        Binary configurations.
    r, phi, A, B
        Extended LV parameters.
    R : int, default 5
        Number of synthetic replicates.
    noise_sd : float, default 0.0
        Standard deviation of additive measurement noise.
    return_meta : bool, default True
        If True, return (Landscape, metadata). Otherwise return Landscape.

    Returns
    -------
    Landscape or (Landscape, dict)
        Replicated community-function landscape and optional metadata.
    """
    from .core import Landscape

    if rng is None:
        rng = np.random.default_rng()

    if R <= 0:
        raise ValueError("R must be a positive integer.")
    if noise_sd < 0.0:
        raise ValueError("noise_sd must be non-negative.")

    X = np.asarray(X, dtype=np.int8)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (M, S).")
    if not np.isin(X, [0, 1]).all():
        raise ValueError("X must contain only 0/1 binary states.")

    N = X.shape[1]

    F_true, equilibria = evaluate_numerical_F_hoi(
        X=X,
        r=r,
        phi=phi,
        A=A,
        B=B,
        t_max=t_max,
        N0_value=N0_value,
        rtol=rtol,
        atol=atol,
        extinction_threshold=extinction_threshold,
        clip_negative=clip_negative,
        return_equilibria=True,
        max_step=max_step,
    )

    values = np.tile(F_true[:, None], (1, R))

    if noise_sd > 0.0:
        values = values + rng.normal(0.0, noise_sd, size=values.shape)

    L = Landscape(
        states=X,
        values=values,
        N=N,
        R=R,
        order="lex",
        feature_names=feature_names,
    )

    if not return_meta:
        return L

    meta = {
        "kind": "extended_lv_with_explicit_third_order_terms",
        "N": N,
        "R": R,
        "noise_sd": float(noise_sd),
        "F_true": F_true,
        "equilibria": equilibria,
        "r": np.asarray(r, dtype=float),
        "phi": np.asarray(phi, dtype=float),
        "A": np.asarray(A, dtype=float),
        "B": np.asarray(B, dtype=float),
    }

    return L, meta


###################################
#         AUXILIAR FUNCTION       #
###################################

def growth_rates(N: int,r) -> np.ndarray:
    """
    Return intrinsic growth rates r for a system of size N.

    For now we choose all ones, but this can be customized.
    """
    return r*np.ones(N, dtype=float)


def is_lv_stable(A: np.ndarray) -> bool:
    """
    Check dynamical stability of the LV system with interaction matrix A.

    The criterion used here is:
        max(Re(eig(A))) < 0
    """
    eigs = np.linalg.eigvals(A)
    return np.max(np.real(eigs)) < 0.0


def is_lv_feasible(A: np.ndarray, r: np.ndarray, X: np.ndarray, eps: float = 0.0) -> bool:
    """
    Check feasibility of all non-zero genotypes for the LV equilibrium.

    For each non-zero genotype x:
        A y = - (r ⊙ x)
    we require y_i > eps only for species with x_i = 1 (present).
    """
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return False

    for x in X[1:]:  # skip all-zero genotype
        rx = r * x
        y = -A_inv @ rx

        # Only check species that are present in x
        present = x.astype(bool)
        if np.any(y[present] <= eps):
            return False

    return True

