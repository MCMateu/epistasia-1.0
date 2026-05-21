##########################################
#          REQUIRED LIBRARIES            #
##########################################
from __future__ import annotations

import os
import numpy as np
from math import comb
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from cmdstanpy import CmdStanModel
import pandas as pd

if TYPE_CHECKING:
    from .core import Landscape  # only for typing, no runtime import


######################################################
#               STATSITIC / NOISE MODULE             #
######################################################


######################################################
#              BATCH EFFECT CORRECTION               #
######################################################

# -------------------- Stan compile (simple cache) --------------------

# -------------------- Stan compile (simple cache) --------------------

_MODEL_CACHE: Dict[str, CmdStanModel] = {}

def _stan_model() -> CmdStanModel:
    """
    Compile once and cache the batch_effect Stan model placed under epistasia/sn/.
    This assumes batch_effect.stan implements the LONG-format data interface (K, m, r, y).
    """
    key = "batch_effect"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    here = os.path.dirname(__file__)
    stan_file = os.path.join(here, "batch_effect.stan")
    if not os.path.exists(stan_file):
        raise FileNotFoundError(
            f"Stan file not found at: {stan_file}. "
            "Place 'batch_effect.stan' under 'epistasia/sn/'."
        )

    model = CmdStanModel(stan_file=stan_file)
    _MODEL_CACHE[key] = model
    return model


# -------------------- Public API --------------------

def correct_batch_effect(
    L,  # Landscape
    lambda_a: float = 1.0,   # kept for backward compatibility (unused)
    lambda_b: float = 1.0,   # kept for backward compatibility (unused)
    return_posteriors: bool = False,
    seed: Optional[int] = None,
    chains: int = 4,
    parallel_chains: Optional[int] = None,
    iter_warmup: int = 1000,
    iter_sampling: int = 1000,
    link: str = "log",
    log_epsilon: float = 1e-3,
    adjust_zero_state: bool = False,
    keep_missing_as_nan: bool = True,
    **sample_kwargs: Any,
):
    """
    Remove batch effects using a hierarchical Bayesian model defined on observed entries.

    LONG data model (Stan side)
    ---------------------------
        y[k] ~ Normal(F_bar[m[k]] + a[r[k]], sigma * b[r[k]])

    Notes
    -----
    - Allows NaNs in L.values; only non-NaN entries are used for inference.
    - Correction is applied only to observed entries; missing entries remain NaN
      unless keep_missing_as_nan=False (then they get imputed with F_bar_hat).
     - Negative values smaller than neg_frac × median(F_mean) are treated as measurement noise and clipped to zero before log transform.
    """
    # Local import to avoid circular import during package initialization
    from ..core import Landscape  # :contentReference[oaicite:1]{index=1}

    # --- Extract raw matrix (X, R) ---
    F = np.asarray(L.values, dtype=float)
    if F.ndim != 2:
        raise ValueError("Landscape.values must be a 2D array of shape (X, R).")
    X, R = F.shape

    states = np.asarray(L.states)
    if states.shape[0] != X:
        raise ValueError("L.states and L.values have inconsistent number of rows.")

    # --- Identify all-zero state and store original values (optional restore) ---
    zero_state_mask = np.all(states == 0, axis=1)
    # "Exactly zero in all replicates" but allowing NaNs: all observed entries are 0,
    # and there is at least one observed entry.
    if np.any(zero_state_mask):
        zrow = F[zero_state_mask, :]  # usually shape (1, R)
        zobs = ~np.isnan(zrow)
        if np.any(zobs) and np.all(zrow[zobs] == 0.0):
            F_zero_original = zrow.copy()
        else:
            F_zero_original = None
    else:
        F_zero_original = None

    # --- Optional link transform ---
    if link not in ("identity", "log"):
        raise ValueError("link must be 'identity' or 'log'.")

    elif link == "log":
        
        neg_frac= 5e-2       
        usual_q= 0.5    #median

        if log_epsilon <= 0:
            raise ValueError("log_epsilon must be > 0 when link='log'.")
    
        F_work = F.copy()
    
        # --- define "usual scale" from Landscape object ---
        F_mean = L.mean_over_replicates()  # (X,) uses nanmean internally
        F_mean_pos = F_mean[(F_mean > 0) & ~np.isnan(F_mean)]
    
        if F_mean_pos.size == 0:
            raise ValueError(
                "link='log' requires some positive mean values to define a usual scale."
            )
    
        usual_scale = float(np.quantile(F_mean_pos, usual_q))
        neg_epsilon = neg_frac * usual_scale
    
        # --- handle negatives in raw data ---
        neg_mask = (F_work < 0) & ~np.isnan(F_work)
    
        if np.any(neg_mask):
            min_neg = float(np.min(F_work[neg_mask]))
    
            if min_neg < -neg_epsilon:
                raise ValueError(
                    f"Found F < -neg_epsilon (min={min_neg:g}). "
                    f"Computed neg_epsilon={neg_epsilon:g} "
                    f"from usual_scale={usual_scale:g} "
                    f"(neg_frac={neg_frac:g}, usual_q={usual_q:g})."
                )
    
            # Clip small negatives to 0
            F_work[neg_mask] = 0.0
    
        F_model = np.log(F_work + log_epsilon)
    else:
        F_model = F


    # --- Prepare data for Stan (LONG format: ignores NaNs) ---
    obs_mask = ~np.isnan(F_model)              # (X, R)
    m_idx, r_idx = np.where(obs_mask)          # 0-based indices
    K = int(obs_mask.sum())
    if K == 0:
        raise ValueError("All entries are NaN: nothing to fit (K=0).")

    y = F_model[m_idx, r_idx].astype(float)

    data = {
        "M": int(X),
        "R": int(R),
        "K": int(K),
        "m": (m_idx + 1).astype(int),          # 1-based for Stan
        "r": (r_idx + 1).astype(int),          # 1-based for Stan
        "y": y,
    }

    model = _stan_model()
    if parallel_chains is None:
        parallel_chains = min(chains, os.cpu_count() or 1)

    fit = model.sample(
        data=data,
        seed=seed,
        chains=chains,
        parallel_chains=parallel_chains,
        iter_warmup=iter_warmup,
        iter_sampling=iter_sampling,
        **sample_kwargs,
    )

    # --- Extract posterior draws ---
    a_draws = fit.stan_variable("a")               # (draws, R)
    b_draws = fit.stan_variable("b")               # (draws, R)
    F_bar_draws = fit.stan_variable("F_bar")       # (draws, X)
    log_sigma_draws = fit.stan_variable("log_sigma")

    a_bar = a_draws.mean(axis=0)                   # (R,)
    b_bar = b_draws.mean(axis=0)                   # (R,)
    F_bar_hat = F_bar_draws.mean(axis=0)           # (X,)
    sigma_draws = np.exp(log_sigma_draws)

    # --- Batch-effect correction in model space (only observed entries) ---
    # For observed (x,r):
    #   residual_pi_hat = (F_model[x,r] - F_bar_hat[x] - a_bar[r]) / b_bar[r]
    #   F_model_adj[x,r] = F_bar_hat[x] + residual_pi_hat
    if keep_missing_as_nan:
        F_model_adj = np.full_like(F_model, np.nan, dtype=float)
    else:
        # impute missing with F_bar_hat in model space (no replicate-specific effect)
        F_model_adj = F_bar_hat[:, None].astype(float)

    F_model_adj[obs_mask] = (
        F_bar_hat[m_idx]
        + (F_model[obs_mask] - F_bar_hat[m_idx] - a_bar[r_idx]) / b_bar[r_idx]
    )

    # --- Inverse transform to original scale ---
    if link == "log":
        # Inverse of log(F + eps) is exp(.) - eps
        F_adj = np.exp(F_model_adj) - log_epsilon
        # Numeric safety: tiny negatives -> 0
        F_adj = np.where(np.isnan(F_adj), np.nan, np.clip(F_adj, 0.0, None))
    else:
        F_adj = F_model_adj

    # --- Optionally keep the all-zero state fixed at its original value ---
    if (F_zero_original is not None) and (not adjust_zero_state):
        F_adj[zero_state_mask, :] = F_zero_original

    # --- Build corrected Landscape (preserve metadata) ---
    L_corr = Landscape(
        L.states,
        F_adj,
        N=getattr(L, "N", None),
        R=getattr(L, "R", R),
        order=getattr(L, "order", "lex"),
        feature_names=getattr(L, "feature_names", None),
    )

    if return_posteriors:
        post = {
            "a": a_draws,
            "b": b_draws,
            "F_bar": F_bar_draws,
            "log_sigma": log_sigma_draws,
            "sigma": sigma_draws,
        }
        return L_corr, post

    return L_corr



######################################################
#                     HELPERS                        # 
######################################################

def residuals_and_covariance(L: Landscape) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute replicate-level residuals P and the empirical noise covariance \hat C.

    P_{x,r} = \tilde F_r(x) - \hat F(x),    \hat F(x) = mean_r \tilde F_r(x)
    C_hat = (P @ P.T) / (R - 1)

    Requires
    --------
    L.values shape (M,R). NaNs are allowed; rows with any NaN are skipped
    for covariance to avoid bias.

    Returns
    -------
    P : np.ndarray, shape (M_clean, R)
        Residuals for rows without NaNs.
    C_hat : np.ndarray, shape (M_clean, M_clean)
        Empirical covariance across states (rows correspond to the kept states).
    """
    # For stable covariance, drop rows with any NaNs
    Lc = L.drop_rows_with_any_nan()
    V = Lc.values  # (M_clean, R)
    Fhat = np.mean(V, axis=1, keepdims=True)  # (M_clean,1)
    P = V - Fhat
    R = Lc.R
    if R < 2:
        raise ValueError("At least 2 replicates required to estimate covariance.")
    C_hat = (P @ P.T) / (R - 1)
    return P, C_hat

######################

def _phi_matrix(states: np.ndarray, N: int) -> np.ndarray:
    """
    Build the matrix Phi of size (M, 2^N) whose columns are Walsh-like
    basis functions φ_s(x) = ∏_{i: s_i=1} z_i(x),
    with shifted spins z_i in {-1, +1} defined from binary states {0,1} by:
        z_i = +1 if state_i == 1, and -1 otherwise.

    Parameters
    ----------
    states : (M, N) array of {0,1}
        Observed genotypes/configurations.
    N : int
        Number of loci (bits).

    Returns
    -------
    Phi : (M, 2^N) float array
        Column s corresponds to φ_s evaluated on all rows x in 'states'.
        Lexicographic order of s (0..2^N-1) is assumed.
    """
    # Convert to spins in {-1, +1}
    Z = np.where(states == 1, 1.0, -1.0)  # (M, N)
    M = states.shape[0]
    K = 1 << N

    # Pre-allocate Phi
    Phi = np.empty((M, K), dtype=float)

    # Column s = 0 corresponds to the constant function φ_∅(x) = 1
    Phi[:, 0] = 1.0

    # For efficiency, accumulate products by reusing previously computed columns:
    # each s can be built from s' removing its least significant 1-bit.
    for s in range(1, K):
        # isolate least significant 1-bit
        lsb = s & -s
        s_prev = s ^ lsb
        # position of that bit (0-based from the right). We want index in [0..N-1] from left.
        bit_pos_from_right = (lsb.bit_length() - 1)
        i = N - 1 - bit_pos_from_right
        Phi[:, s] = Phi[:, s_prev] * Z[:, i]

    return Phi

########################################################
#                   NOISE PROJECTIONS                  #
########################################################

###################################
# Core projection (pure function) #
###################################

def project_noise_var_WH(
    C_hat: np.ndarray,
    states: np.ndarray,
    N: int,
    orders: np.ndarray,
) -> Tuple[np.ndarray, Dict[int, float]]:
    """
    Project the replicate covariance C_hat onto the Walsh-Hadamard (WH) basis.

    It computes, for each WH mode s (s=0..2^N-1),
        Var[η_s] = 2^{-2N} * φ_s^T C_hat φ_s,
    where φ_s is the column of Phi corresponding to mode s,
    and returns order-averaged K_S = <Var[η_s]>_{|s|=S} for S>=1.

    Parameters
    ----------
    C_hat : (M, M) float array
        Empirical covariance of replicate residuals across the M observed states.
        Must be aligned with 'states' rows (i.e., computed after dropping NaNs).
    states : (M, N) int/bool array in {0,1}
        The observed states used to build C_hat.
    N : int
        Number of loci (bits).
    orders : (2^N,) int array
        Hamming weight |s| for each WH mode index s in [0..2^N-1].
        (Use the same ordering as your WH transform.)

    Returns
    -------
    var_eta_s : (2^N,) float array
        Per-mode noise variance Var[η_s].
    K_S : dict[int, float]
        Order-averaged noise variance per order S (S>=1).
        K_S[S] = mean_s Var[η_s] over all s with |s|=S.
    """
    # Build Phi (M x 2^N)
    Phi = _phi_matrix(states, N)  # columns are φ_s(x)

    # Quadratic form diag( Phi^T C_hat Phi ) without forming the full product:
    # q_s = φ_s^T C_hat φ_s
    # einsum "ms,mt,st->s" uses:
    #  - ms: Phi
    #  - mt: Phi
    #  - st: C_hat  (s,t here denote state indices; name clash with 's' (mode) is harmless)
    q = np.einsum("ms,mt,st->s", Phi, Phi, C_hat, optimize=True)

    # Scale by 2^{-2N}
    var_eta_s = q / float(1 << (2 * N))

    # Order-averaged K_S for S >= 1 (exclude S=0 / constant)
    K_S: Dict[int, float] = {}
    unique_orders = np.unique(orders)
    for S in unique_orders:
        if S == 0:
            continue
        mask = (orders == S)
        # numeric safety if mask is empty
        if np.any(mask):
            K_S[int(S)] = float(np.mean(var_eta_s[mask]))

    return var_eta_s, K_S

##################################
#    Correct variance spectrum   #
##################################

def correct_variance_spectrum(V_S_obs: dict[int, np.ndarray], K_S: dict[int, float], N: int):
    """
    Return V_bar(S) ≈ V_obs(S) - C(N,S) * K_S, clipped at 0.
    """
    V_corr = {}
    for S, v in V_S_obs.items():
        noise = comb(N, S) * K_S.get(S, 0.0)
        V_corr[S] = np.clip(v - noise, a_min=0.0, a_max=None)
    return V_corr

###################################################################
#                           BOOTSTRAPS                            #
###################################################################

def bootstrap_null(
    F_df,
    B=1000,
    multipliers="normal",     # {"rademacher","normal"}
    preserve_residual_corr=False,      # True  -> keep cross-state correlations (wild-cluster)
                                  # False -> break correlations across states (iid-by-state)
    rng=None,
):
    """
    Null bootstrap for full fitness landscape (mean-zero synthetic landscapes).

    Concept
    -------
    Builds mean-zero bootstrap landscapes F^(b) = beta^(b) from replicate-level residuals.
    - If preserve_residual_corr=True: PRESERVES empirical cross-state covariance (wild-cluster).
    - If preserve_residual_corr=False: DESTROYS cross-state correlations (i.i.d. across states) by
      reshuffling residuals row-wise (per replicate) before forming the wild perturbation.

    Parameters
    ----------
    F_df : pandas.DataFrame
        First N columns = binary states, remaining columns = replicate fitness values.
    B : int
        Number of bootstrap draws.
    multipliers : {"rademacher","normal"}
        Type of bootstrap multipliers. R elif link == "log":
             
             neg_frac: float = 1e-3,          # 0.1% = 1e-3
             usual_q: float = 0.5,            # mediana como escala "usual"

             
             if log_epsilon <= 0:
                 raise ValueError("log_epsilon must be > 0 when link='log'.")
         
             F_work = F.copy()
         
             # --- define "usual scale" from Landscape object ---
             F_mean = L.mean_over_replicates()  # (X,) uses nanmean internally
             F_mean_pos = F_mean[(F_mean > 0) & ~np.isnan(F_mean)]
         
             if F_mean_pos.size == 0:
                 raise ValueError(
                     "link='log' requires some positive mean values to define a usual scale."
                 )
         
             usual_scale = float(np.quantile(F_mean_pos, usual_q))
             neg_epsilon = neg_frac * usual_scale
         
             # --- handle negatives in raw data ---
             neg_mask = (F_work < 0) & ~np.isnan(F_work)
         
             if np.any(neg_mask):
                 min_neg = float(np.min(F_work[neg_mask]))
         
                 if min_neg < -neg_epsilon:
                     raise ValueError(
                         f"Found F < -neg_epsilon (min={min_neg:g}). "
                         f"Computed neg_epsilon={neg_epsilon:g} "
                         f"from usual_scale={usual_scale:g} "
                         f"(neg_frac={neg_frac:g}, usual_q={usual_q:g})."
                     )
         
                 # Clip small negatives to 0
                 F_work[neg_mask] = 0.0
         
             F_model = np.log(F_work + log_epsilon)
         else:
             F_model = Fademacher is robust to asymmetries.
    preserve_residual_corr : bool
        Whether to preserve biological/experimental correlations across states.
    rng : np.random.Generator or None
        Random generator. If None, a new Generator is created.

    Returns
    -------
    F_boot : pandas.DataFrame
        First N columns identical to F_df.
        Then B columns = bootstrap replicates of the NULL landscape (mean zero).
    """
    if rng is None:
        rng = np.random.default_rng()

    vals = F_df.values
    M, C = vals.shape

    N = 0
    for j in range(C):
        if np.isin(vals[:, j], [0, 1]).all():
            N += 1
        else:
            break

    X = vals[:, N:].astype(float)
    R = X.shape[1]
    if R < 2:
        raise ValueError("At least 2 replicates required for wild bootstrap.")

    F_bar = np.nanmean(X, axis=1, keepdims=True)

    if not np.isfinite(F_bar).all():
        raise ValueError("Some states have no valid replicates.")

    Xi = X - F_bar
    valid = np.isfinite(Xi)
    n_valid = valid.sum(axis=1)

    if np.any(n_valid < 2):
        raise ValueError("Every state needs at least 2 valid replicates for null bootstrap.")

    Xi0 = np.where(valid, Xi, 0.0)

    F_boot_mat = np.empty((M, B), dtype=float)

    for b in range(B):
        if multipliers == "normal":
            w = rng.standard_normal(R)
        elif multipliers == "rademacher":
            w = rng.choice([-1.0, 1.0], size=R)
        else:
            raise ValueError("multipliers must be 'normal' or 'rademacher'")

        if preserve_residual_corr:
            Xi_eff = Xi0
            denom = np.sqrt(n_valid * (n_valid - 1))
        else:
            Xi_eff = np.empty_like(Xi0)
            for r in range(R):
                col = Xi[:, r]
                col_valid = col[np.isfinite(col)]
                if col_valid.size == 0:
                    Xi_eff[:, r] = 0.0
                else:
                    Xi_eff[:, r] = rng.choice(col_valid, size=M, replace=True)

            denom = np.sqrt(R * (R - 1))

        beta = (Xi_eff @ w) / denom
        beta -= np.nanmean(beta)

        F_boot_mat[:, b] = beta

    state_cols = list(F_df.columns[:N])
    boot_cols = [f"boot_{b+1}" for b in range(B)]

    return pd.DataFrame(
        np.hstack([vals[:, :N], F_boot_mat]),
        columns=state_cols + boot_cols,
    )


def bootstrap_uncertainty(
    F_df,
    B=1000,
    multipliers="normal",
    flavor="iid",
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    vals = F_df.values
    M, C = vals.shape

    N = 0
    for j in range(C):
        if np.isin(vals[:, j], [0, 1]).all():
            N += 1
        else:
            break

    X = vals[:, N:].astype(float)
    R = X.shape[1]

    n_valid = np.sum(np.isfinite(X), axis=1)
    if np.any(n_valid == 0):
        raise ValueError("Some states have no valid replicates.")

    Fhat = np.nanmean(X, axis=1)
    Xi = X - Fhat[:, None]
    Xi[~np.isfinite(Xi)] = 0.0

    F_boot_mat = np.empty((M, B), dtype=float)

    for b in range(B):
        if multipliers == "normal":
            W = rng.standard_normal((M, R))
        elif multipliers == "rademacher":
            W = rng.choice([-1.0, 1.0], size=(M, R))
        else:
            raise ValueError("multipliers must be 'normal' or 'rademacher'")

        if flavor == "wildcluster":
            # Same multiplier per replicate, preserving replicate-level correlation
            if multipliers == "normal":
                w = rng.standard_normal(R)
            else:
                w = rng.choice([-1.0, 1.0], size=R)

            beta = np.sum(Xi * w[None, :], axis=1)
            denom = np.sqrt(np.maximum(n_valid * (n_valid - 1), 1))

        elif flavor == "iid":
            # Independent uncertainty per state
            beta = np.sum(Xi * W, axis=1)
            denom = np.sqrt(np.maximum(n_valid * (n_valid - 1), 1))

        else:
            raise ValueError("flavor must be 'iid' or 'wildcluster'")

        beta = beta / denom

        # States with only one valid replicate have no estimable residual variance
        beta[n_valid < 2] = 0.0

        F_b = Fhat + beta

        F_boot_mat[:, b] = F_b

    state_cols = list(F_df.columns[:N])
    boot_cols = [f"boot_{b+1}" for b in range(B)]

    return pd.DataFrame(
        np.hstack([vals[:, :N], F_boot_mat]),
        columns=state_cols + boot_cols,
    )
