"""Shared entity-season unique-signal transfer utilities for CPU audits.

The unit used for reconstructibility is an entity-season profile.  Candidate
profiles are reconstructed from the mean champion numeric feature vector with
source OOF Ridge and source-fitted/frozen target apply.  The surviving profile
is mapped back to pitch rows, where one fixed Ridge direction is fitted against
the source champion residual.  The matched null repeats the whole entity
permutation, reconstruction and residual-direction procedure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


ALPHA = 100.0
REPS = 400


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _ridge_pair(xs, xt, ys):
    med = np.nanmedian(xs, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    xs = np.where(np.isfinite(xs), xs, med)
    xt = np.where(np.isfinite(xt), xt, med)
    sc = StandardScaler().fit(xs)
    model = Ridge(alpha=ALPHA).fit(sc.transform(xs), ys)
    return model.predict(sc.transform(xs)), model.predict(sc.transform(xt))


def residualise_profiles(x_source, x_target, z_source, z_target):
    """Return source OOF and target-frozen unique components."""
    h_oof, h_target = reconstruction_operators(x_source, x_target)
    return apply_reconstruction(h_oof, h_target, z_source, z_target)


def reconstruction_operators(x_source, x_target):
    """Precompute the exact linear OOF/frozen Ridge prediction operators.

    StandardScaler + Ridge is linear in the response.  Matched-null iterations
    permute only the response profiles, so refitting thousands of identical
    design matrices is wasteful.  These matrices reproduce the same fit,
    including Ridge's unpenalised intercept, and make the 400-repetition null a
    matrix multiplication rather than 2,400 separate solves per boundary.
    """
    xs, xt = np.asarray(x_source, float), np.asarray(x_target, float)
    medx = np.nanmedian(xs, axis=0)
    medx = np.where(np.isfinite(medx), medx, 0.0)
    xs = np.where(np.isfinite(xs), xs, medx)
    xt = np.where(np.isfinite(xt), xt, medx)
    kf = KFold(5, shuffle=True, random_state=20260827)
    n = len(xs)
    h_oof = np.zeros((n, n), float)
    for train, test in kf.split(xs):
        sc = StandardScaler().fit(xs[train])
        a, b = sc.transform(xs[train]), sc.transform(xs[test])
        coef = np.linalg.solve(a.T @ a + ALPHA*np.eye(a.shape[1]), a.T)
        h_oof[np.ix_(test, train)] = b @ coef + 1.0/len(train)
    sc = StandardScaler().fit(xs)
    a, b = sc.transform(xs), sc.transform(xt)
    coef = np.linalg.solve(a.T @ a + ALPHA*np.eye(a.shape[1]), a.T)
    h_target = b @ coef + 1.0/n
    return h_oof, h_target


def apply_reconstruction(h_oof, h_target, z_source, z_target):
    zs, zt = np.asarray(z_source, float), np.asarray(z_target, float)
    medz = np.nanmedian(zs, axis=0)
    medz = np.where(np.isfinite(medz), medz, 0.0)
    zs = np.where(np.isfinite(zs), zs, medz)
    zt = np.where(np.isfinite(zt), zt, medz)
    oof = h_oof @ zs
    target_hat = h_target @ zs
    return zs-oof, zt-target_hat, oof


def _map_profiles(rows, ids, values, entity):
    tab = pd.DataFrame(values, index=pd.Index(ids, name=entity))
    return rows[[entity]].join(tab, on=entity).iloc[:, 1:].to_numpy(float)


def _direction(xs, xt, rs):
    fit, add = _ridge_pair(xs, xt, rs)
    return fit - np.nanmean(fit), add - np.nanmean(fit)


def _aligned_profiles(source_profile, target_profile, source_champion,
                      target_champion):
    ids = np.intersect1d(
        np.intersect1d(source_profile.index, target_profile.index),
        np.intersect1d(source_champion.index, target_champion.index))
    if len(ids) < 30:
        raise RuntimeError(f"too few common entity profiles: {len(ids)}")
    return (ids, source_profile.loc[ids].to_numpy(float),
            target_profile.loc[ids].to_numpy(float),
            source_champion.loc[ids].to_numpy(float),
            target_champion.loc[ids].to_numpy(float))


def audit(source_rows, target_rows, source_profile, target_profile,
          source_champion, target_champion, source_resid, target_resid,
          entity, reps=REPS):
    """Run novelty, unique transfer and the complete matched entity null."""
    ids, zs, zt, cxs, cxt = _aligned_profiles(
        source_profile, target_profile, source_champion, target_champion)
    h_oof, h_target = reconstruction_operators(cxs, cxt)
    us, ut, recon = apply_reconstruction(h_oof, h_target, zs, zt)
    src_u = _map_profiles(source_rows, ids, us, entity)
    tgt_u = _map_profiles(target_rows, ids, ut, entity)
    src_u = np.nan_to_num(src_u)
    tgt_u = np.nan_to_num(tgt_u)
    fit, add = _direction(src_u, tgt_u, source_resid)
    rho = safe_corr(add, target_resid)

    null = np.empty(reps, float)
    for k in range(reps):
        rng = np.random.default_rng(20260827 + k)
        p = rng.permutation(len(ids))
        nus, nut, _ = apply_reconstruction(h_oof, h_target, zs[p], zt[p])
        nxs = np.nan_to_num(_map_profiles(source_rows, ids, nus, entity))
        nxt = np.nan_to_num(_map_profiles(target_rows, ids, nut, entity))
        _, nadd = _direction(nxs, nxt, source_resid)
        null[k] = abs(safe_corr(nadd, target_resid))

    var_z = float(np.nanvar(zs))
    var_u = float(np.nanvar(us))
    masks = {
        "R": target_rows.game_type.eq("R").to_numpy(),
        "F": target_rows.game_type.eq("F").to_numpy(),
        "early": target_rows.game_month.le(6).to_numpy(),
        "late": target_rows.game_month.gt(6).to_numpy(),
    }
    return {
        "n_common_entities": int(len(ids)),
        "candidate_variance": var_z,
        "unique_variance": var_u,
        "unique_variance_share": var_u / var_z if var_z > 0 else 0.0,
        "reconstructibility_r2": float(r2_score(zs, recon)),
        "rho_source": safe_corr(fit, source_resid),
        "rho": rho,
        "null_percentile": float(np.mean(null <= abs(rho)) * 100),
        "null_p95": float(np.quantile(null, .95)),
        "null_p99": float(np.quantile(null, .99)),
        "segments": {k: safe_corr(add[m], target_resid[m])
                     for k, m in masks.items()},
    }
