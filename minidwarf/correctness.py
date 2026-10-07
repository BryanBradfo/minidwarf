# SPDX-License-Identifier: Apache-2.0
import numpy as np


def check_correct(actual, expected, rtol, atol) -> bool:
    """Return True if every actual output array matches the expected array within rtol/atol."""
    if len(actual) != len(expected):
        return False
    return all(np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=False)
               for a, e in zip(actual, expected))


def compare(actual, expected, rtol, atol) -> dict:
    """Per-check detail: pass/fail, max abs/rel error, tol_ratio = max |a-e| / (atol + rtol|e|), NaN count."""
    if len(actual) != len(expected) or any(np.shape(a) != np.shape(e) for a, e in zip(actual, expected)):
        return {"passed": False, "max_abs_err": None, "max_rel_err": None, "tol_ratio": None, "nan_count": None}
    abs_err = rel_err = ratio = 0.0; nans = 0
    for a, e in zip(actual, expected):
        a = np.asarray(a, np.float64); e = np.asarray(e, np.float64)
        nans += int(np.isnan(a).sum())
        d = np.abs(a - e); d = np.where(np.isnan(d), np.inf, d)
        if d.size == 0: continue
        tol = atol + rtol * np.abs(e)
        abs_err = max(abs_err, float(d.max()))
        rel_err = max(rel_err, float((d / np.maximum(np.abs(e), max(atol, 1e-30))).max()))
        ratio = max(ratio, float((d / np.where(tol > 0, tol, 1e-30)).max()))
    return {"passed": check_correct(actual, expected, rtol, atol), "max_abs_err": abs_err,
            "max_rel_err": rel_err, "tol_ratio": ratio, "nan_count": nans}
