"""Online recalibration: adaptive-conformal-style update (Gibbs & Candes) of a tempering exponent on the 21-bin forecast."""
import numpy as np

ALPHA, GAMMA, KGAIN = 0.10, 0.02, 3.0


class Recalibrated:
    """Wrap predict fn. After each outcome, check whether it fell inside the predicted central 90% interval and update
    alpha_t += GAMMA*(ALPHA - err). Undercoverage (alpha_t < ALPHA) flattens the forecast (p**kappa, kappa<1); overcoverage sharpens it."""

    def __init__(self, fn):
        self.fn, self.st = fn, {}

    def __call__(self, h):
        s = self.st.get(h.coin)
        a = ALPHA
        if s is not None and s["t"] == h.t - 1:
            b = int(np.searchsorted(h.edges, np.log(h.close[-1] / h.close[-2])))
            cum = np.cumsum(s["p"])
            lo_c = cum[b - 1] if b > 0 else 0.0
            inside = (lo_c < 1 - ALPHA / 2) and (cum[b] > ALPHA / 2)
            a = float(np.clip(s["a"] + GAMMA * (ALPHA - (0.0 if inside else 1.0)), 0.01, 0.5))
        else:
            a = ALPHA
        raw = np.asarray(self.fn(h), float)
        kappa = float(np.clip(np.exp(KGAIN * (a - ALPHA) / ALPHA), 0.5, 2.0))
        p = np.clip(raw, 1e-12, None) ** kappa
        p /= p.sum()
        self.st[h.coin] = dict(t=h.t, p=raw / raw.sum(), a=a)
        return p


def coverage(P, O, elig, alpha=ALPHA):
    """Share of outcomes inside the central (1-alpha) interval and in each tail, from stored forecasts."""
    cum = np.cumsum(P, axis=2)
    cb = np.take_along_axis(cum, O[..., None], axis=2)[..., 0]
    lo = np.where(O > 0, np.take_along_axis(cum, np.maximum(O - 1, 0)[..., None], axis=2)[..., 0], 0.0)
    inside = (lo < 1 - alpha / 2) & (cb > alpha / 2)
    low_tail = cb <= alpha / 2
    up_tail = lo >= 1 - alpha / 2
    e = elig
    return dict(inside90=float(inside[e].mean()), low_tail5=float(low_tail[e].mean()), up_tail5=float(up_tail[e].mean()))
