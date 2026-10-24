import numpy as np
from scipy.stats import t as tdist

PARAMS = {"win": 60, "tilt": 0.05, "mom": 20, "df": 4.0}


def predict(history):
    c = history.close
    r = np.diff(np.log(c[-(PARAMS["win"] + 1):]))
    sd = r.std() if len(r) > 5 else 0.05
    m = np.diff(np.log(c[-(PARAMS["mom"] + 1):])).mean()
    df = PARAMS["df"]
    loc = PARAMS["tilt"] * np.tanh(m / max(sd, 1e-6)) * sd
    scale = max(sd, 1e-4) * np.sqrt((df - 2) / df)
    cdf = tdist.cdf((history.edges - loc) / scale, df)
    return np.diff(np.r_[0.0, cdf, 1.0])
