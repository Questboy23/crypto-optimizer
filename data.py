"""Kraken daily USD data, monthly universe, splits, coin groups, and the holdout lock."""
import inspect, os, numpy as np, polars as pl
from functools import lru_cache

RAW = os.path.join(os.path.dirname(__file__), "data", "kraken_daily_usd.parquet")
FIAT = {"EUR", "GBP", "AUD", "CAD", "CHF", "JPY", "AED", "USD"}
STABLE = {"USDT", "USDC", "DAI", "PYUSD", "TUSD", "USDD", "EURT", "USDG", "RLUSD", "FDUSD", "UST", "USTC", "PAX", "GUSD", "BUSD", "EURR", "EURQ", "USDQ", "USDR", "USDS", "AUDX", "USD1"}
N_UNIVERSE = 30
GAP_MONTHS, HOLDOUT_MONTHS = 6, 12


@lru_cache(1)
def _raw() -> pl.DataFrame:
    df = pl.read_parquet(RAW).with_columns(pl.col("pair").str.replace(r"USD$", "").alias("coin"))
    df = df.filter(~pl.col("coin").is_in(list(FIAT | STABLE)) & (pl.col("coin") != ""))
    # drop pegged assets the name list misses: >90% of closes within 3% of $1 with 5+ years... use dispersion
    peg = (df.group_by("coin").agg(((pl.col("close") - 1).abs() < 0.03).mean().alias("f"), pl.len().alias("n"))
             .filter((pl.col("f") > 0.9) & (pl.col("n") > 200))["coin"].to_list())
    df = df.filter(~pl.col("coin").is_in(peg))
    # normalize ticker aliases that appear twice
    return df.sort(["coin", "date"])


def end_date():
    return _raw()["date"].max()


def split_dates():
    from dateutil.relativedelta import relativedelta
    end = end_date().replace(hour=0, minute=0)
    hold_start = (end + relativedelta(days=1)) - relativedelta(months=HOLDOUT_MONTHS)
    dev_end = hold_start - relativedelta(months=GAP_MONTHS)
    return dev_end, hold_start  # dev: date < dev_end; gap: dev_end <= date < hold_start; holdout: >= hold_start


def _caller_is_final_report() -> bool:
    return any(os.path.basename(f.filename) == "final_report.py" for f in inspect.stack()[2:])


def panel(allow_holdout=False):
    """Wide close/volume panels indexed by every calendar day (ffill price, 0 volume on no-trade days)."""
    dev_end, hold_start = split_dates()
    if allow_holdout and not _caller_is_final_report():
        raise PermissionError("holdout is readable only by final_report.py")
    df = _raw()
    if not allow_holdout:
        df = df.filter(pl.col("date") < hold_start)
    px = df.pivot(index="date", on="coin", values="close").sort("date")
    vol = df.with_columns((pl.col("close") * pl.col("volume")).alias("dv")).pivot(index="date", on="coin", values="dv").sort("date")
    days = pl.DataFrame({"date": pl.datetime_range(px["date"].min(), px["date"].max(), "1d", eager=True)})
    px = days.join(px, on="date", how="left").with_columns(pl.exclude("date").forward_fill())
    vol = days.join(vol, on="date", how="left").with_columns(pl.exclude("date").fill_null(0.0))
    return px, vol


@lru_cache(1)
def universe_by_month():
    """{month_start: [coins]}: top N by trailing-30d dollar volume using only days before month start."""
    px, vol = panel(allow_holdout=False)
    d = vol["date"].to_numpy()
    V = vol.drop("date").to_numpy()
    coins = np.array(vol.columns[1:])
    out = {}
    months = sorted({np.datetime64(x, "M") for x in d})
    for m in months:
        start = np.datetime64(m, "D")
        mask = (d >= start - np.timedelta64(30, "D")) & (d < start)
        if mask.sum() < 25:
            continue
        s = np.nan_to_num(V[mask].sum(axis=0))
        k = np.argsort(-s)[:N_UNIVERSE]
        out[str(start)] = [c for c, v in zip(coins[k], s[k]) if v > 0]
    return out


def coin_groups(seed=0, k=4):
    """k disjoint random groups from coins ever in the development universe, excluding BTC (XBT) and ETH."""
    ever = sorted({c for v in universe_by_month().values() for c in v} - {"XBT", "ETH", "XXBT", "XETH"})
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(ever))
    return [sorted(perm[i::k]) for i in range(k)]
