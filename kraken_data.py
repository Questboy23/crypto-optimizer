"""Read only the daily (1440) CSVs out of Kraken's bulk OHLCVT zips via HTTP range requests (disk is full; zips are ~10 GB)."""
import io, zipfile, sys, requests, polars as pl
B = "https://assets.kraken.com/marketing/institutions/"
PARTS = [(f"{B}Kraken_OHLCVT_Full.zip.part0{i}", s) for i, s in enumerate([2097152000] * 4 + [1157945626])]

class Ranged(io.RawIOBase):
    def __init__(self, parts):
        self.parts, self.pos = parts, 0
        self.size = sum(s for _, s in parts)
        self.s = requests.Session()
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos
    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0: return 0
        got, off = 0, 0
        for url, sz in self.parts:
            if self.pos < off + sz and got < n:
                a = self.pos + got - off
                e = min(sz, a + (n - got)) - 1
                for _ in range(5):
                    try:
                        r = self.s.get(url, headers={"Range": f"bytes={a}-{e}"}, timeout=120); r.raise_for_status(); break
                    except Exception: pass
                b[got:got + len(r.content)] = r.content; got += len(r.content)
            off += sz
        self.pos += got
        return got

def single(url):
    size = int(requests.head(url, allow_redirects=True).headers["content-length"])
    return Ranged([(url, size)])

def daily(raw, quote="USD"):
    z = zipfile.ZipFile(io.BufferedReader(raw, 8 << 20))
    names = [n for n in z.namelist() if n.endswith("_1440.csv") and n.rsplit("/", 1)[-1][:-9].endswith(quote)]
    out = []
    for n in names:
        pair = n.rsplit("/", 1)[-1][:-9]
        df = pl.read_csv(z.read(n), has_header=False, new_columns=["ts", "open", "high", "low", "close", "volume", "trades"])
        out.append(df.with_columns(pl.lit(pair).alias("pair")))
    return pl.concat(out)

if __name__ == "__main__":
    full = daily(Ranged(PARTS))
    print("full", full.shape, file=sys.stderr)
    try:
        q3 = daily(single(B + "Kraken_OHLCVT_2026Q3.zip"))
        full = pl.concat([full, q3]).unique(["pair", "ts"]).sort(["pair", "ts"])
    except Exception as e:
        print("Q3 failed", repr(e)[:100], file=sys.stderr)
    full = full.with_columns(pl.from_epoch("ts").alias("date")).drop("ts")
    full.write_parquet("data/kraken_daily_usd.parquet")
    print("saved", full.shape, full["pair"].n_unique(), full["date"].min(), full["date"].max())
