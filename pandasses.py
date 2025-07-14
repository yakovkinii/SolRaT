import time
import numpy as np
import pandas as pd
import polars as pl


class PolarsFrame:
    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __getattr__(self, name):
        if name in self._df.columns:
            return self._df[name]
        raise AttributeError(f"'PolarsFrame' object has no attribute '{name}'")

    def __getitem__(self, key):
        return self._df[key]

    def __setitem__(self, key, value):
        self._df = self._df.with_columns(pl.Series(name=key, values=value))

    def merge(self, other, on, how="left"):
        return PolarsFrame(self._df.join(other._df, on=on, how=how))

    def sum(self):
        return self._df.sum().to_series()[0]

    def select(self, expr):
        return self._df.select(expr)

    def to_polars(self):
        return self._df

    def __repr__(self):
        return repr(self._df)



# Create synthetic data
rows = 10_000_000
np.random.seed(42)
data1 = np.random.rand(rows, 5) + 1e-3
data2 = np.random.rand(rows, 2) + 1e-3

columns1 = ["key", "a", "b", "c", "d"]
columns2 = ["key", "mult"]

df1 = pd.DataFrame(data1, columns=columns1)
df2 = pd.DataFrame(data2, columns=columns2)

pdf1 = pl.DataFrame(df1)
pdf2 = pl.DataFrame(df2)
pdfl1 = pl.LazyFrame(df1)
pdfl2 = pl.LazyFrame(df2)

nwf1 = PolarsFrame(pdf1)
nwf2 = PolarsFrame(pdf2)

# --- Pandas ---
t0 = time.perf_counter()
merged_pd = df1.merge(df2, on="key")
merged_pd["result"] = merged_pd["a"] * merged_pd["b"] * merged_pd["c"] * merged_pd["d"] * merged_pd["mult"]
result = merged_pd["result"].sum()
t1 = time.perf_counter()
print(f"[Pandas] Time: {t1 - t0:.2f} sec, Result: {result:.2f}")

# --- Polars ---
t0 = time.perf_counter()
merged_pl = pdf1.join(pdf2, on="key")
result = (
    merged_pl.select(
        (pl.col("a") * pl.col("b") * pl.col("c") * pl.col("d") * pl.col("mult")).alias("result")
    ).sum()
)
t1 = time.perf_counter()
print(f"[Polars] Time: {t1 - t0:.2f} sec, Result: {result.item():.2f}")

# --- Polars Lazy ---
merged_lazy = pdfl1.join(pdfl2, on="key")
result = (
    merged_lazy.select(
        (pl.col("a") * pl.col("b") * pl.col("c") * pl.col("d") * pl.col("mult")).alias("result")
    ).sum()
)
t0 = time.perf_counter()
result = result.collect()
t1 = time.perf_counter()
print(f"[Polars Lazy] Time: {t1 - t0:.2f} sec, Result: {result.item():.2f}")

t0 = time.perf_counter()
merged_pd = nwf1.merge(nwf2, on="key")
merged_pd["result"] = merged_pd["a"] * merged_pd["b"] * merged_pd["c"] * merged_pd["d"] * merged_pd["mult"]
result = merged_pd["result"].sum()
t1 = time.perf_counter()
print(f"[Narwhals] Time: {t1 - t0:.2f} sec, Result: {result:.2f}")
