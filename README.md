# Crypto optimizer

Benchmark and optimizer for next-day crypto return forecasts, scored in bits on unseen data (Kraken daily USD, 30 coins per group).

- `bench.py` scores a `predict(history)` program, with a leak guard, search penalty and EV after Kraken fees.
- `engine/stats.py` has block bootstrap, TOST and a fixed-share mixture. `gates.py` runs kill gates 1 to 3.
- `optimize.py` (Optuna) and `evolve.py` (OpenEvolve with local Ollama) search programs. `certificate.py` decides done / data_limited.
- Result: no tradable edge, see `REPORT.md` and the live page at https://reedsmoneymaker23.com/optimizer/

Run: `uv sync`, `uv run python kraken_data.py` (range-reads Kraken zips), `make check`, `python3 build_site.py`.
