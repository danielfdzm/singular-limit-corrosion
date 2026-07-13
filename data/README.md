# Numerical Data

- `summaries/` contains the compact CSV tables and the sampled trace used by the notebook.
- `metadata/` records the geometry, mesh, and parameter choices for the included runs.

The included summaries and metadata are not overwritten when experiments are
recomputed. A fresh run creates `computed/` for regenerable tables and
finite-element solution arrays; that output folder is ignored by version control.