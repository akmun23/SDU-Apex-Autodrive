# Phase 1 — replay minimum/maximum correction

The old percentile helper computed `ceil(p*n)-1`; at `p=0` it produced a
negative value that converted to a large unsigned index, then clamped to the
last (maximum) sample. Added explicit nearest-rank endpoint handling and
direct min/max helpers with regression tests. Replayed both traces after the
fix.
