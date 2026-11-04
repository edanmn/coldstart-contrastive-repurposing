# Extended membership control: plan written before running (2026-11-03)

Script: code/membership_control_v2.py. Inputs ecfp4, random, chemberta, gcn on Phase 2;
ecfp4 and random on the Phase 1 slice (1,001 drugs shared with Phase 2, M = 250).
BLAS threads pinned: OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=8.

H1 (data size). sizehold (class + 400 other drugs removed, negatives seen) vs matched
(class + the 400 negatives removed). Equal data; only negative membership differs.
Positive: sizehold > matched in classes where membership mattered (random vectors in
all gaining classes). Negative: sizehold ~ matched, i.e. the hold-matched gap was data size.

H2 (other encoders). ChemBERTa and GCN: does a membership signal exist (mem_hold > 0.5),
and does matched - full stay positive? A GCN with mem ~0.5 would mean its flat deltas
are not hiding a membership artifact.

H3 (replication). Phase 1 ecfp4: matched - full > 0 in the same four classes
(AED, SSRI, beta-blocker, antipsychotic); random vectors: matched gain near 0.
Caveat: same compounds, different signatures (different cells and plates).

H4 (drug level, membership-free). LOPO on matched vs full encoders, background drawn
from M (unseen in matched, seen in full). Pooled Wilcoxon over drugs.

H5 (AED). Weak transcriptional response distorts the full-data structure embedding.
Predictions: per-drug response strength (signature norm, replicate correlation)
correlates negatively with the per-drug matched gain; AEDs have weak responses.
A null correlation rejects H5.

H6 (published model). finlayson_model.py ports Finlayson et al.'s published encoders and
quadruplet margin loss (see its docstring for the three changes). Same arms as above.
Positive: a membership signal (mem_hold > 0.5) and hold - full gains that shrink under
matching would show the confound is not specific to our InfoNCE aligner.
