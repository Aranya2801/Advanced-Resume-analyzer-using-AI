# Results across generator seeds [13, 7, 42, 101, 2024]

Mean ± std over seeds, test split. Hybrid weights re-tuned on each seed's dev JDs.

## Setting A (full pool)

| Method | nDCG@10 | MAP | Stuffed@10 |
|---|---:|---:|---:|
| v1 keyword overlap (original repo) | 0.991 ± 0.004 | 0.944 ± 0.007 | 0.044 ± 0.009 |
| TF-IDF cosine | 0.997 ± 0.002 | 0.988 ± 0.005 | 0.025 ± 0.007 |
| BM25 | 0.998 ± 0.002 | 0.986 ± 0.006 | 0.031 ± 0.010 |
| Skills: exact only | 0.778 ± 0.032 | 0.766 ± 0.013 | 0.257 ± 0.032 |
| Skills: + families + evidence | 0.967 ± 0.008 | 0.879 ± 0.008 | 0.056 ± 0.013 |
| Dense: whole doc, 512 chars (v1) | 0.542 ± 0.038 | 0.411 ± 0.027 | 0.020 ± 0.023 |
| Dense: requirement coverage | 0.625 ± 0.034 | 0.488 ± 0.037 | 0.106 ± 0.046 |
| Hybrid, pool mode (+ TF-IDF, tuned on dev) | 0.992 ± 0.005 | 0.974 ± 0.008 | 0.042 ± 0.015 |
| Hybrid, pool mode (library default weights) | 0.994 ± 0.005 | 0.977 ± 0.004 | 0.042 ± 0.015 |

## Setting B (hard pool)

| Method | nDCG@10 | MAP | Stuffed@10 |
|---|---:|---:|---:|
| v1 keyword overlap (original repo) | 0.854 ± 0.029 | 0.764 ± 0.034 | 0.099 ± 0.023 |
| TF-IDF cosine | 0.940 ± 0.013 | 0.934 ± 0.028 | 0.098 ± 0.008 |
| BM25 | 0.933 ± 0.018 | 0.928 ± 0.031 | 0.117 ± 0.013 |
| Skills: exact only | 0.377 ± 0.017 | 0.288 ± 0.009 | 0.298 ± 0.002 |
| Skills: + families + evidence | 0.561 ± 0.025 | 0.444 ± 0.023 | 0.268 ± 0.009 |
| Dense: whole doc, 512 chars (v1) | 0.744 ± 0.037 | 0.535 ± 0.059 | 0.012 ± 0.007 |
| Dense: requirement coverage | 0.714 ± 0.017 | 0.541 ± 0.029 | 0.081 ± 0.012 |
| Hybrid, pool mode (+ TF-IDF, tuned on dev) | 0.946 ± 0.015 | 0.937 ± 0.029 | 0.090 ± 0.012 |
| Hybrid, pool mode (library default weights) | 0.879 ± 0.013 | 0.857 ± 0.022 | 0.163 ± 0.018 |
