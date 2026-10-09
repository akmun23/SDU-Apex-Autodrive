# Exact-two-packet yaw atlas: held-out cell inventory — 2026-10-08

This report is generated from the existing exact-two-packet audit. It
uses held-out validation captures only for the table below; test and
final-test partitions were not read. Speeds are floored to 1 m/s bins
and signed physical steering is rounded to 0.1 rad. Each row compares
the same validation cell for the global/event candidate and the
directly supported local speed-steering-event expert bank.

The local bank has no fallback: `local n` is the number of rows with a
directly supported local expert; `all n` is the total exact-two rows
in that cell. A dash means the cell has validation data but no local
expert prediction. Zero rows means no exact-two validation data in the
cell. Error columns are yaw-rate residuals in rad/s; they are not
yaw-angle errors. The separate response-phase angle results are in
`validation_atlas.json` and are teacher-forced one-step windows, not
recursive free-running yaw trajectories.

## Per-cell results

| GT speed (m/s) | Signed steering (rad) | All n | Global RMSE | Global p95 | Global max | Global >0.1 | Local n / all n | Val runs | Local RMSE | Local p95 | Local max | Local >0.1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–1 | -0.5 | 31 | 0.084 | 0.191 | 0.256 | 5 | 17 / 31 | 1 | 0.024 | 0.051 | 0.052 | 0 |
| 0–1 | -0.4 | 4 | 0.070 | 0.114 | 0.128 | 1 | 0 / 4 | 0 | — | — | — | — |
| 0–1 | -0.3 | 47 | 0.096 | 0.221 | 0.322 | 9 | 36 / 47 | 1 | 0.089 | 0.215 | 0.290 | 6 |
| 0–1 | -0.2 | 53 | 0.045 | 0.087 | 0.130 | 2 | 40 / 53 | 1 | 0.024 | 0.032 | 0.129 | 1 |
| 0–1 | -0.1 | 37 | 0.026 | 0.053 | 0.097 | 0 | 35 / 37 | 2 | 0.014 | 0.033 | 0.049 | 0 |
| 0–1 | +0.0 | 47 | 0.029 | 0.058 | 0.094 | 0 | 46 / 47 | 1 | 0.031 | 0.077 | 0.154 | 1 |
| 0–1 | +0.1 | 40 | 0.021 | 0.043 | 0.058 | 0 | 36 / 40 | 2 | 0.020 | 0.042 | 0.059 | 0 |
| 0–1 | +0.2 | 49 | 0.050 | 0.111 | 0.123 | 4 | 36 / 49 | 1 | 0.018 | 0.041 | 0.075 | 0 |
| 0–1 | +0.3 | 45 | 0.064 | 0.119 | 0.147 | 8 | 27 / 45 | 1 | 0.049 | 0.126 | 0.154 | 3 |
| 0–1 | +0.4 | 4 | 0.083 | 0.138 | 0.153 | 1 | 0 / 4 | 0 | — | — | — | — |
| 0–1 | +0.5 | 32 | 0.087 | 0.211 | 0.247 | 6 | 24 / 32 | 1 | 0.064 | 0.179 | 0.204 | 3 |
| 1–2 | -0.5 | 341 | 0.070 | 0.146 | 0.426 | 33 | 276 / 341 | 3 | 0.051 | 0.112 | 0.324 | 16 |
| 1–2 | -0.4 | 117 | 0.039 | 0.087 | 0.181 | 4 | 62 / 117 | 3 | 0.037 | 0.070 | 0.184 | 3 |
| 1–2 | -0.3 | 398 | 0.052 | 0.108 | 0.303 | 24 | 279 / 398 | 3 | 0.040 | 0.071 | 0.299 | 11 |
| 1–2 | -0.2 | 346 | 0.041 | 0.082 | 0.143 | 11 | 236 / 346 | 4 | 0.039 | 0.086 | 0.201 | 8 |
| 1–2 | -0.1 | 264 | 0.034 | 0.062 | 0.211 | 5 | 209 / 264 | 5 | 0.040 | 0.083 | 0.197 | 8 |
| 1–2 | +0.0 | 494 | 0.022 | 0.044 | 0.182 | 2 | 460 / 494 | 4 | 0.021 | 0.042 | 0.167 | 4 |
| 1–2 | +0.1 | 259 | 0.031 | 0.056 | 0.188 | 3 | 215 / 259 | 5 | 0.053 | 0.050 | 0.620 | 4 |
| 1–2 | +0.2 | 396 | 0.048 | 0.105 | 0.223 | 23 | 300 / 396 | 4 | 0.043 | 0.099 | 0.368 | 15 |
| 1–2 | +0.3 | 419 | 0.052 | 0.107 | 0.292 | 23 | 279 / 419 | 3 | 0.041 | 0.067 | 0.291 | 10 |
| 1–2 | +0.4 | 121 | 0.077 | 0.143 | 0.379 | 13 | 44 / 121 | 1 | 0.030 | 0.035 | 0.189 | 1 |
| 1–2 | +0.5 | 346 | 0.078 | 0.169 | 0.397 | 41 | 245 / 346 | 3 | 0.044 | 0.087 | 0.312 | 11 |
| 2–3 | -0.5 | 444 | 0.059 | 0.016 | 0.556 | 10 | 376 / 444 | 5 | 0.010 | 0.009 | 0.159 | 1 |
| 2–3 | -0.4 | 525 | 0.052 | 0.034 | 0.534 | 11 | 488 / 525 | 6 | 0.017 | 0.020 | 0.208 | 5 |
| 2–3 | -0.3 | 369 | 0.044 | 0.092 | 0.322 | 18 | 313 / 369 | 6 | 0.031 | 0.035 | 0.234 | 5 |
| 2–3 | -0.2 | 251 | 0.077 | 0.169 | 0.462 | 20 | 178 / 251 | 6 | 0.069 | 0.142 | 0.452 | 14 |
| 2–3 | -0.1 | 269 | 0.032 | 0.060 | 0.226 | 4 | 208 / 269 | 7 | 0.043 | 0.068 | 0.365 | 8 |
| 2–3 | +0.0 | 1031 | 0.032 | 0.076 | 0.380 | 17 | 995 / 1031 | 6 | 0.025 | 0.034 | 0.317 | 11 |
| 2–3 | +0.1 | 271 | 0.040 | 0.083 | 0.259 | 13 | 211 / 271 | 7 | 0.057 | 0.068 | 0.385 | 8 |
| 2–3 | +0.2 | 254 | 0.082 | 0.157 | 0.409 | 28 | 122 / 254 | 6 | 0.044 | 0.111 | 0.186 | 8 |
| 2–3 | +0.3 | 398 | 0.060 | 0.100 | 0.686 | 21 | 328 / 398 | 6 | 0.024 | 0.029 | 0.226 | 4 |
| 2–3 | +0.4 | 521 | 0.048 | 0.042 | 0.446 | 14 | 452 / 521 | 6 | 0.015 | 0.027 | 0.120 | 2 |
| 2–3 | +0.5 | 548 | 0.049 | 0.017 | 0.503 | 11 | 443 / 548 | 5 | 0.006 | 0.011 | 0.052 | 0 |
| 3–4 | -0.5 | 390 | 0.061 | 0.019 | 0.697 | 8 | 325 / 390 | 4 | 0.011 | 0.004 | 0.190 | 1 |
| 3–4 | -0.4 | 809 | 0.057 | 0.028 | 0.680 | 11 | 705 / 809 | 7 | 0.045 | 0.011 | 0.856 | 4 |
| 3–4 | -0.3 | 605 | 0.066 | 0.057 | 0.684 | 20 | 471 / 605 | 7 | 0.029 | 0.013 | 0.286 | 7 |
| 3–4 | -0.2 | 344 | 0.096 | 0.199 | 0.892 | 28 | 250 / 344 | 7 | 0.121 | 0.179 | 1.003 | 31 |
| 3–4 | -0.1 | 397 | 0.056 | 0.114 | 0.384 | 27 | 295 / 397 | 8 | 0.082 | 0.223 | 0.476 | 26 |
| 3–4 | +0.0 | 1665 | 0.035 | 0.079 | 0.394 | 61 | 1613 / 1665 | 7 | 0.026 | 0.032 | 0.297 | 24 |
| 3–4 | +0.1 | 391 | 0.056 | 0.112 | 0.400 | 23 | 319 / 391 | 8 | 0.030 | 0.058 | 0.204 | 10 |
| 3–4 | +0.2 | 374 | 0.069 | 0.132 | 0.481 | 29 | 254 / 374 | 7 | 0.026 | 0.045 | 0.240 | 3 |
| 3–4 | +0.3 | 676 | 0.094 | 0.051 | 1.634 | 23 | 514 / 676 | 8 | 0.032 | 0.023 | 0.323 | 9 |
| 3–4 | +0.4 | 824 | 0.067 | 0.026 | 0.713 | 13 | 720 / 824 | 7 | 0.040 | 0.010 | 0.730 | 8 |
| 3–4 | +0.5 | 443 | 0.062 | 0.019 | 0.659 | 9 | 387 / 443 | 4 | 0.008 | 0.008 | 0.095 | 0 |
| 4–5 | -0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 4–5 | -0.4 | 209 | 0.027 | 0.031 | 0.249 | 4 | 182 / 209 | 5 | 0.018 | 0.026 | 0.158 | 2 |
| 4–5 | -0.3 | 220 | 0.019 | 0.028 | 0.162 | 3 | 115 / 220 | 6 | 0.010 | 0.012 | 0.062 | 0 |
| 4–5 | -0.2 | 181 | 0.030 | 0.063 | 0.188 | 2 | 88 / 181 | 1 | 0.004 | 0.005 | 0.019 | 0 |
| 4–5 | -0.1 | 255 | 0.028 | 0.061 | 0.168 | 2 | 171 / 255 | 6 | 0.048 | 0.089 | 0.263 | 8 |
| 4–5 | +0.0 | 684 | 0.035 | 0.053 | 0.394 | 17 | 591 / 684 | 8 | 0.031 | 0.033 | 0.263 | 21 |
| 4–5 | +0.1 | 280 | 0.035 | 0.067 | 0.175 | 9 | 218 / 280 | 6 | 0.026 | 0.061 | 0.153 | 3 |
| 4–5 | +0.2 | 203 | 0.047 | 0.054 | 0.425 | 5 | 136 / 203 | 3 | 0.015 | 0.030 | 0.092 | 0 |
| 4–5 | +0.3 | 201 | 0.018 | 0.029 | 0.158 | 2 | 86 / 201 | 5 | 0.018 | 0.028 | 0.101 | 1 |
| 4–5 | +0.4 | 184 | 0.035 | 0.030 | 0.242 | 5 | 150 / 184 | 5 | 0.025 | 0.047 | 0.234 | 2 |
| 4–5 | +0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 5–6 | -0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 5–6 | -0.4 | 77 | 0.015 | 0.037 | 0.054 | 0 | 77 / 77 | 3 | 0.019 | 0.056 | 0.065 | 0 |
| 5–6 | -0.3 | 75 | 0.014 | 0.031 | 0.041 | 0 | 61 / 75 | 4 | 0.006 | 0.018 | 0.020 | 0 |
| 5–6 | -0.2 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 5–6 | -0.1 | 100 | 0.023 | 0.059 | 0.091 | 0 | 4 / 100 | 1 | 0.004 | 0.006 | 0.006 | 0 |
| 5–6 | +0.0 | 291 | 0.011 | 0.032 | 0.053 | 0 | 213 / 291 | 4 | 0.005 | 0.015 | 0.015 | 0 |
| 5–6 | +0.1 | 67 | 0.024 | 0.063 | 0.086 | 0 | 28 / 67 | 3 | 0.023 | 0.059 | 0.064 | 0 |
| 5–6 | +0.2 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 5–6 | +0.3 | 69 | 0.023 | 0.036 | 0.149 | 1 | 63 / 69 | 3 | 0.022 | 0.030 | 0.154 | 1 |
| 5–6 | +0.4 | 66 | 0.016 | 0.032 | 0.058 | 0 | 27 / 66 | 3 | 0.014 | 0.029 | 0.031 | 0 |
| 5–6 | +0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 6–7 | -0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 6–7 | -0.4 | 227 | 0.032 | 0.023 | 0.265 | 5 | 181 / 227 | 5 | 0.022 | 0.053 | 0.165 | 2 |
| 6–7 | -0.3 | 110 | 0.023 | 0.035 | 0.131 | 1 | 71 / 110 | 4 | 0.017 | 0.042 | 0.063 | 0 |
| 6–7 | -0.2 | 227 | 0.050 | 0.018 | 0.444 | 3 | 149 / 227 | 2 | 0.010 | 0.003 | 0.076 | 0 |
| 6–7 | -0.1 | 330 | 0.042 | 0.093 | 0.291 | 13 | 199 / 330 | 5 | 0.009 | 0.015 | 0.053 | 0 |
| 6–7 | +0.0 | 419 | 0.048 | 0.080 | 0.490 | 19 | 335 / 419 | 6 | 0.034 | 0.035 | 0.264 | 11 |
| 6–7 | +0.1 | 345 | 0.048 | 0.089 | 0.408 | 14 | 254 / 345 | 5 | 0.021 | 0.017 | 0.231 | 2 |
| 6–7 | +0.2 | 232 | 0.030 | 0.019 | 0.333 | 2 | 168 / 232 | 2 | 0.008 | 0.004 | 0.066 | 0 |
| 6–7 | +0.3 | 103 | 0.022 | 0.035 | 0.131 | 2 | 43 / 103 | 1 | 0.024 | 0.054 | 0.074 | 0 |
| 6–7 | +0.4 | 231 | 0.027 | 0.019 | 0.288 | 3 | 195 / 231 | 5 | 0.021 | 0.032 | 0.152 | 3 |
| 6–7 | +0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 7–8 | -0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 7–8 | -0.4 | 169 | 0.015 | 0.016 | 0.157 | 1 | 51 / 169 | 4 | 0.018 | 0.031 | 0.113 | 1 |
| 7–8 | -0.3 | 132 | 0.007 | 0.012 | 0.018 | 0 | 0 / 132 | 0 | — | — | — | — |
| 7–8 | -0.2 | 99 | 0.011 | 0.020 | 0.037 | 0 | 0 / 99 | 0 | — | — | — | — |
| 7–8 | -0.1 | 178 | 0.049 | 0.082 | 0.317 | 7 | 32 / 178 | 2 | 0.015 | 0.025 | 0.031 | 0 |
| 7–8 | +0.0 | 413 | 0.055 | 0.175 | 0.225 | 35 | 237 / 413 | 4 | 0.000 | 0.000 | 0.002 | 0 |
| 7–8 | +0.1 | 166 | 0.047 | 0.080 | 0.331 | 4 | 32 / 166 | 4 | 0.054 | 0.065 | 0.281 | 1 |
| 7–8 | +0.2 | 92 | 0.012 | 0.019 | 0.034 | 0 | 1 / 92 | 1 | 0.000 | 0.000 | 0.000 | 0 |
| 7–8 | +0.3 | 109 | 0.007 | 0.013 | 0.020 | 0 | 9 / 109 | 1 | 0.004 | 0.006 | 0.007 | 0 |
| 7–8 | +0.4 | 164 | 0.029 | 0.026 | 0.302 | 2 | 124 / 164 | 4 | 0.021 | 0.024 | 0.182 | 1 |
| 7–8 | +0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 8–9 | -0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 8–9 | -0.4 | 134 | 0.029 | 0.022 | 0.251 | 2 | 87 / 134 | 2 | 0.012 | 0.014 | 0.057 | 0 |
| 8–9 | -0.3 | 17 | 0.012 | 0.022 | 0.023 | 0 | 0 / 17 | 0 | — | — | — | — |
| 8–9 | -0.2 | 129 | 0.047 | 0.022 | 0.507 | 2 | 0 / 129 | 0 | — | — | — | — |
| 8–9 | -0.1 | 346 | 0.045 | 0.058 | 0.696 | 5 | 281 / 346 | 4 | 0.017 | 0.029 | 0.144 | 2 |
| 8–9 | +0.0 | 415 | 0.045 | 0.063 | 0.319 | 16 | 391 / 415 | 2 | 0.080 | 0.180 | 0.886 | 23 |
| 8–9 | +0.1 | 345 | 0.042 | 0.071 | 0.410 | 11 | 306 / 345 | 4 | 0.024 | 0.034 | 0.230 | 4 |
| 8–9 | +0.2 | 134 | 0.033 | 0.023 | 0.266 | 2 | 90 / 134 | 1 | 0.016 | 0.003 | 0.141 | 1 |
| 8–9 | +0.3 | 12 | 0.012 | 0.018 | 0.018 | 0 | 3 / 12 | 1 | 0.001 | 0.001 | 0.001 | 0 |
| 8–9 | +0.4 | 143 | 0.022 | 0.027 | 0.173 | 2 | 101 / 143 | 2 | 0.008 | 0.012 | 0.058 | 0 |
| 8–9 | +0.5 | 0 | — | — | — | — | 0 / 0 | 0 | — | — | — | — |
| 9–10 | -0.5 | 142 | 0.008 | 0.019 | 0.034 | 0 | 0 / 142 | 0 | — | — | — | — |
| 9–10 | -0.4 | 115 | 0.011 | 0.026 | 0.034 | 0 | 0 / 115 | 0 | — | — | — | — |
| 9–10 | -0.3 | 125 | 0.008 | 0.016 | 0.021 | 0 | 0 / 125 | 0 | — | — | — | — |
| 9–10 | -0.2 | 65 | 0.013 | 0.021 | 0.024 | 0 | 0 / 65 | 0 | — | — | — | — |
| 9–10 | -0.1 | 252 | 0.038 | 0.093 | 0.145 | 9 | 199 / 252 | 3 | 0.028 | 0.069 | 0.137 | 3 |
| 9–10 | +0.0 | 694 | 0.047 | 0.079 | 0.343 | 27 | 667 / 694 | 2 | 0.115 | 0.198 | 0.788 | 63 |
| 9–10 | +0.1 | 252 | 0.049 | 0.085 | 0.403 | 9 | 196 / 252 | 3 | 0.093 | 0.121 | 0.709 | 13 |
| 9–10 | +0.2 | 68 | 0.014 | 0.020 | 0.026 | 0 | 0 / 68 | 0 | — | — | — | — |
| 9–10 | +0.3 | 126 | 0.008 | 0.016 | 0.019 | 0 | 0 / 126 | 0 | — | — | — | — |
| 9–10 | +0.4 | 114 | 0.010 | 0.025 | 0.035 | 0 | 7 / 114 | 2 | 0.007 | 0.012 | 0.012 | 0 |
| 9–10 | +0.5 | 143 | 0.006 | 0.016 | 0.027 | 0 | 2 / 143 | 1 | 0.003 | 0.003 | 0.003 | 0 |
| 10–11 | -0.5 | 314 | 0.052 | 0.055 | 0.371 | 8 | 12 / 314 | 1 | 0.220 | 0.366 | 0.370 | 6 |
| 10–11 | -0.4 | 220 | 0.082 | 0.228 | 0.366 | 19 | 14 / 220 | 1 | 0.229 | 0.327 | 0.333 | 7 |
| 10–11 | -0.3 | 271 | 0.050 | 0.054 | 0.299 | 12 | 31 / 271 | 1 | 0.123 | 0.221 | 0.280 | 10 |
| 10–11 | -0.2 | 202 | 0.017 | 0.033 | 0.104 | 1 | 52 / 202 | 2 | 0.012 | 0.001 | 0.088 | 0 |
| 10–11 | -0.1 | 315 | 0.062 | 0.135 | 0.405 | 42 | 217 / 315 | 3 | 0.119 | 0.328 | 0.543 | 31 |
| 10–11 | +0.0 | 1240 | 0.051 | 0.107 | 0.330 | 66 | 1190 / 1240 | 2 | 0.127 | 0.239 | 0.752 | 135 |
| 10–11 | +0.1 | 320 | 0.056 | 0.131 | 0.234 | 35 | 177 / 320 | 3 | 0.130 | 0.199 | 0.720 | 33 |
| 10–11 | +0.2 | 189 | 0.021 | 0.041 | 0.118 | 2 | 3 / 189 | 1 | 0.087 | 0.133 | 0.143 | 1 |
| 10–11 | +0.3 | 267 | 0.047 | 0.106 | 0.251 | 14 | 15 / 267 | 1 | 0.165 | 0.253 | 0.254 | 8 |
| 10–11 | +0.4 | 222 | 0.060 | 0.128 | 0.346 | 12 | 16 / 222 | 1 | 0.223 | 0.414 | 0.423 | 9 |
| 10–11 | +0.5 | 284 | 0.045 | 0.052 | 0.401 | 6 | 11 / 284 | 1 | 0.148 | 0.283 | 0.287 | 3 |
| 11–12 | -0.5 | 26 | 0.187 | 0.333 | 0.343 | 11 | 0 / 26 | 0 | — | — | — | — |
| 11–12 | -0.4 | 25 | 0.181 | 0.311 | 0.317 | 15 | 0 / 25 | 0 | — | — | — | — |
| 11–12 | -0.3 | 25 | 0.116 | 0.210 | 0.278 | 7 | 1 / 25 | 1 | 0.023 | 0.023 | 0.023 | 0 |
| 11–12 | -0.2 | 79 | 0.044 | 0.096 | 0.232 | 4 | 51 / 79 | 1 | 0.000 | 0.000 | 0.001 | 0 |
| 11–12 | -0.1 | 111 | 0.039 | 0.074 | 0.210 | 4 | 104 / 111 | 3 | 0.056 | 0.117 | 0.306 | 8 |
| 11–12 | +0.0 | 359 | 0.046 | 0.081 | 0.333 | 15 | 359 / 359 | 2 | 0.079 | 0.085 | 0.711 | 12 |
| 11–12 | +0.1 | 116 | 0.056 | 0.117 | 0.354 | 9 | 70 / 116 | 3 | 0.091 | 0.091 | 0.734 | 2 |
| 11–12 | +0.2 | 66 | 0.055 | 0.164 | 0.239 | 5 | 0 / 66 | 0 | — | — | — | — |
| 11–12 | +0.3 | 16 | 0.149 | 0.280 | 0.280 | 6 | 0 / 16 | 0 | — | — | — | — |
| 11–12 | +0.4 | 27 | 0.153 | 0.294 | 0.332 | 10 | 3 / 27 | 1 | 0.055 | 0.070 | 0.072 | 0 |
| 11–12 | +0.5 | 24 | 0.172 | 0.342 | 0.344 | 9 | 0 / 24 | 0 | — | — | — | — |

## Training expert coverage (causal selector grid)

This grid counts fitted train-only experts selected by rear-wheel speed
and signed steering-feedback bins. A value is the number of distinct
causal event experts available in that cell; it is not a validation
accuracy claim. Unlike the validation table above, this makes missing
model cells explicit even when no validation sample happened to visit
them. The 1 m/s × 0.1 rad grid is a finite resolution, not every real
valued point in the continuous domain.

The fitted bank has 136 event experts across 82/132 selector cells; 50 cells have no local expert.

| Rear-wheel speed cell (m/s) | -0.5 | -0.4 | -0.3 | -0.2 | -0.1 | -0.0 | +0.1 | +0.2 | +0.3 | +0.4 | +0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0–1 | 1 | 1 | 1 | 2 | 1 | 2 | 2 | 2 | 1 | 0 | 1 |
| 1–2 | 1 | 1 | 0 | 2 | 1 | 2 | 2 | 1 | 1 | 0 | 1 |
| 2–3 | 2 | 2 | 1 | 2 | 1 | 2 | 2 | 1 | 2 | 1 | 1 |
| 3–4 | 2 | 3 | 3 | 3 | 1 | 3 | 3 | 2 | 3 | 3 | 2 |
| 4–5 | 0 | 1 | 0 | 1 | 3 | 3 | 2 | 1 | 1 | 1 | 0 |
| 5–6 | 0 | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 0 | 0 | 0 |
| 6–7 | 0 | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 0 | 1 | 0 |
| 7–8 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 1 | 0 |
| 8–9 | 0 | 1 | 0 | 1 | 3 | 3 | 2 | 1 | 0 | 1 | 0 |
| 9–10 | 0 | 0 | 0 | 0 | 3 | 3 | 1 | 0 | 0 | 1 | 0 |
| 10–11 | 0 | 0 | 0 | 1 | 2 | 3 | 1 | 0 | 0 | 0 | 0 |
| 11–12 | 0 | 0 | 0 | 1 | 3 | 3 | 1 | 0 | 0 | 0 | 0 |

## Measured response phase versus causal expert route

`Measured phase` is an offline label derived from the recorded response;
it is for diagnosis only and cannot route a live observer. `Causal
route` is the route actually used by the local predictor. Errors below
are for the same exact-two validation rows as the main table.

| Measured response phase | Causal route | n | RMSE | p95 | Max | >0.1 |
|---|---|---:|---:|---:|---:|---:|
| command_gap | hold | 1021 | 0.007 | 0.004 | 0.097 | 0 |
| command_gap | turn_in | 50 | 0.185 | 0.273 | 0.296 | 48 |
| command_gap | unwind | 223 | 0.025 | 0.053 | 0.174 | 5 |
| onset | hold | 4196 | 0.019 | 0.009 | 0.370 | 20 |
| onset | turn_in | 2830 | 0.051 | 0.110 | 0.375 | 163 |
| reversal | hold | 4914 | 0.028 | 0.030 | 0.620 | 70 |
| reversal | turn_in | 1643 | 0.064 | 0.171 | 0.425 | 146 |
| reversal | unwind | 278 | 0.140 | 0.247 | 0.886 | 65 |
| throttle_acceleration | hold | 84 | 0.000 | 0.000 | 0.000 | 0 |
| throttle_active_braking | hold | 101 | 0.014 | 0.030 | 0.052 | 0 |
| throttle_cut | hold | 369 | 0.016 | 0.037 | 0.087 | 0 |
| throttle_down | hold | 796 | 0.022 | 0.046 | 0.234 | 6 |
| throttle_throttle_reduction | hold | 54 | 0.004 | 0.008 | 0.010 | 0 |
| throttle_up | hold | 1016 | 0.016 | 0.029 | 0.168 | 6 |
| unwind | hold | 3112 | 0.014 | 0.007 | 0.333 | 7 |
| unwind | unwind | 1129 | 0.184 | 0.553 | 1.003 | 191 |

## Largest local-bank error groups

These are held-out cell aggregates, not independent physical laws;
the count above 0.1 rad/s highlights error mass while max exposes
isolated tails.

| GT speed cell | Steering cell | Val rows | Val runs | RMSE | p95 | Max | >0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10–11 | +0.0 | 1190 | 2 | 0.127 | 0.239 | 0.752 | 135 |
| 9–10 | +0.0 | 667 | 2 | 0.115 | 0.198 | 0.788 | 63 |
| 10–11 | +0.1 | 177 | 3 | 0.130 | 0.199 | 0.720 | 33 |
| 3–4 | -0.2 | 250 | 7 | 0.121 | 0.179 | 1.003 | 31 |
| 10–11 | -0.1 | 217 | 3 | 0.119 | 0.328 | 0.543 | 31 |
| 3–4 | -0.1 | 295 | 8 | 0.082 | 0.223 | 0.476 | 26 |
| 3–4 | +0.0 | 1613 | 7 | 0.026 | 0.032 | 0.297 | 24 |
| 8–9 | +0.0 | 391 | 2 | 0.080 | 0.180 | 0.886 | 23 |
| 4–5 | +0.0 | 591 | 8 | 0.031 | 0.033 | 0.263 | 21 |
| 1–2 | -0.5 | 276 | 3 | 0.051 | 0.112 | 0.324 | 16 |
| 1–2 | +0.2 | 300 | 4 | 0.043 | 0.099 | 0.368 | 15 |
| 2–3 | -0.2 | 178 | 6 | 0.069 | 0.142 | 0.452 | 14 |

## What explains the largest tails so far

The exact-two validation cross-tab above shows the local route does not
have a distinct causal `reversal` state. The offline measured response
label is not allowed to choose a live model, but it exposes where the
causal router is mixing response phases:

- Measured reversal routed as hold: 4,722 rows; 63 exceed 0.1 rad/s (max 0.620).
- Measured reversal routed as turn-in: 1,474 rows; 53 exceed 0.1 rad/s (max 0.323).
- Measured reversal routed as unwind: 278 rows; 65 exceed 0.1 rad/s (RMSE 0.140, max 0.886).
- Measured unwind routed as hold: 2,298 rows; none exceed 0.1 rad/s;
  measured unwind routed as unwind: 815 rows; 90 exceed 0.1 rad/s
  (RMSE 0.099, max 1.007).

Run-held-out exact-two feature ablation confirms steering actuator
feedback/command history is essential, but the tested extra channels do
not explain the residual tail. Removing steering-actuator features
changes reversal CV RMSE 0.168→0.289 rad/s and >0.1 misses 105→312; for unwind it changes 0.053→0.208 rad/s and >0.1 misses 86→595.

Adding IMU roll and roll rate instead gives reversal CV RMSE 0.172 versus 0.168 rad/s, and unwind 0.053 versus 0.053. Rear-wheel, throttle, IMU-yaw-history, and receipt-timing ablations likewise do not remove the held-out failure tail.

Simulator-truth body-motion information was also tested only as an
offline diagnostic, never as a predictor input. On the independent
packet-phase validation capture, adding GT body u/v did not improve
reversal RMSE (0.142→0.150 rad/s) or unwind RMSE (0.111→0.107 rad/s). Thus wheel/body speed mismatch and roll are not established as the missing universal cause. The best-supported explanation so far is a transition-phase/steering-response mismatch, with an incomplete causal event router; it is not yet a fully solved physical law.

## Coverage and angle-error interpretation

- Exact-two validation: 31,086 rows, 120 occupied center-based 1 m/s × 0.1 rad signed cells (115 have at least 20 rows); speed 0.49–11.26 m/s, |steering| up to 0.50 rad.
- Local predictions appear in 103 / 120 occupied validation cells, but per-row local coverage is lower because not every maneuver event in those cells has a supported expert.
- The coarse local bank contains 136 experts and predicts 70.2% of exact-two validation rows without fallback.
- Local-bank phase integration: 227 / 1387 phases fully supported; maximum cumulative absolute yaw-angle error 1.046°; 0 supported phases above 5°.
- Causal-event model phase integration: 1387 / 1387 phases fully supported; maximum cumulative absolute yaw-angle error 1.622°; 0 phases above 5°.
- This does **not** certify the full requested domain: unsupported
  cells are not counted as passing, and phase integration uses
  one-step teacher-forced residuals rather than recursive rollout.
  The exact-two dataset has no validation near 12 m/s and lacks the
  high-speed/high-steering surface described in the parent audit.

Source: `live_runs/racing_model_diagnostics_20261008/yaw_full_domain_exact_two_v2/validation_atlas.json` and `report.json`.
