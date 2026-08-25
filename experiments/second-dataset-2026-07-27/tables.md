### Structure

| | German credit | Taiwan |
|:--|--:|--:|
| n_test | 300 | 9000 |
| d | 20 | 23 |
| curve length L | 21 | 24 |
| tensor cells | 9261 | 13824 |
| B | 2000 | 2000 |
| B_full logit / rf | 300 / 30 | 300 / 30 |

### P6, the estimator bridge on the RGE-RGR pair

| rung | construction | German logit | Taiwan logit | German rf | Taiwan rf |
|--:|:--|--:|--:|--:|--:|
| 1 | scalar RGE (bare) x scalar RGR (bare) [deposit pair] | +0.5967 | +0.5028 (mc se 0.0168) | +0.3415 | +0.4688 (mc se 0.0169) |
| 2 | scalar RGE (class_ord) x scalar RGR (class_ord) [convention only] | +0.6144 | +0.4952 (mc se 0.0168) | +0.3437 | +0.4417 (mc se 0.0175) |
| 3 | curve-mean RGE x scalar RGR (class_ord) [RGE estimator swapped] | +0.4249 | +0.4490 (mc se 0.0189) | +0.2879 | +0.4930 (mc se 0.0164) |
| 4 | scalar RGE (class_ord) x curve-mean RGR (swap) [RGR estimator swapped] | -0.2083 | -0.3789 (mc se 0.0186) | -0.1529 | -0.3174 (mc se 0.0200) |
| 5 | curve-mean RGE x noise-sweep RGR mean [sweep, same family] | +0.5354 | +0.6062 (mc se 0.0144) | +0.4711 | +0.7530 (mc se 0.0099) |
| 6 | curve-mean RGE x curve-mean RGR (swap) [study pair] | -0.1372 | -0.5423 (mc se 0.0161) | -0.3102 | -0.2266 (mc se 0.0212) |

### P1, P2, curve-summary correlations

| quantity | German | Taiwan | Taiwan mc se |
|:--|--:|--:|--:|
| corr(curve-mean RGA-RGE), logit | +0.1996 | +0.2805 | 0.0202 |
| corr(curve-mean RGA-RGR), logit | -0.0583 | -0.2540 | 0.0216 |
| corr(curve-mean RGE-RGR), logit | -0.1372 | -0.5423 | 0.0161 |
| corr(curve-mean RGA-RGE), rf | +0.1574 | +0.2477 | 0.0204 |
| corr(curve-mean RGA-RGR), rf | -0.0800 | -0.1078 | 0.0220 |
| corr(curve-mean RGE-RGR), rf | -0.3102 | -0.2266 | 0.0212 |

### P3 and P4, the volume composite and the cross-terms

| arm | model | German delta % | Taiwan delta % (mc se) | German empirical % | Taiwan empirical % (mc se) |
|:--|:--|--:|--:|--:|--:|
| arithmetic | logit | -0.91 | -9.83 (1.01) | +0.89 | -11.96 (4.27) |
| geometric | logit | -1.86 | -13.40 (1.60) | +1.45 | -11.05 (3.87) |
| rms | logit | +1.26 | -5.17 (0.66) | -1.41 | -8.47 (3.56) |
| topsis | logit | -1.50 | -8.00 (0.85) | -3.10 | -12.29 (3.83) |
| arithmetic | rf | -3.08 | -4.18 (1.16) | -6.49 | -4.60 (3.04) |
| geometric | rf | -3.09 | -4.12 (1.10) | -5.21 | -5.27 (3.45) |
| rms | rf | -1.50 | -1.52 (1.08) | -4.86 | -1.43 (3.38) |
| topsis | rf | -3.11 | -5.70 (1.15) | -7.56 | -5.53 (3.23) |

### Taiwan volume composite: point values and intervals

| arm | model | V | paired boot 95% | width | delta 95% measured | delta 95% cross-terms zero |
|:--|:--|--:|:--|--:|:--|:--|
| arithmetic | logit | 0.541986 | [0.537812, 0.544829] | 0.007017 | [0.538428, 0.545543] | [0.538078, 0.545893] |
| geometric | logit | 0.394666 | [0.389651, 0.398415] | 0.008764 | [0.390288, 0.399044] | [0.389702, 0.399631] |
| rms | logit | 0.650805 | [0.647634, 0.652922] | 0.005288 | [0.648144, 0.653465] | [0.648007, 0.653603] |
| topsis | logit | 0.304668 | [0.300413, 0.307439] | 0.007026 | [0.301098, 0.308237] | [0.300813, 0.308523] |
| arithmetic | rf | 0.614617 | [0.610384, 0.618848] | 0.008465 | [0.610425, 0.618808] | [0.610250, 0.618983] |
| geometric | rf | 0.530703 | [0.525756, 0.537056] | 0.011299 | [0.525062, 0.536344] | [0.524830, 0.536577] |
| rms | rf | 0.674230 | [0.670909, 0.677246] | 0.006337 | [0.671062, 0.677397] | [0.671014, 0.677445] |
| topsis | rf | 0.370052 | [0.366319, 0.373521] | 0.007202 | [0.366422, 0.373683] | [0.366215, 0.373890] |

### P5, the two-link chain

| quantity | scheme | German | Taiwan | Taiwan mc se |
|:--|:--|--:|--:|--:|
| cross-link correlation | fixed_draw | +0.7222 | +0.7836 | 0.0084 |
| width, measured covariance | fixed_draw | 0.079804 | 0.018824 | |
| width, cross-terms zero | fixed_draw | 0.060817 | 0.014103 | |
| understatement % | fixed_draw | 23.79 | 25.08 | 0.18 |
| cross-link correlation | redraw | +0.7120 | +0.7253 | 0.0102 |
| width, measured covariance | redraw | 0.079954 | 0.018968 | |
| width, cross-terms zero | redraw | 0.061108 | 0.014460 | |
| understatement % | redraw | 23.57 | 23.77 | 0.22 |

### Per-severity-index RGE-RGR correlation

| index | German logit | Taiwan logit | German rf | Taiwan rf |
|--:|--:|--:|--:|--:|
| 0 | n/a | n/a | n/a | n/a |
| 1 | +0.1418 | -0.1848 | +0.0280 | +0.1255 |
| 2 | +0.1122 | -0.0826 | +0.0434 | +0.1131 |
| 3 | +0.1004 | -0.1564 | -0.0074 | +0.0971 |
| 4 | -0.0097 | -0.0929 | +0.0044 | +0.0168 |
| 5 | -0.0272 | -0.0498 | -0.0337 | +0.0438 |
| 6 | -0.0769 | -0.0884 | -0.0241 | -0.0575 |
| 7 | -0.1039 | -0.0935 | -0.0948 | -0.0567 |
| 8 | -0.1259 | -0.1670 | -0.1432 | -0.1461 |
| 9 | -0.1711 | -0.2207 | -0.1926 | -0.1836 |
| 10 | -0.1436 | -0.2400 | -0.1592 | -0.1443 |
| 11 | -0.1338 | -0.3012 | -0.1147 | -0.1156 |
| 12 | -0.1616 | -0.2579 | -0.1733 | -0.1507 |
| 13 | -0.1408 | -0.2132 | -0.2028 | -0.2078 |
| 14 | -0.1835 | -0.2809 | -0.2127 | -0.1725 |
| 15 | -0.1863 | -0.2895 | -0.2971 | -0.1710 |
| 16 | -0.2221 | -0.2962 | -0.3041 | -0.2204 |
| 17 | -0.2506 | -0.2959 | -0.2536 | -0.3857 |
| 18 | -0.2248 | -0.3638 | -0.5088 | -0.2117 |
| 19 | -0.3132 | -0.3294 | -0.3667 | -0.2088 |
| 20 | -0.0432 | -0.4015 | -0.0347 | -0.1612 |
| 21 | n/a | -0.3434 | n/a | -0.4074 |
| 22 | n/a | -0.4748 | n/a | -0.2298 |
| 23 | n/a | +0.0111 | n/a | -0.0047 |

Taiwan logit half-grid means: RGE-RGR first -0.1525, second -0.2946; RGA-RGR first -0.2049, second -0.1082

German logit mean per-index: {'RGA_RGE': 0.11449655462723214, 'RGA_RGR': -0.02652458643499797, 'RGE_RGR': -0.10818469887680082}

Taiwan logit mean per-index: {'RGA_RGE': 0.1272833741028449, 'RGA_RGR': -0.15656992931706049, 'RGE_RGR': -0.22663479481422705}

Taiwan rf half-grid means: RGE-RGR first -0.0280, second -0.2110; RGA-RGR first -0.0734, second -0.0542

German rf mean per-index: {'RGA_RGE': 0.0571742180488091, 'RGA_RGR': -0.03500245754980171, 'RGE_RGR': -0.1523872760067539}

Taiwan rf mean per-index: {'RGA_RGE': 0.14797465133181076, 'RGA_RGR': -0.06381154241812823, 'RGE_RGR': -0.12345121855873739}

### Coding sensitivity, y' = 1 - y

| model | primary rung 6 | flipped rung 6 | primary rung 4 | flipped rung 4 |
|:--|--:|--:|--:|--:|
| logit | -0.5423 | -0.5423 | -0.3789 | -0.3789 |
| rf | -0.2266 | -0.2266 | -0.3174 | -0.3174 |

### Conditioning checks

| model | arm | sd frozen | sd full recompute | ratio | mc band |
|:--|:--|--:|--:|--:|:--|
| logit | arithmetic | 0.001956 | 0.001979 | 1.012 | [0.966, 1.057] |
| logit | geometric | 0.002389 | 0.002454 | 1.027 | [0.954, 1.106] |
| logit | rms | 0.001454 | 0.001553 | 1.068 | [1.029, 1.104] |
| rf | arithmetic | 0.002183 | 0.002442 | 1.119 | [1.009, 1.255] |
| rf | geometric | 0.003028 | 0.003221 | 1.064 | [0.974, 1.170] |
| rf | rms | 0.001626 | 0.001923 | 1.183 | [1.056, 1.317] |

### Scalar construction, both conditioning schemes

| model | scheme | RGA-RGE | RGA-RGR | RGE-RGR |
|:--|:--|--:|--:|--:|
| logit | fixed_draw | +0.2093 | +0.2224 | +0.5028 |
| logit | redraw | +0.2093 | +0.1909 | +0.4149 |
| rf | fixed_draw | +0.2191 | +0.1315 | +0.4688 |
| rf | redraw | +0.2191 | +0.1495 | +0.4189 |

### Verdict

verdict: INCONCLUSIVE

generalises_conditions:
  - (a) rung 4 negative on both models: True
  - (b) rung 5 positive on both models: True
  - (c) rung 6 negative on both models: True
  - (d) delta understatement of the three volume variants within 6 pp of zero on both models: False
  - (e) chain understatement >= 10 pp and rho > 0 under both schemes: True

failure_conditions:
  - (a') rung 4 positive on both models: False
  - (b') rung 6 positive on both models: False
  - (c') volume delta understatement exceeds +15 pp on both models (read as: the largest of the three variants exceeds +15 on each model): False
  - (d') chain understatement below 5 pp under both schemes: False

inconclusive_triggers:
  - quantities_inside_two_mc_standard_errors_of_zero: []
  - models_disagree_in_sign_on_rung_4: False
  - models_disagree_in_sign_on_rung_6: False
  - coding_sensitivity_flips_rung_6: False
  - coding_sensitivity_flips_rung_4: False

total runtime: 3321.8 s (55.4 min)
