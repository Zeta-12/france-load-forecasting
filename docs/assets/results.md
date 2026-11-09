### Point forecasts, 2024-04-01 to 2026-08-31

|  | MAPE (%) | MAE (MW) | RMSE (MW) | Bias (MW) | Daily peak MAE (MW) |
|---|---|---|---|---|---|
| Same hour last week | 6.13 | 3201 | 4675 | 76 | 3306 |
| LightGBM | 1.64 | 834 | 1155 | 78 | 889 |
| CatBoost | 1.58 | 799 | 1078 | 93 | 846 |
| Mean of LightGBM and CatBoost | 1.56 | 789 | 1076 | 86 | 843 |
| RTE J-1, as published | 2.70 | 1322 | 1651 | -1065 | 1586 |
| RTE J-1, level-adjusted | 1.59 | 792 | 1039 | 6 | 875 |
| Mean of this model and RTE J-1 (adjusted) | 1.27 | 640 | 862 | 46 | 709 |

### Weather input of the target day (LightGBM)

|  | MAPE (%) | MAE (MW) | RMSE (MW) | Bias (MW) | Daily peak MAE (MW) |
|---|---|---|---|---|---|
| ERA5 (observed weather, upper bound) | 1.59 | 801 | 1094 | 72 | 850 |
| ECMWF IFS day-1 forecast, debiased | 1.64 | 834 | 1155 | 78 | 889 |
| ECMWF IFS day-1 forecast, raw | 1.69 | 856 | 1177 | 203 | 909 |
| ECMWF IFS day-2 forecast, debiased | 1.66 | 846 | 1183 | 78 | 906 |

### Real-time series, 2026-09-08 to 2026-11-07

|  | MAPE (%) | MAE (MW) | RMSE (MW) | Bias (MW) | Daily peak MAE (MW) |
|---|---|---|---|---|---|
| LightGBM, absolute load target | 1.52 | 709 | 958 | 237 | 764 |
| LightGBM | 1.49 | 697 | 957 | 115 | 707 |
| CatBoost | 1.48 | 694 | 961 | 142 | 682 |
| Mean of LightGBM and CatBoost | 1.44 | 674 | 931 | 129 | 673 |
| RTE J-1, as published | 1.33 | 608 | 780 | 189 | 631 |
| Mean of this model and RTE J-1 | 1.17 | 543 | 717 | 159 | 553 |

### Target (LightGBM, day-1 weather)

|  | MAPE (%) | MAE (MW) | RMSE (MW) | Bias (MW) | Daily peak MAE (MW) |
|---|---|---|---|---|---|
| Absolute load target | 1.63 | 829 | 1130 | 249 | 878 |
| Target relative to last week (default) | 1.64 | 834 | 1155 | 78 | 889 |

### Worst days (Mean of LightGBM and CatBoost)

|  | Day | MAPE (%) | RTE J-1 MAPE (%) | Bias (MW) |
|---|---|---|---|---|
| 2025-11-22 | Sat | 8.33 | 2.89 | -5653 |
| 2025-05-01 | Thu (holiday) | 5.95 | 5.84 | 2158 |
| 2024-11-25 | Mon | 5.93 | 2.24 | 3173 |
| 2025-11-23 | Sun | 5.75 | 1.95 | -3616 |
| 2026-01-14 | Wed | 5.63 | 1.87 | 3482 |
| 2024-12-24 | Tue | 5.31 | 1.44 | 3070 |
| 2025-04-13 | Sun | 5.24 | 5.56 | -2190 |
| 2026-05-03 | Sun | 5.21 | 5.09 | -2053 |
| 2026-04-19 | Sun | 5.17 | 5.09 | -2077 |
| 2024-11-30 | Sat | 5.04 | 4.11 | 2887 |

### 90 % intervals

|  | Coverage (%) | Worst month (%) | Mean width (MW) | Width (% of load) | Interval score (MW) |
|---|---|---|---|---|---|
| LightGBM quantile regression | 69.1 | 56.8 | 2495 | 5.0 | 6604 |
| Rolling conformal, per hour | 92.5 | 83.3 | 3960 | 7.8 | 4858 |
| Rolling conformal | 88.5 | 78.9 | 3303 | 6.6 | 4730 |
| Adaptive conformal (ACI), used live | 89.9 | 81.0 | 3561 | 7.1 | 4732 |
| Conformalized quantile regression | 89.6 | 79.3 | 4042 | 8.0 | 5366 |
