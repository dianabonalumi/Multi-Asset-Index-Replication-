# Early Warning System for Financial Markets

An anomaly-detection pipeline that flags weekly **market stress episodes** from 43 market features (equity indices, government bond yields, credit indices, FX, commodities) over 2000–2021.

> Course project, *Fintech* — Politecnico di Milano (Business Case 4).
> **Team / my contribution:** _TODO: list the team members and describe your own part._

## Use case

Quantitative finance, precision-first. A false alarm forces an unnecessary market exit (transaction costs and missed upside), while a missed crisis is tolerable if other risk layers exist.

**Primary metric:** `Prec@R≥0.50`, i.e. the maximum precision achievable while keeping recall at or above 0.50.

## Data

The notebook expects `Dataset4_EWS.xlsx` with two sheets: `Markets` (43 weekly features plus the label `Y`) and `Metadata`.
The data are sourced from Bloomberg through the course material and are **not included** in this repository.

| Property | Value |
|---|---|
| Observations | 1,111 weeks (Jan 2000 – Apr 2021) |
| Features | 43 market variables |
| Missing values | 0 |
| Stress weeks (`Y=1`) | 237 (21.3%), imbalance 3.7:1 |

## Approach

1. **Exploratory analysis**: class imbalance, distributions, outliers, correlations and a data quality card.
2. **Stationarity and preprocessing**: ADF and KPSS tests, transformations, StandardScaler vs RobustScaler.
3. **Feature engineering**: three feature subsets (equity, macro, full), PCA for correlated clusters, rolling volatility, spreads and z-scores.
4. **Anomaly-aware chronological split**: models are trained on **normal weeks only** (first 60% of the timeline, 472 weeks); the next 20% (222 weeks) is used for threshold calibration; the last 20% (222 weeks, Jan 2017 – Apr 2021) is held out for the final test.
5. **Models**: Gaussian baseline (MVG), GMM, Sparse PCA, Autoencoder and VAE (both tuned with Optuna), OCSVM, LOF, HBOS, and a score-averaging ensemble.
6. **Latent space analysis** with PCA and UMAP.
7. **Domain analysis**: crisis cards, discriminating power per feature (Cohen's d), K-means clustering of anomalies by crisis type.
8. **Interpretability**: SHAP (global and local explanations for the most anomalous weeks).
9. **Signal filters**: a VIX market-regime gate and a persistence filter (signal confirmed only after at least 2 consecutive weeks).

## Main results

Test-set results at the use-case threshold (max precision subject to recall ≥ 0.50):

| Model | Prec@R≥0.5 | AUC-PR | AUC-ROC |
|---|---|---|---|
| MVG baseline | 0.600 | 0.526 | 0.777 |
| **Sparse PCA** | 0.556 | 0.482 | 0.768 |
| HBOS | 0.517 | 0.480 | 0.751 |
| Score-averaging ensemble | 0.500 | 0.535 | 0.775 |
| AE (Optuna) | 0.485 | 0.533 | 0.774 |
| VAE (Optuna) | 0.485 | 0.534 | 0.791 |
| LOF | 0.484 | 0.541 | 0.778 |
| OCSVM | 0.469 | 0.527 | 0.780 |
| GMM (Optuna) | 0.432 | 0.502 | 0.757 |

- On the pure `Prec@R≥0.5` metric the simple Gaussian baseline leads. In the weighted multi-criteria ranking (precision 40%, AUC-PR 25%, AUC-ROC 15%, F1 10%, interpretability 5%, simplicity 5%), **Sparse PCA** ranks first.
- Deep models (AE, VAE) did not beat simpler ones. The notebook attributes this to the regime shift between the training period (GFC and EU sovereign crisis) and the test period (post-QE low volatility, then COVID).
- The **persistence filter** improved precision for 5 of 9 models and degraded none. For Sparse PCA, precision went from 0.556 to 0.737, false positives from 12 to 5, with recall from 0.517 to 0.483.
- Note: in the notebook's comparative table, OCSVM, LOF, HBOS and the ensemble show precision 0.13 and recall 1.00 at the default threshold (they flag every week), so only their `Prec@R≥0.5` and score-based AUC values are informative.

## How to run

The notebook was developed locally and on Colab.

1. Put `Dataset4_EWS.xlsx` in a folder you can access.
2. Edit the `FILE_PATH` variable at the top of the notebook (it currently points to a local path).
3. Install the dependencies and run the notebook top to bottom.

```bash
pip install pandas numpy scipy scikit-learn statsmodels torch xgboost optuna \
            pyod umap-learn pacmap captum shap matplotlib seaborn openpyxl
```

Outputs saved by the notebook: `data_quality_card.csv` and `p3_results_ranked.csv`.

## Repository contents

- `Business_Case4_FINAL.ipynb`: full pipeline, from EDA to model comparison.

## Limitations

- The test set contains few stress weeks, so precision estimates are noisy.
- The label `Y` is defined externally; results depend on how stress periods are defined.
- Conclusions about deep vs simple models hold for this short series with structural breaks and may not generalise.
