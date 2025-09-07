## Downstream Prediction

Two complementary runners are provided:

### Option A — Train & External Validate (save models + ROC figs)

**Script:** `cohort_experiment.py`

- **Train on main cohort (pick one embedding set per run)**

```bash
# Supervised embedding
python cohort_experiment.py   --train-main   --main-baseline data/main/baseline.csv   --embed-supervised data/main/supervised.csv   --models-dir outputs/main/models   --figs-dir outputs/main/figs

# Unsupervised embedding(s)  (CSV or NPY; if multiple given, the first is used for training)
python cohort_experiment.py   --train-main   --main-baseline data/main/baseline.csv   --embed-unsupervised data/main/unsup_a.csv data/main/unsup_b.csv   --models-dir outputs/main/models   --figs-dir outputs/main/figs
```

- **Predict on external cohort with saved models**

```bash
python cohort_experiment.py   --predict-external   --external-baseline data/external/baseline.csv   --external-embed-unsupervised data/external/unsup_a.npy data/external/unsup_b.npy   --models-dir outputs/main/models   --figs-dir outputs/external/figs
```

- **Train + External Validate in one pass**

```bash
python cohort_experiment.py   --train-main --predict-external   --main-baseline data/main/baseline.csv   --embed-supervised data/main/supervised.csv   --external-baseline data/external/baseline.csv   --external-embed-supervised data/external/supervised.csv   --models-dir outputs/main/models   --figs-dir outputs/external/figs
```

**Outputs:**  
- `outputs/main/models/{LR,RF,GB}.pkl` — full trained pipelines  
- `outputs/external/figs/<dataset>.png` — overlaid ROC curves  

---

### Option B — Cross-Validated Benchmarking (nested CV + GridSearch)

**Script:** `af_embed_cv.py`

```bash
python af_embed_cv.py   --baseline data/main/baseline.csv   --supervised data/main/supervised.csv   --unsupervised data/main/unsup_a.csv data/main/unsup_b.csv   --results outputs/cv/results.csv   --outer-splits 5   --inner-splits 3   --seed 42
```

**Outputs:**  
- `outputs/cv/results.csv` with rows:
```
dataset,model,auc_mean,auc_std,f1_mean,f1_std
```

---

**Notes:**  
- `target` column must be present in baseline CSV.  
- Option A accepts unsupervised embeddings as **CSV or NPY**; Option B expects **CSV**.  
- To compare multiple embeddings with Option A, run multiple times with different `--models-dir`.
