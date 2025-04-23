<!-- README.md  –  copy–paste-ready -->

# Causal-VAE Stress-Testing & Portfolio Lab  
*A mini-research repo for the MBZUAI ML-808 course project (2025)*  

---

## 1. What is this?

We build a **causally-aware Variational Auto-Encoder (Causal-VAE)** that  
1. learns a small macro-financial DAG,  
2. generates realistic stress scenarios via `do()`-style shocks, and  
3. optimises a simple equity/bond portfolio for 95 % Conditional VaR.  

The full pipeline runs in **\< 10 minutes on any single GPU** (or ~25 min CPU-only).

---

## 2. Project Road-map  

| Step | Folder / Script | Output artefacts |
|------|-----------------|------------------|
| **1 Data & EDA** | `notebooks/01_data_collection.ipynb` | `data/processed/final_processed_data.csv` |
| **2 Causal DAG** | `notebooks/02_dag_learning.ipynb` | `dag_edges.json`, `results/dag/…png` |
| **3 Baselines** | `notebooks/03_baselines.ipynb` | VAR & LSTM-VAE metrics |
| **4 Causal-VAE** | `src/train_cvae.py` | `causal/cvae_best.pt`, `cvae_metrics.json` |
| **5 Scenarios** | `src/generate_scenarios.py` | `scenarios/scenario_paths.npz`, figure |
| **6 Portfolio** | `src/optimise_portfolio.py` | `portfolio/optimal_weights.csv`, bar-plot |
| **7 Evaluation** | `src/evaluate.py` | `results/diag_metrics.json`, Q–Q plots |
| **8 Report & Slides** | `report/`, `slides/` | NeurIPS PDF, pptx/Canva link |

---

## 3. File Structure (v-1)

```
ML808_causality/ │ ├── data/ │ ├── raw/ # immutable downloads │ └── processed/ │ ├── notebooks/ # exploratory Jupyter work │ ├── src/ # runnable python scripts │ ├── train_cvae.py │ ├── generate_scenarios.py │ ├── optimise_portfolio.py │ └── evaluate.py │ ├── causal/ # trained model + plots ├── scenarios/ # .npz stress paths ├── portfolio/ # optimisation outputs │ ├── report/ # LaTeX → NeurIPS PDF └── slides/ # 15-min deck (PDF / Canva link)
```


---

## 4. Quick Start

```bash
git clone https://github.com/YourHandle/ML808_causality.git
cd ML808_causality

# 1️⃣  create env
conda env create -f environment.yml   # or  pip install -r requirements.txt
conda activate ml808

# 2️⃣  download + preprocess
jupyter nbconvert --to notebook --execute notebooks/01_data_collection.ipynb

# 3️⃣  train the Causal-VAE
python src/train_cvae.py              # ≈ 6 min on 4090, 23 min CPU

# 4️⃣  generate scenarios & optimise portfolio
python src/generate_scenarios.py
python src/optimise_portfolio.py

# 5️⃣  run diagnostics
python src/evaluate.py

Figures and CSVs appear under causal/, scenarios/, portfolio/, results/.
```
***
## 5. Data Lineage Preview

| Column              | Source Ticker | Provider       | Note                             |
|---------------------|---------------|----------------|----------------------------------|
| SP500_Returns       | ^GSPC         | Yahoo Finance  | Monthly log-return (%)          |
| GS10_Level          | GS10          | FRED           | 10-yr Treasury yield            |
| FEDFUNDS_Level      | FEDFUNDS      | FRED           | Effective policy rate           |
| FEDFUNDS_BpsChange  | calc.         | —              | 1-month diff (bp)               |
| Inflation_YoY       | CPIAUCSL      | FRED           | 12-m YoY % change               |

> Missing CPI values (1998-04 … 1999-03) → rows trimmed.  
> **Last refresh:** 2025-04-22 14:15 UTC

---

## 6. Main Results (test 2021-06 → 2025-03)

| Model       | RMSE ↓ | KS avg ↓ | 95 % CVaR gap ↓  |
|-------------|--------|----------|------------------|
| VAR(1)      | 10.4   | —        | —                |
| TimeGAN     | 4.1    | 0.55     | —                |
| Causal-VAE  | 2.4    | 0.33     | ≤ 5 pp           |

> Portfolio weight from ‘curve10bp’ scenario (EQ 6 % / BND 94 %) halves realised CVaR compared to naïve 60/40.

---

## 7. Extending the Project

- **More assets** – Add tickers in `notebooks/01_data_collection.ipynb`, then re-run DAG + training.  
- **Custom shocks** – Edit `SCENARIOS` dict in `src/generate_scenarios.py`.  
- **Different risk metric** – Replace CVaR function in `optimise_portfolio.py`.

---

## 8. Licence & Citation

**Code:** MIT Licence  
**Data:** FRED & Yahoo Finance (public domain for research)

```latex
@misc{KhantaalElmay2025,
  title  = {Causal-VAE Stress Testing},
  author = {Khantaal, Anjali and Elmay, Abdelrahman},
  year   = 2025,
  note   = {\url{https://github.com/Anjali-Khantaal/ML808_causality}}
}
```

## 🙋 9. Contact

Questions, ideas, or bug reports → open an Issue or ping:  
📧 [anjali.khantaal@mbzuai.ac.ae](mailto:anjali.khantaal@mbzuai.ac.ae) | [abdelrahman.elmay@mbzuai.ac.ae](mailto:abdelrahman.elmay@mbzuai.ac.ae)


