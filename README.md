# Time Series Forecaster — Daily Sales

Forecast daily sales from order-line data using classical (Prophet) and deep-learning (LSTM) approaches.

## Project structure

```
time-series-forecaster/
├── data/
│   └── raw/
│       └── sales_data.csv          # original order-line data
├── notebooks/
│   ├── 01_exploration.ipynb        # EDA, gap analysis, naive baseline
│   ├── 02_prophet.ipynb             # Prophet models + components + CV
│   ├── 03_LSTM.ipynb                # LSTM model
│   └── results/
│       └── test_scores.json        # metrics for final comparison
├── src/                            # (placeholder for future packaging)
├── requirements.txt
└── README.md
```

## Pipeline (common to all models)

1. Load `sales_data.csv`
2. Keep only completed revenue (`STATUS ∈ {Shipped, Resolved}`)
3. Aggregate to **daily sales**
4. Keep the series **irregular** — missing calendar dates are **not** zero-filled  
   (≈ 65 % of weekdays have no records; zero-filling would teach the model that sales collapse to zero)
5. Temporal train/test split on observed days only (≈ 80 / 20)
6. Evaluate with MAE, RMSE and MAPE on the exact same test window

## Results (test set)

| Model                        | MAE    | RMSE   | MAPE   |
|------------------------------|--------|--------|--------|
| Naive baseline               | 25,532 | 31,122 | 265.2% |
| Prophet (default)            | 21,965 | 26,272 | 360.1% |
| **Prophet (yearly seasonality)** | **17,207** | **20,489** | **202.0%** |
| LSTM (lookback=14)           | 18,475 | 22,658 | 276.6% |

- All values in **dollars**.
- MAPE is high for every model because the series contains many low-value days; **MAE / RMSE are the primary metrics**.
- Best model: **Prophet with yearly seasonality**.

## Key observations

- **Trend** — smooth upward movement; consistent with genuine growth rather than noise.
- **Weekly** — sparse weekends, busier mid-week (matches the gap analysis in notebook 01).
- **Yearly** — clear Oct–Dec peak. The peak was fully observed only twice, yet enabling yearly seasonality still improved test MAE.
- **LSTM** — beats the naive baseline and default Prophet, lands close to the yearly Prophet. With only 240 observed days the network is data-hungry; classical seasonality modelling proved the stronger inductive bias here.
- Cross-validation shows error grows with forecast horizon; practical accuracy is limited to roughly 1–2 weeks ahead.

## How to run

```bash
# 1. Create / activate environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Open the notebooks in order
jupyter notebook notebooks/
```

Run the notebooks sequentially (01 → 02 → 03).  
Each notebook rebuilds the same daily series so results stay comparable.

## Requirements

See `requirements.txt`. Main packages:

- pandas, numpy, matplotlib
- scikit-learn
- prophet
- tensorflow / keras

## Next steps (optional)

- Notebook 04 — load `results/test_scores.json` and produce a final side-by-side comparison plot.
- Experiment with longer LSTM lookback or an explicit day-of-week feature.
- Try a hybrid model (Prophet residuals + LSTM).
