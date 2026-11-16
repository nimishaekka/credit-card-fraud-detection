# Credit Card Fraud Detection

End-to-end fraud detection on a highly imbalanced dataset (0.17% fraud), using a production-style time-based split and a decision threshold chosen by cost.

## Highlights
- **Time-based split (70/15/15):** the model trains on past transactions and is tested on later ones, so no future data leaks into training.
- **Imbalance handled without resampling:** class weighting (`scale_pos_weight`) instead of SMOTE, which can leak information on time-ordered data.
- **Evaluated with precision-recall:** at this fraud rate, a model that never flags fraud would still score 99.8% accuracy.
- **Cost-based threshold:** a missed fraud costs the transaction amount and a false alarm costs a $5 review. The threshold is tuned on validation data, then applied unchanged to test.
- **Explainability:** SHAP summary of the drivers behind each prediction.

## Key EDA findings
- Fraud amounts cluster at both ends: many tiny (~$1) transactions, consistent with card-testing, plus a cluster near $100.
- The fraud rate peaks in the early hours, at about 6–10× the average.

## Results (test set)
Test set: the last 15% of transactions by time (42,722 rows, 52 frauds).

| Model | PR-AUC | ROC-AUC |
|---|---|---|
| Logistic regression (baseline) | 0.691 | 0.978 |
| **LightGBM** | **0.740** | 0.952 |

At the cost-optimal threshold (chosen on validation data):
- **Recall 73%, precision 60%:** 38 of 52 frauds caught, with 25 false alarms in 42,722 transactions
- **Total cost cut by 55%:** $2,745 with the model vs. $6,169 with no model
- **57.5% of fraud dollars caught** ($3,549 of $6,169)

**Why ROC-AUC and PR-AUC disagree:** with 0.12% fraud in the test set, ROC-AUC is dominated by how well legit transactions are ranked. PR-AUC measures performance on the fraud class itself, which is where LightGBM is clearly better.

**Key finding:** the model catches 73% of fraud *cases* but only 57.5% of fraud *dollars*. The frauds it misses tend to be larger transactions.

## What drives the model (SHAP)
- **V14** has the strongest single effect: low values push predictions sharply toward fraud.
- **High V4** values increase fraud risk.
- The engineered **hour** feature ranks #1 overall. With only 2 days of data, this time-of-day pattern needs validating on a longer period.
- **Smaller amounts** push slightly toward fraud, which fits the finding that more high-value fraud is missed.

## Run it
**On Kaggle (easiest):** open the notebook in a Kaggle Notebook, add the [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset as an input, and click Run All. The notebook finds `creditcard.csv` automatically.

**Locally:** (the notebook falls back to `data/creditcard.csv`)
```bash
pip install -r requirements.txt
# Download creditcard.csv from Kaggle into ./data/
jupyter notebook credit-card-fraud-detection.ipynb
```

## Project structure
```
├── credit-card-fraud-detection.ipynb
├── requirements.txt
└── README.md
```

## Limitations and next steps
- The test set has only 52 frauds, so the metrics are noisy and a few cases can move them noticeably.
- Missed frauds skew high-value. Next: weight training samples by amount, or optimise directly for dollars caught.
- The dataset covers only 2 days, so concept drift can't be tested.
- The PCA-anonymised features hide raw signals. On real data I'd add velocity features (for example, card transactions in the last hour) and merchant risk.
- Next: serve the model behind a FastAPI endpoint with a Docker image.
