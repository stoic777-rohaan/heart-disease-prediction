# ❤️ Heart Disease Prediction

A machine learning project that predicts the likelihood of heart disease from
patient clinical data, built and evaluated on the UCI-style Heart Failure
Prediction dataset.

**🔗 Live app:** [https://heartbeat-predictor-1.preview.emergentagent.com/?utm_source=share]

## What's in this repo

| Path | Description |
|---|---|
| `notebooks/heart_ml.ipynb` | Full ML workflow: EDA, cleaning, encoding, model comparison, and export of the final model |
| `data/heart.csv` | Training dataset |
| `models/` | Saved model, scaler, and expected feature columns (`.pkl`) |
| `app/app.py` | Streamlit prototype I built to serve the model locally |

## Project workflow

1. **EDA** — distribution checks on Age, RestingBP, Cholesterol, MaxHR; class
   balance of the target; correlation heatmap.
2. **Data cleaning** — replaced physiologically invalid zero-values in
   `Cholesterol` and `RestingBP` with the column mean; removed duplicates.
3. **Encoding** — one-hot encoded categorical features with `pd.get_dummies`.
4. **Modeling** — trained and compared 5 classifiers (Logistic Regression,
   Gaussian Naive Bayes, Decision Tree, SVM, KNN) on accuracy and F1-score.
5. **Export** — serialized the best-performing model, scaler, and column
   order with `joblib` for reuse in the app layer.

## Tech stack

`pandas` · `scikit-learn` · `seaborn` / `matplotlib` · `Streamlit` · `joblib`

## Running the Streamlit prototype locally

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app/app.py
```

## A note on the live app

The model training and data science work in this repo — the notebook, the
cleaning/encoding decisions, and the model selection — is my own. For the
polished, publicly-hosted version of the app (linked above), I used
[Emergent](https://emergent.sh) to help build the UI and handle deployment,
since app development/deployment wasn't the focus of this project for me.
The Streamlit script in `app/` is my own prototype of the same prediction
logic, meant to show how the model is actually served.
