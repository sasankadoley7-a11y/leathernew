# Pineapple Leather Analytics Studio

A Streamlit app for descriptive, diagnostic, predictive, and prescriptive analytics for a pineapple leaf fiber luxury-material startup.

## What the app does

- Loads the bundled synthetic dataset with 2,500 respondents, or your own CSV/XLSX file
- Runs descriptive and diagnostic analysis
- Trains a classification model to predict customer inclination toward the business
- Tracks classification metrics: accuracy, precision, recall, F1-score, ROC-AUC, ROC curve, confusion matrix, and feature importance
- Trains a regression model to predict spending score
- Builds K-Means customer segments
- Runs association rule mining with support, confidence, and lift
- Scores future customer uploads and recommends marketing actions

## Files

- `app.py` - Streamlit application
- `sample_training_data.csv` - bundled synthetic dataset
- `generate_sample_data.py` - generator used to create the sample dataset
- `requirements.txt` - packages for deployment
- `future_customer_template.csv` - example file for future scoring uploads

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud deployment

1. Upload all files in this folder to the root of your GitHub repository.
2. In Streamlit Community Cloud, connect the repo.
3. Set the main file path as `app.py`.
4. Deploy.

## Expected schema for future customer uploads

Use the same input feature columns as the training data, excluding target columns:

- `purchase_intent_binary`
- `purchase_intent_5level`
- `estimated_spending_score`
- `hidden_persona`

The app derives helper fields like `willingness_proxy` and `concern_index` automatically.
