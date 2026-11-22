# ML Spam Classifier: Training Pipeline

This Google Colab notebook trains the machine learning model behind the
[Spam Filter Engine](https://github.com/pridem755/email-spam-filtering-backend), the Django backend
that classifies incoming messages as **spam** or **ham**.

The notebook covers the full process: loading the data, exploration, preprocessing,
training, evaluation and tuning. It exports a model file that the
[Spam Filter Engine](https://github.com/pridem755/email-spam-filtering-backend) loads for inference.

## Pipeline overview

1. Load the merged dataset
2. Exploratory data analysis
3. Text preprocessing
4. TF-IDF feature engineering
5. Train/test split
6. Model training, with hyperparameter tuning when needed
7. Evaluation and metrics
8. Threshold optimization
9. Error analysis
10. Model export for Django

## Dataset

**The dataset is not included in this repo** because the CSV is about 150 MB.

The notebook expects a CSV with two columns:

| Column    | Description                          |
|-----------|--------------------------------------|
| `message` | Raw message/email text               |
| `label`   | `spam` or `ham`                      |

## Exploratory data analysis

- Class distribution, with bar and pie charts and the spam/ham balance ratio
- Message length and word count distributions by class, with histograms and box plots
- Word clouds of the most common words in ham and spam messages

## Preprocessing

The `preprocess_text()` function:
- Lowercases the text
- Replaces URLs with `URL_TOKEN`
- Replaces email addresses with `EMAIL_TOKEN`
- Replaces phone numbers with `PHONE_TOKEN`
- Collapses extra whitespace

It also removes outlier messages of 5,000 or more characters, which the box plots
identified.

## Features and model

The model is a scikit-learn `Pipeline`:

- **`TfidfVectorizer`**: unigrams and bigrams, `max_features=10000`, `sublinear_tf=True`,
  English stop words removed, `min_df=2`, `max_df=0.95`
- **`LogisticRegression`**: `class_weight='balanced'`, `solver='lbfgs'`, `max_iter=2000`

The data is split 80/20 with stratification (`random_state=42`).

If the baseline misses its targets (accuracy ≥ 0.97, precision ≥ 0.95,
recall ≥ 0.93), `GridSearchCV` tunes `max_features`, `ngram_range`, `C` and
`class_weight`.

## Evaluation

The notebook reports:
- Accuracy, precision, recall, F1-score and ROC-AUC
- A classification report and confusion matrix
- An ROC curve
- 5-fold cross-validation
- Threshold optimization from 0.70 to 0.80, picking the best F1 with precision ≥ 0.95
- Error analysis of false positives and false negatives
- Feature importance: the top 20 spam and ham indicator terms

Results from the saved run:

| Metric | Value |
|---|---|
| Accuracy | 0.972 |
| Precision (spam) | 0.971 |
| Recall (spam) | 0.975 |
| F1-score (spam) | 0.973 |
| ROC-AUC | 0.997 |
| 5-fold CV accuracy | 0.971 ± 0.002 |
| Recommended threshold | 0.70 |

## Output

The notebook writes two files:
- `spam_classifier_v1.joblib`: the trained pipeline (TF-IDF + Logistic Regression)
- `model_metadata_v1.json`: version, training date, sample counts, metrics and the
  recommended threshold

These files go into the [Spam Filter Engine](https://github.com/pridem755/email-spam-filtering-backend)
backend. The last cell reloads the saved model and runs a test prediction.

## How to run

This notebook is built for Google Colab. The first cell installs every required
package, so no separate `requirements.txt` is needed.

1. Open `ml_training_pipeline.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Upload your dataset to Google Drive as `MyDrive/spam_dataset.csv`.
3. Run all cells. When asked, allow Colab to mount your Drive. The model and
   metadata files download automatically at the end.
