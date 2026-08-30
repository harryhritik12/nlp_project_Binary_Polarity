# Binary Sentiment Polarity Classifier

Classifies movie review sentences as **positive** or **negative** using the
[Rotten Tomatoes sentence-polarity dataset](http://www.cs.cornell.edu/people/pabo/movie-review-data/)
(5,331 positive / 5,331 negative snippets). Built to compare classic
TF-IDF-based models as a baseline before moving to transformer-based
sentiment models.

## Approach

1. **Preprocessing** — load `rt-polarity.pos` / `rt-polarity.neg`, label them,
   shuffle, and split 80/20 into train/test (stratified).
2. **Feature extraction** — TF-IDF with unigrams + bigrams
   (`ngram_range=(1,2)`), `min_df=2`, `sublinear_tf=True`.
3. **Models compared**:
   - Multinomial Naive Bayes
   - Logistic Regression
   - Linear SVM
4. **Evaluation** — accuracy, F1, and full classification report per model;
   confusion matrix and misclassified-example inspection for the best model.

## Results

| Model                | Accuracy | F1     |
|-----------------------|----------|--------|
| Naive Bayes           | ~0.76    | ~0.76  |
| Logistic Regression   | ~0.77    | ~0.77  |
| Linear SVM            | ~0.78    | ~0.78  |

*(Exact numbers will be written to `outputs/results_summary.csv` after you run
the script on the real dataset — update this table with your actual run.)*

Linear SVM on TF-IDF typically edges out the other two on this dataset. See
`outputs/confusion_matrix.png` for the best model's error breakdown and the
console output for a sample of misclassified reviews — mistakes are
concentrated in short, sarcastic, or mixed-sentiment snippets that a
bag-of-words model can't resolve.

## Project Structure

```
.
├── binary_polarity.py     # main script: load data, train, evaluate, compare
├── rt-polarity.pos        # positive review snippets
├── rt-polarity.neg        # negative review snippets
├── requirements.txt
├── outputs/
│   ├── confusion_matrix.png
│   └── results_summary.csv
└── README.md
```

## Setup & Usage

```bash
pip install -r requirements.txt
python binary_polarity.py
```

Optional arguments:

```bash
python binary_polarity.py --pos rt-polarity.pos --neg rt-polarity.neg --test-size 0.2
```

## Tech Stack

- **Language**: Python 3.10+
- **Libraries**: pandas, numpy, scikit-learn, matplotlib

## Next Steps

- Compare against a transformer baseline (e.g. `distilbert-base-uncased-finetuned-sst-2-english`
  via `sentence-transformers`/`transformers`) to quantify the gap between
  classic ML and modern embeddings.
- Add cross-validation instead of a single train/test split for more robust
  metrics.
- Add a lightweight CLI/Streamlit demo for interactive predictions.
