import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


def score_predictions(y_test, y_pred, model):
    return {
        'Model': str(model),
        'ROC AUC Score': np.round(roc_auc_score(y_test, y_pred), 3),
        'F1 score': np.round(f1_score(y_test, y_pred > 0.5), 3),
        'Accuracy score': np.round(accuracy_score(y_test, y_pred > 0.5), 3)
    }


def train_and_score_models(train_df, test_df, features, models):
    output_df = pd.DataFrame()
    for model in models:
        X_train, y_train = train_df[features], train_df['label']
        X_test, y_test = test_df[features], test_df['label']

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        scores = score_predictions(y_test, y_pred, model)
        output_df = output_df.append(scores, ignore_index=True)

    return output_df
