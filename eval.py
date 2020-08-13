import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import average_precision_score
import xgboost as xgb
from imblearn.over_sampling import ADASYN

import sim_pu


def roc_shuffle(y_true, y_probas, base, rounds=1000):
    """Returns single-sided p-value for a given base value"""
    results = np.zeros(shape=rounds)
    for i in range(rounds):
        results[i] = roc_auc_score(y_true, np.random.permutation(y_probas))
    return np.mean(base > results)


def ap_shuffle(y_true, y_probas, base, rounds=1000):
    """Returns single-sided p-value for a given base value"""
    results = np.zeros(shape=rounds)
    for i in range(rounds):
        results[i] = average_precision_score(
            y_true, np.random.permutation(y_probas)
        )
    return np.mean(base > results)


def class_test(embeddings, labels, val_mask=None, method="cr", enrichment=0):

    """Provides classification report with given embeddings and labels.

    Parameters
    ----------
    embeddings : np.ndarray
        An array with embeddings
    labels : np.ndarray
        Array with with 'id' column and binary
        labels in each column. Ids must match the position of objects
        in the embeddings array.
    val_mask : np.ndarray or None
        Mask of the nodes in the validation set.
    method : str
        'auc' or 'cr' (classification report).
        Defaults to 'cr'.
    enrichment : int
        If greater then zero, stochasticly enriches positive class through KNN
        similiarity. The value of the parameter is k for KNN.

    Returns
    -------
    dict
        Classification report / auc for every label.
    """
    train_data = embeddings
    # train/val/test split
    if val_mask is None:
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, labels, test_size=0.4, random_state=42
        )
    else:
        val_data = train_data[val_mask]
        train_data = train_data[np.logical_not(val_mask)]
        val_labels = labels[val_mask]
        train_labels = labels[np.logical_not(val_mask)]
    val_data, test_data, val_labels, test_labels = train_test_split(
        val_data, val_labels, test_size=0.5, random_state=42
    )

    resampler = ADASYN(random_state=42)
    # binary model for each class in labels
    preds = []
    probas = []
    for col in range(labels.shape[1]):
        cur_train_labels = train_labels[:, col]
        if enrichment > 0:
            cur_train_labels = sim_pu.prob_labels(
                cur_train_labels,
                sim_pu.knn_prob(train_data, cur_train_labels, enrichment),
            )
        x, y = resampler.fit_resample(train_data, cur_train_labels)
        model = xgb.XGBClassifier(
            objective="binary:logistic", nthread=11,
        ).fit(
            x,
            y,
            eval_set=[(val_data, val_labels[:, col])],
            eval_metric="auc",
            early_stopping_rounds=5,
            verbose=False,
        )
        preds.append(model.predict(test_data))
        probas.append(model.predict_proba(test_data)[:, 1])
    if method == "cr":
        return classification_report(
            test_labels,
            np.stack(preds).T,
            zero_division=False,
            output_dict=True,
        )
    if method == "auc":
        res = dict()
        for i, (y, y_probas) in enumerate(zip(test_labels.T, probas)):
            u = roc_auc_score(y, y_probas)
            v = average_precision_score(y, y_probas)
            res[i] = {
                "roc": u,
                "ap": v,
                "roc_pval": roc_shuffle(y, y_probas, u),
                "ap_pval": ap_shuffle(y, y_probas, v),
            }
        return res
