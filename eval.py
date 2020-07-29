import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
from imblearn.over_sampling import ADASYN


def class_test(embeddings, labels, val_mask=None, method="cr"):

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

    Returns
    -------
    dict
        Classification report / auc for every label
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

    ada = ADASYN(random_state=42)
    # binary model for each class in labels
    preds = []
    probas = []
    for col in range(labels.shape[1]):
        x, y = ada.fit_resample(train_data, train_labels[:, col])
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
            res[i] = roc_auc_score(y, y_probas)
        return res
