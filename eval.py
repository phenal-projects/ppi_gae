import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from imblearn.over_sampling import ADASYN


def class_test(embeddings, labels, val_set=None):

    """Provides classification report with given embeddings and labels.

    Parameters
    ----------
    embeddings : np.ndarray
        An array with embeddings
    labels_data : np.ndarray
        Array with with 'id' column and binary
        labels in each column. Ids must match the position of objects
        in the embeddings array.
    classes : list of str
        Columns in the labels_data to use
        val_set (set of int, optional): Ids in the validation set.
        Defaults to None.

    Returns
    -------
    dict
        Classification report.
    """
    train_data = embeddings
    # train/val/test split
    if val_set is None:
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, labels, test_size=0.4, random_state=42
        )
    else:
        val_mask = np.arange(labels.shape[0]).isin(val_set)
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
    return classification_report(
        test_labels, np.stack(preds).T, zero_division=False, output_dict=True
    )
