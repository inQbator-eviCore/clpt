"""Metrics Calculation module for evaluating models."""

import logging
import os
from math import sqrt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import confusion_matrix, auc, precision_recall_curve, roc_auc_score, average_precision_score
from src.constants.constants import POSITIVE_LABEL, NEGATIVE_LABEL, METRICS_FILE, PROBABILITY_FILE, \
    METRICS_DEFAULT_FILE, PROBABILITY_DEFAULT_FILE
from src.constants.annotation_constants import PROBABILITY, PREDICTION, ACTUAL_LABEL
from src.utils import stringify


class ConfusionMatrixMetrics:
    """Class to calculate the confusion matrix methods to evaluate a model.

    Attributes:
        tn: a float that represents the number of true negatives
        fp: a float that represents the number of false positives
        tp: a float that represents the number of true positives
        fn: a float that represents the number of false negatives
        split_type: a string the represents the type of dataset split
        logger: logger
    """
    def __init__(self, tp, fp, tn, fn, split_type: str = None, logger: logging.Logger = None):
        """Initailize the class to calculate the confusion matric metrics.

        Args:
            tp (float): true posititve
            fp (float): false positive
            tn (float): true negative
            fn (float): false negative
            split_type (str): the type of dataset split
            logger (logging.Logger): logger

        Raises:
            ValueError: if the volume is 0
        """
        self.tn, self.fp, self.fn, self.tp = tn, fp, fn, tp
        self.volume = self.tp + self.fp + self.tn + self.fn
        if self.volume == 0:
            raise ValueError("Evaluation volume should be greater than 0.")

        self.split_type = split_type
        self.logger = logger or logging.getLogger(__name__)
        self.logger.debug("FP:%s, TP:%s, FN:%s, TN:%s, TOTAL: %s", self.fp, self.tp, self.fn, self.tn, self.volume)
        if self.volume < 50:
            self.logger.warning("Total evaluation records are low: %s", self.volume)

    @property
    def confusion_matrix(self):
        """Confusion matrix initialized in init method of the class.

        Returns:
            np.array: The confusion matrix
        """
        return np.array([[self.tn, self.fp], [self.fn, self.tp]])

    @property
    def fpr(self):
        """False positive rate.

        Returns:
            float: The false positive rate as a float between 0 and 1
        """
        return self.fp / (self.fp + self.tn) if (self.fp + self.tn) > 0 else 0.

    @property
    def precision(self):
        """Precision.

        Returns:
            float: The precision as a float between 0 and 1
        """
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.

    @property
    def recall(self):
        """Recall.

        Returns:
            float: The recall as a float between 0 and 1
        """
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.

    @property
    def tpfp_ratio(self):
        """Rate of true positive to false positive.

        Returns:
            float: The ratio of true positive to false positives
        """
        return self.tp / self.fp if self.fp > 0 else 0.

    @property
    def mcc(self):
        r"""Matthews correlation coefficient is a binary classification metric for when there is a high class imbalance.

        https: // en.wikipedia.org/wiki/Matthews_correlation_coefficient
          - 1 = perfect prediction
          - 0 = random prediction
          - -1 = total disagreement

        Returns:
            float: The Matthews correlation coefficient
        """
        mcc_denominator = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        return ((self.tp * self.tn) - (self.fp * self.fn)) / sqrt(mcc_denominator) if mcc_denominator > 0 else 0.

    @property
    def pvr(self):
        """Performance vs. random calculation.

        PVR = approval_rate/false_positive_rate

        Returns:
            float: The performance vs. random
        """
        return self.automation_rate / self.fpr if self.fpr > 0. else 0.

    @property
    def f1(self):
        """F-measure.

        Returns:
           float: Standard F1 measure
        """
        sum_pr = self.precision + self.recall
        return (2. * self.precision * self.recall) / sum_pr if sum_pr > 0. else 0.

    def get_all_metrics(self) -> dict:
        """Return all metrics in a dictionary.

        Returns:
            Dict: contains all of the metrics
        """
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn,
                'precision': self.precision, 'recall': self.recall, 'f1': self.f1,
                'mcc': self.mcc, 'pvr': self.pvr, 'tpfp_ratio': self.tpfp_ratio,
                'volume': self.volume, 'fpr': self.fpr}

    def export_metrics_to_yaml(self, output_dir: str):
        """Write scores to yaml at a given output directory.

        Args:
            output_dir (str): Path to the directory to save yaml
        """
        results = self.get_all_metrics()
        for key, value in results.items():
            if (self.split_type is None):
                self.logger.info("\t%s: %s", key, results[key])
            else:
                self.logger.info("\t%s_%s: %s", self.split_type, key, results[key])
            results[key] = stringify(results[key])

        if (self.split_type is None):
            path = os.path.join(output_dir, METRICS_DEFAULT_FILE)
        else:
            path = os.path.join(output_dir, METRICS_FILE.format(self.split_type))

        with open(path, 'w') as f:
            yaml.add_representer(dict, lambda self, data: yaml.representer.SafeRepresenter
                                 .represent_dict(self, data.items()))
            yaml.dump(results, f)
            if (self.split_type is None):
                self.logger.info("results exported to %s", path)
            else:
                self.logger.info("%s results exported to %s", self.split_type, path)


class ScoreCalculations(ConfusionMatrixMetrics):
    """Calculate metrics for evaluating CDR models.

    Attributes:
       predictions: List[float] of predicted probabilities
       targets: List[float] of ground truth
       threshold: float between 0 and 1 of threshold to be used for classification
    """

    def __init__(self, predictions, targets, threshold: float, split_type: str = None, logger: logging.Logger = None):
        """Initialize the class to score the model.

        Args:
            predictions (List[float]): predicted probabilities
            targets (List[float]): ground truth
            threshold (float): threshold to be used for classification
            split_type (str): the type of dataset split (train, test, valid etc.)
            logger (logging.Logger): logger

        Raises:
            ValueError: if the volume is 0
        """
        if len(targets) == 0 or len(predictions) == 0:
            raise ValueError("Evaluation volume should be greater than 0.")

        self.split_type = split_type
        self.y_true, self.y_pred = targets, predictions
        predictions = np.array(predictions)
        predictions[predictions >= threshold] = POSITIVE_LABEL
        predictions[predictions < threshold] = NEGATIVE_LABEL

        self.threshold = threshold
        self._confusion_matrix = confusion_matrix(targets, predictions).ravel()
        self.tn, self.fp, self.fn, self.tp = self._confusion_matrix

        super().__init__(self.tp, self.fp, self.tn, self.fn, split_type, logger)

    @property
    def auc_roc(self):
        """Calculate AUC RoC.

        Returns:
            float: Area under curve for Receiver operator characteristic
        """
        return roc_auc_score(self.y_true, self.y_pred)

    @property
    def auc_pr(self):
        """Calculate AUC PR which is better than AUC RoC for imbalanced dataset.

        Returns:
            float: Area under curve for Precision Recall
        """
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred)
        return auc(recall, precision)

    @property
    def average_precision(self):
        """Calculate AUC PR which is better than AUC RoC for imbalanced dataset.

        Returns:
            float: Area under curve for Precision Recall
        """
        return average_precision_score(self.y_true, self.y_pred)

    def get_all_metrics(self) -> dict:
        """Return all metrics in a dictionary.

        Returns:
            Dict: All metrics
        """
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn, 'tn': self.tn,
                'precision': self.precision, 'recall': self.recall, 'f1': self.f1,
                'auc_roc': self.auc_roc, 'auc_pr': self.auc_pr, 'mcc': self.mcc,
                'pvr': self.pvr, 'tpfp_ratio': self.tpfp_ratio, 'volume': self.volume,
                'fpr': self.fpr, 'average_precision': self.average_precision,
                'threshold': self.threshold}

    def export_predicted_probas_to_csv(self, output_dir: str, extra_cols):
        """Write predicted probabilities to csv with header.

        Args:
            output_dir (str): Path to the directory to save
            extra_cols (List[Tuple[str, List[float]]): list of tuple of columns and values
        """
        dataset = pd.DataFrame()
        dataset[PROBABILITY] = np.array(self.y_pred)
        dataset[ACTUAL_LABEL] = np.array(self.y_true)
        dataset.loc[dataset[PROBABILITY] >= self.threshold, PREDICTION] = POSITIVE_LABEL
        dataset.loc[dataset[PROBABILITY] < self.threshold, PREDICTION] = NEGATIVE_LABEL

        if (self.split_type is None):
            path = os.path.join(output_dir, PROBABILITY_DEFAULT_FILE)
        else:
            path = os.path.join(output_dir, PROBABILITY_FILE.format(self.split_type))

        for col_name, values in extra_cols:
            dataset[col_name] = pd.Series(values, index=dataset.index)
        dataset.to_csv(path, index=False, header=True, date_format='%Y/%m/%d %H:%m:%S')
        if (self.split_type is None):
            self.logger.info("scores exported to %s", path)
        else:
            self.logger.info("%s scores exported to %s", self.split_type, path)


def count_stats(y_actual, y_pred):
    """Get the number of FP,FN,TP, TN

    Args:
        y_actual: pd.Series that contains the true labels
        y_pred: pd.Series that contains the predicted labels

    Returns:
        FP: Counts of false positives
        FN: Counts of false negatives
        TP: Counts of true positives
        TN: Counts of true negatives
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)):
        if y_actual[i] == y_pred[i] == 1:
            TP += 1
        elif (y_pred[i] == 1) and (y_actual[i] != y_pred[i]):
            FP += 1
        elif y_actual[i] == y_pred[i] == 0:
            TN += 1
        elif (y_pred[i] == 0) and (y_actual[i] != y_pred[i]):
            FN += 1
    return TP, FP, TN, FN


def compare_entity(gold_standards, predictions):
    """Compare the gold standard entity with the predicted entity.

    Args:
        gold_standards: a dictionary that contains the gold standard entity with the key equals the doc/clao name(s)
                        and the value is a list of entities
        predictions: a dictionary that contains the predicted entities

    Returns:
        FP: Counts of false positives
        FN: Counts of false negatives
        TP: Counts of true positives
        TN: Counts of true negatives
    """
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    if len(gold_standards.keys()) != len(predictions.keys()):
        raise ValueError("The number of documents in gold standards does not match with the number of document "
                         "in the predictions.")
    if len(gold_standards) == 0 or len(predictions) == 0:
        raise ValueError("Evaluation volume should be greater than 0.")

    for doc_name in gold_standards.keys():
        for entity in predictions[doc_name]:
            if entity in gold_standards[doc_name]:
                TP += 1
            else:
                FP += 1
        for true_entity in gold_standards[doc_name]:
            if true_entity not in predictions[doc_name]:
                FN += 1
    return TP, FP, TN, FN
