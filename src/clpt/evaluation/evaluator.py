"""Evaluate the model and output metrics"""
import logging
import pandas as pd
from functools import reduce
from src.clpt.evaluation.score_calculations import ConfusionMatrixMetrics, ScoreCalculations, \
    ScoreCalculationsMultiLabels, compare_entity
from src.constants.annotation_constants import PREDICTION, PROBABILITY, ACTUAL_LABEL, DOCUMENT_NAME
from src.utils import extract_group_entity_from_claos, extract_gold_standard_outcome_from_claos, \
    extract_predicted_probability_from_claos, extract_gold_standard_multi_outcome_from_claos, \
    extract_predict_multi_outcome_from_claos, extract_prediction_from_claos
from src.clao.text_clao import METRICS
from operator import itemgetter
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates and saves results of a models to a YAML file.

    Attributes:
        outcome_type: if the outcome is binary or entity etc
        target_dir: a location to export the metrics YAML file
        claos: a list of CLAOs
        threshold: the threshold for classification used to categorize probability into binary outcome
        split_type: test, train or evaluation
    """
    def __init__(self, outcome_type, target_dir, claos, threshold=None, split_type: str = None):
        # TODO: make constant variables to an enum
        self.split_type = split_type
        self.target_dir = target_dir
        self.threshold = threshold
        if outcome_type:
            self.outcome_type = outcome_type
        if outcome_type == 'multi_labels' or outcome_type == 'binary':
            actual_label_from_claos = extract_gold_standard_outcome_from_claos(claos)
            df_gold_standard = pd.DataFrame({DOCUMENT_NAME: list(actual_label_from_claos.keys()),
                                             ACTUAL_LABEL: list(actual_label_from_claos.values())})
            if threshold:
                predicted_probs_from_claos = extract_predicted_probability_from_claos(claos)
                df_predicted_probs = pd.DataFrame({DOCUMENT_NAME: list(predicted_probs_from_claos.keys()),
                                                   PROBABILITY: list(predicted_probs_from_claos.values())})
                df_predicted_probs.dropna(inplace=True)
                dfs_to_merge = [df_gold_standard, df_predicted_probs]
                self.df_merged = reduce(lambda left, right: pd.merge(left, right, on=[DOCUMENT_NAME], how='inner'),
                                        dfs_to_merge)
                self.predictions = self.df_merged[PROBABILITY]
            else:
                predictions_from_claos = extract_prediction_from_claos(claos)
                df_predicted = pd.DataFrame({DOCUMENT_NAME: list(predictions_from_claos.keys()),
                                             PREDICTION: list(predictions_from_claos.values())})
                dfs_to_merge = [df_gold_standard, df_predicted]
                self.df_merged = reduce(lambda left, right: pd.merge(left, right, on=[DOCUMENT_NAME], how='inner'),
                                        dfs_to_merge)
                self.predictions = self.df_merged[PREDICTION]
            self.gold_standards = self.df_merged[ACTUAL_LABEL]

        if outcome_type == 'multi_cols':
            actual_label_from_claos = extract_gold_standard_multi_outcome_from_claos(claos)
            df_gold_standard = pd.DataFrame.from_dict(actual_label_from_claos, orient='index')
            df_gold_standard = df_gold_standard.rename_axis('doc_name').reset_index()
            predictions_from_claos = extract_predict_multi_outcome_from_claos(claos)
            df_predicted_probs = pd.DataFrame.from_dict(predictions_from_claos, orient='index')
            df_predicted_probs = df_predicted_probs.rename_axis('doc_name').reset_index()
            dfs_to_merge = [df_gold_standard, df_predicted_probs]
            self.df_merged = reduce(lambda left, right: pd.merge(left, right, on=['doc_name'], how='inner'),
                                    dfs_to_merge)

        if outcome_type == 'entity':
            self.threshold = None
            self.gold_standards = extract_gold_standard_outcome_from_claos(claos)
            self.predictions = extract_group_entity_from_claos(claos)
            self.tp, self.fp, self.tn, self.fn = compare_entity(self.gold_standards, self.predictions)

    def calculate_metrics(self, claos):
        """Calculate the metrics based on the counts of tp/fp/tn/fn or baseded on the predicted
        probabilities (with threshold) and the target label."""
        metrics = {}
        if self.outcome_type == 'entity':
            metrics = ConfusionMatrixMetrics(self.tp, self.fp, self.tn, self.fn, self.split_type)
            metrics.export_metrics_to_yaml(self.target_dir)
        
        if self.outcome_type == 'binary':
            if self.threshold is not None:
                metrics = ScoreCalculations(self.predictions, self.gold_standards, self.threshold, self.split_type)
            else:
                if self.threshold is None:
                    metrics = ScoreCalculations(self.predictions, self.gold_standards, None, self.split_type)
            results = metrics.get_all_metrics()
            tp, fp, fn, tn, p, r, f1, mc, pvr, tpfpratio, volume, fpr = itemgetter('tp', 'fp', 'fn', 'tn',
                                                                                   'precision', 'recall', 'f1',
                                                                                   'mcc', 'pvr', 'tpfp_ratio',
                                                                                   'volume', 'fpr')(results)
            metr = METRICS(tp, fp, fn, tn, p, r, f1, mc, pvr, tpfpratio, volume, fpr)
            metrics.export_metrics_to_yaml(self.target_dir)
            for clao in claos:
                clao.insert_annotation(METRICS, metr)

        if self.outcome_type == 'multi_labels':
            metrics = ScoreCalculationsMultiLabels(self.predictions, self.gold_standards, self.split_type)
            results = metrics.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results)
            metr = METRICS(label_Name=None, tp=None, fp=None, fn=None, tn=None, precision=p,
                           recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None,
                           volume=None, fpr=None, acc_score=acc_score)
            metrics.export_metrics_to_yaml(self.target_dir)
            for clao in claos:
                clao.insert_annotation(METRICS, metr)
        return metrics
