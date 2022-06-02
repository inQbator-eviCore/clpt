"""Evaluate the model and output metrics"""
import logging
import pandas as pd
from functools import reduce
from src.clpt.evaluation.score_calculations import ConfusionMatrixMetrics, ScoreCalculations, compare_entity
from src.constants.annotation_constants import PROBABILITY, ACTUAL_LABEL, DOCUMENT_NAME
from src.utils import extract_group_entity_from_claos, extract_gold_standard_outcome_from_claos, \
    extract_predicted_probability_from_claos


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

        if outcome_type == 'binary':
            actual_label_from_claos = extract_gold_standard_outcome_from_claos(claos)
            predicted_probs_from_claos = extract_predicted_probability_from_claos(claos)
            df_gold_standard = pd.DataFrame({DOCUMENT_NAME: list(actual_label_from_claos.keys()),
                                             ACTUAL_LABEL: list(actual_label_from_claos.values())})

            df_predicted_probs = pd.DataFrame({DOCUMENT_NAME: list(predicted_probs_from_claos.keys()),
                                               PROBABILITY: list(predicted_probs_from_claos.values())})

            dfs_to_merge = [df_gold_standard, df_predicted_probs]
            self.df_merged = reduce(lambda left, right: pd.merge(left, right, on=[DOCUMENT_NAME], how='inner'),
                                    dfs_to_merge)

            if (len(actual_label_from_claos) != self.df_merged.shape[0]) \
                    or (len(predicted_probs_from_claos) != self.df_merged.shape[0]):
                raise ValueError("Gold standard volume and predicted probabilities volume are not the same.")

            self.gold_standards = self.df_merged[ACTUAL_LABEL]
            self.predictions = self.df_merged[PROBABILITY]

        if outcome_type == 'entity':
            self.gold_standards = extract_gold_standard_outcome_from_claos(claos)
            self.predictions = extract_group_entity_from_claos(claos)
            self.tp, self.fp, self.tn, self.fn = compare_entity(self.gold_standards, self.predictions)

    def calculate_metrics(self):
        """Calculate the metrics based on the counts of tp/fp/tn/fn or baseded on the predicted
        probabilities (with threshold) and the target label."""
        if self.threshold is None:
            metrics = ConfusionMatrixMetrics(self.tp, self.fp, self.tn, self.fn, self.split_type)
            metrics.export_metrics_to_yaml(self.target_dir)
        else:
            metrics = ScoreCalculations(self.predictions, self.gold_standards, self.threshold, self.split_type)
            metrics.export_metrics_to_yaml(self.target_dir)
