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
            # raise ValueError("The outcome type is binary and the threshold cannot be null.")
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
        if self.outcome_type == 'multi_cols':
            metrics_treatment = ScoreCalculationsMultiLabels(self.df_merged['treatment_data'].tolist(),
                                                             self.df_merged['Treatment_Text_pred'].tolist(),
                                                             self.split_type)
            results_treatment = metrics_treatment.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_treatment)
            metr_treat = METRICS(label_Name='Treatment_Text', tp=None, fp=None, fn=None, tn=None, precision=p,
                                 recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None, volume=None,
                                 fpr=None, acc_score=acc_score)
            # Experimental Text
            metrics_exp = ScoreCalculationsMultiLabels(self.df_merged['Experimental'].tolist(),
                                                       self.df_merged['experimental_pred'].tolist(),
                                                       self.split_type)
            results_exp = metrics_exp.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_exp)
            metr_exp = METRICS(label_Name='Experimental', tp=None, fp=None, fn=None, tn=None, precision=p,
                               recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None, volume=None, fpr=None,
                               acc_score=acc_score)
            # Insufficient Info
            metrics_insufficient = ScoreCalculationsMultiLabels(self.df_merged['insufficient_info'].tolist(),
                                                                self.df_merged['insufficient_info_pred'].tolist(),
                                                                self.split_type)
            results_insufficient = metrics_insufficient.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_insufficient)
            metr_insufficient = METRICS(label_Name='Insufficient_Info', tp=None, fp=None, fn=None, tn=None,
                                        precision=p, recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None,
                                        volume=None, fpr=None, acc_score=acc_score)
            # No cytoscopy
            metrics_cystoscopy = ScoreCalculationsMultiLabels(self.df_merged['no_cystoscopy'].tolist(),
                                                              self.df_merged['no_cystoscopy_pred'].tolist(),
                                                              self.split_type)
            results_cystoscopy = metrics_cystoscopy.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_cystoscopy)
            metr_cystoscopy = METRICS(label_Name='no_cystoscopy', tp=None, fp=None, fn=None, tn=None,
                                      precision=p, recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None,
                                      volume=None, fpr=None, acc_score=acc_score)
            # No volume
            metrics_volume = ScoreCalculationsMultiLabels(self.df_merged['no_volume'].tolist(),
                                                          self.df_merged['no_volume_pred'].tolist(),
                                                          self.split_type)
            results_volume = metrics_volume.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_volume)
            metr_volume = METRICS(label_Name='no_volume', tp=None, fp=None, fn=None, tn=None,
                                  precision=p, recall=r, f1=f1, mcc=None, pvr=None,
                                  tpfpratio=None, volume=None, fpr=None, acc_score=acc_score)
            # volume gt 80
            metrics_volume_gt_80 = ScoreCalculationsMultiLabels(self.df_merged['volume_gt_80'].tolist(),
                                                                self.df_merged['volume_gt_80_pred'].tolist(),
                                                                self.split_type)
            results_volume_gt_80 = metrics_volume_gt_80.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_volume_gt_80)
            metr_volume_gt_80 = METRICS(label_Name='volume_gt_80', tp=None, fp=None, fn=None, tn=None,
                                        precision=p, recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None,
                                        volume=None, fpr=None, acc_score=acc_score)
            # no drugs
            metrics_no_drugs = ScoreCalculationsMultiLabels(self.df_merged['no_pills'].tolist(),
                                                            self.df_merged['no_pills_pred'].tolist(),
                                                            self.split_type)
            results_no_drugs = metrics_no_drugs.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_no_drugs)
            metr_no_drugs = METRICS(label_Name='no_pills', tp=None, fp=None, fn=None, tn=None, precision=p,
                                    recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None, volume=None,
                                    fpr=None, acc_score=acc_score)
            # age lt 50
            metrics_age_lt_50 = ScoreCalculationsMultiLabels(self.df_merged['age_lt_50'].tolist(),
                                                             self.df_merged['age_lt_50_pred'].tolist(),
                                                             self.split_type)
            results_age_lt_50 = metrics_age_lt_50.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_age_lt_50)
            metr_age_lt_50 = METRICS(label_Name='age_lt_50', tp=None, fp=None, fn=None, tn=None, precision=p,
                                     recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None, volume=None,
                                     fpr=None, acc_score=acc_score)
            # age lt 45
            metrics_age_lt_45 = ScoreCalculationsMultiLabels(self.df_merged['age_lt_45'].tolist(),
                                                             self.df_merged['age_lt_45_pred'].tolist(),
                                                             self.split_type)
            results_age_lt_45 = metrics_age_lt_45.get_all_metrics()
            p, r, f1, acc_score = itemgetter('precision', 'recall', 'f1', 'acc')(results_age_lt_45)
            metr_age_lt_45 = METRICS(label_Name='age_lt_45', tp=None, fp=None, fn=None, tn=None, precision=p,
                                     recall=r, f1=f1, mcc=None, pvr=None, tpfpratio=None,
                                     volume=None, fpr=None, acc_score=acc_score)

            for clao in claos:
                clao.insert_annotation(METRICS, metr_treat)
                clao.insert_annotation(METRICS, metr_exp)
                clao.insert_annotation(METRICS, metr_insufficient)
                clao.insert_annotation(METRICS, metr_cystoscopy)
                clao.insert_annotation(METRICS, metr_volume)
                clao.insert_annotation(METRICS, metr_volume_gt_80)
                clao.insert_annotation(METRICS, metr_no_drugs)
                clao.insert_annotation(METRICS, metr_age_lt_50)
                clao.insert_annotation(METRICS, metr_age_lt_45)
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
