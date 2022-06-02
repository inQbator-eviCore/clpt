
import os
from math import sqrt
import pytest
import numpy as np
from src.clpt.evaluation.score_calculations import ScoreCalculations, count_stats

def test_score_calculations_exceptions():
    with pytest.raises(ValueError, match="Evaluation volume should be greater than 0"):
        _ = ScoreCalculations([], [], 0.5)


def test_score_calculations():
    """Tests calculation of first level metrics"""
    labels = [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.]
    predictions = [.76, 0.49, .52, .12, .89, .64, .86, .99, .41, .57]
    threshold = 0.5
    tp = 6.
    fp = 1.
    tn = 1.
    fn = 2.
    volume = tp + fp + tn + fn
    ar = (tp + fp) / volume
    fpr = fp / (fp + tn)
    pvr = ar / fpr
    tpfp_ratio = tp / fp
    mcc = ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    metrics = ScoreCalculations(predictions, labels, threshold)
    scores = metrics.get_all_metrics()

    assert threshold == metrics.threshold == scores['threshold']
    assert volume == metrics.volume == scores['volume']
    assert fpr == metrics.fpr == scores['fpr']
    assert recall == metrics.recall == scores['recall']
    assert precision == metrics.precision == scores['precision']
    assert (tp + fp) / volume == metrics.automation_rate == scores['automation_rate']
    assert tn / volume == metrics.idr == scores['idr']
    assert tpfp_ratio == metrics.tpfp_ratio == scores['tpfp_ratio']
    assert pvr == metrics.pvr == scores['pvr']
    assert sqrt(tpfp_ratio * pvr) == metrics.yamm == scores['yamm']
    assert mcc == metrics.mcc == scores['mcc']
    assert (2 * precision * recall) / (precision + recall) == metrics.f1 == scores['f1']
    assert round(metrics.auc_pr, 4) == round(0.9262400793650793, 4) == round(scores['auc_pr'], 4)
    assert round(metrics.average_precision, 4) == round(0.9317956349206349, 4) == round(scores['average_precision'], 4)
    np.testing.assert_array_equal(metrics.confusion_matrix, np.array([[tn, fp], [fn, tp]]))


def test_persist_metrics_to_yaml(tmpdir):
    """Test whether the metrics persisted properly to yaml file"""
    target_dir = tmpdir.strpath

    labels = [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.]
    predictions = [.76, 0.49, .52, .12, .89, .64, .86, .99, .41, .57]
    threshold = 0.5
    metrics = ScoreCalculations(predictions, labels, threshold)
    metrics.export_metrics_to_yaml(target_dir)
    assert os.path.exists(os.path.join(target_dir, f"results.yaml"))


def test_persist_prediction_probas_to_csv(tmpdir):
    target_dir = tmpdir.strpath

    labels = [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.]
    ids = list(range(len(labels)))
    predictions = [.76, 0.49, .52, .12, .89, .64, .86, .99, .41, .57]
    threshold = 0.5
    metrics = ScoreCalculations(predictions, labels, threshold)
    metrics.export_predicted_probas_to_csv(target_dir, [('id', ids)])

    assert os.path.exists(os.path.join(target_dir, f"predictions.csv"))


def test_count_stats():
    labels = [1., 1., 1., 0., 1., 0., 1., 1., 1., 1.]
    predictions = [1, 0, 1, 0, 1, 1, 1, 1, 0, 1]
    tp = 6.
    fp = 1.
    tn = 1.
    fn = 2.
    TP2, FP2, TN2, FN2 = count_stats(y_actual=labels, y_pred=predictions)
    assert tp == TP2
    assert fp == FP2
    assert tn == TN2
    assert fn == FN2
