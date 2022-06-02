"""Evaluation module provides functions to evaluate results of model performances.

If the outcome is probabilities, model is evaluated based on the target threshold and predicted probabilities.
Specified metrics, confusion matrix and accuracy are logged a YAML file.

If the outcome is entity from NER, model is evaluated based on if the predicted entity matches with the true.
"""
