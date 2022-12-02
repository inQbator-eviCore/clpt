"""Utilities to be used throughout the CLAO and the pipeline.

A set of utility functions.
"""
import re
from typing import List
from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
from datetime import datetime
from src.clao.text_clao import Embedding, Span, Entity, ActualLabel, PredictProbabilities, Predictions, EmbeddingVector


def add_new_key_to_cfg(cfg: DictConfig, value: str, *keys: str) -> None:
    """Add value to config section following key path, where key(s) do not already exist in config

    Args:
        cfg: config to add value to
        value: value to add to config
        *keys: config section names pointing to the desired key. Final item in list will be given value of value

    Returns: None

    """
    cfg_section = cfg
    for key in keys[:-1]:
        cfg_section = cfg_section[key]
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg_section[keys[-1]] = value


def match(regex: re.Pattern, string: str, offset: int, keep_between: bool) -> List[Span]:
    """
    Splits given input string into a List of Spans based on provided regex
    Args:
        regex: Compiled regular expression object
        string: The string to split
        offset: character index of the document that the input text starts at
        keep_between: true if text between matches should be included in list, else false

    Returns:
        A list of Spans split from the input text by the regex
    """
    span_array = []
    end_index = len(string)
    cursor = 0

    while cursor < end_index:
        for m in regex.finditer(string):
            start = m.start()
            if keep_between and start > cursor:
                span_array.append(Span(offset + cursor, offset + start))

            end = m.end()
            span_array.append(Span(offset + start, offset + end))
            cursor = end

        text = string[cursor:]
        if len(text.strip()) > 0:
            span_array.append(Span(offset + cursor, offset + end_index))

        cursor = end_index

    return span_array


def stringify(obj):
    """Helps serialize numpy objects.

    Args:
        obj: object that will be serialized

    Returns:
        obj: original value of the obj for actual data type of the object
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, datetime):
        return obj.__str__()
    else:
        return obj


def extract_group_entity_to_list(clao_info):
    """Extract the predicted entity_group from a CLAO.
        Args:
            clao: a ClinicalLanguageAnnotationObject

        Returns:
            group_entity_list: a list that contains the predicted entity
        """
    entities = clao_info.get_annotations(Entity)
    entity_groups = []
    for entity in entities:
        if entity.literal not in entity_groups:
            entity_groups.append(entity.literal)
    return entity_groups


def extract_group_entity_from_claos(claos):
    """Extract the predicted entity_group from all CLAOs.
    Args:
        clao: a list of ClinicalLanguageAnnotationObject, e.g. dc.claos

    Returns:
        entity_groups_dict: a dictionary that contains the document name as the key and the list of predicted
        entity_group as the value
    """
    group_entity_from_claos_dict = {}

    for clao in claos:
        group_entity_from_claos_dict[clao.name] = extract_group_entity_to_list(clao)
    return group_entity_from_claos_dict


def extract_gold_standard_outcome_from_claos(claos):
    """Extract the gold standard entities or target label which have been inserted into CLAO."""
    gold_standard_dic = {}
    for clao in claos:
        for t in clao.get_annotations(ActualLabel):
            gold_standard_dic[clao.name] = t.actual_label_value
    return gold_standard_dic


def extract_vector_from_claos(claos):
    """Extract the gold standard entities or target label which have been inserted into CLAO."""
    vector_dic = {}
    for clao in claos:
        for t in clao.get_annotations(EmbeddingVector):
            vector_dic[clao.name] = t.vector
    return vector_dic


def extract_embedding_from_claos(claos):
    """Extract the gold standard entities or target label which have been inserted into CLAO."""
    vector_dic = {}
    for clao in claos:
        for t in clao.get_annotations(Embedding):
            vector_dic[clao.name] = t.vector
    return vector_dic


def extract_predicted_probability_from_claos(claos):
    """Extract the predicted probability from CLAO."""
    predicted_probability_dic = {}
    for clao in claos:
        for t in clao.get_annotations(PredictProbabilities):
            predicted_probability_dic[clao.name] = t.probability
    return predicted_probability_dic


def extract_prediction_from_claos(claos):
    """Extract the prediction from CLAO."""
    predictions_dic = {}
    for clao in claos:
        for t in clao.get_annotations(Predictions):
            predictions_dic[clao.name] = t.prediction
    return predictions_dic
