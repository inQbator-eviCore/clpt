"""Ingest the gold-standard annotations or the target outcome(s) into CLAOs."""
import logging
import os
import pandas as pd
import json

from src.clao.text_clao import ActualLabel, PredictProbabilities, Predictions
from src.clpt.ingestion.document_collector import DocumentCollector
from src.constants.annotation_constants import ACTUAL_LABEL, PREDICTION, PROBABILITY

logger = logging.getLogger(__name__)


class TruthCollector:
    """Ingest the gold standard outcome/annotations into each of CLAOs.

    Args:
        dc (DocumentCollector): DocumentCollector with a list of CLAOs
        outcome_file (str): the file which contains the target outcomes, such as gold standard annotations or actual
            labels
        outcome_type (str): the type of outcome, such as entity or binary
    """
    def __init__(self, dc: DocumentCollector, outcome_file: str, outcome_type: str):
        self.dc = dc
        self.outcome_file = outcome_file
        self.outcome_type = outcome_type

    @staticmethod
    def load_outcome_from_csv(dc: DocumentCollector, outcome_file: str):
        """Load a gold-standard outcome from a csv file into a Pandas DataFrame and add each label to CLAOs.

        Args:
            dc (DocumentCollector): DocumentCollector with a list of CLAOs
            outcome_file (str): the name of the file that contains the target outcome(s)
        """
        outcomes = pd.read_csv(outcome_file)
        for clao in dc.claos:
            if ACTUAL_LABEL in outcomes.columns:
                actual_label = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, ACTUAL_LABEL].item()
                clao.insert_annotation(ActualLabel, ActualLabel(actual_label))
            if PREDICTION in outcomes.columns:
                predicted_label = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, PREDICTION].item()
                clao.insert_annotation(Predictions, Predictions(predicted_label))
            if PROBABILITY in outcomes.columns:
                probability = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, PROBABILITY].item()
                clao.insert_annotation(PredictProbabilities, PredictProbabilities(probability))

    @staticmethod
    def load_entity_from_json(dc: DocumentCollector, outcome_file: str):
        """Load a gold-standard entity from the annotation file into a dictionary format and add to CLAOs.

        Args:
            dc (DocumentCollector): DocumentCollector with a list of CLAOs
            outcome_file (str): the name of the file that contains the gold-standard annotations in json format
        """
        with open(outcome_file, 'r') as f:
            outcomes = json.load(f)
        for clao in dc.claos:
            actual_label = outcomes[clao.name]
            clao.insert_annotation(ActualLabel, ActualLabel(actual_label))

    @staticmethod
    def load_span_from_json():
        """Load a gold-standard span (start index, end index) from the annotation file into a dictionary format
        and add to CLAOs."""
        pass

    @staticmethod
    def load_span_linking_from_json():
        """Load a gold-standard span and entity (start index, end index, entity) from the annotation file into
        a dictionary format and add to CLAOs."""
        pass

    def ingest(self):
        if os.path.isdir(os.path.dirname(self.outcome_file)):
            logger.info(f"Ingesting outcomes from {self.outcome_file}")
        else:
            logger.warning("A gold standard file is not provided.")
        if self.outcome_type == 'binary':
            self.load_outcome_from_csv(dc=self.dc, outcome_file=self.outcome_file)
        if self.outcome_type == 'entity':
            self.load_entity_from_json(dc=self.dc, outcome_file=self.outcome_file)
        if self.outcome_type == 'span':
            raise NotImplementedError(f"Loading span (start index and end index) is not implemented."
                                      f"for outcome '{self.outcome_type}'")
        if self.outcome_type == 'span_linking':
            raise NotImplementedError(f"Loading span (start index and end index) and linking is not implemented."
                                      f"for outcome '{self.outcome_type}'")
