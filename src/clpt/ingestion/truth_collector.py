"""Ingest the gold-standard annotations or the target outcome(s) into CLAOs."""
import logging
import os
import pandas as pd
import json

from src.clao.text_clao import ActualMultiLabels, ActualLabel, ActualDataSource, PredictProbabilities, Predictions
from src.clpt.ingestion.document_collector import DocumentCollector
from src.constants.annotation_constants import ACTUAL_LABEL, PREDICTION, PROBABILITY, DATA_SOURCE

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

    def load_outcome_from_csv(self):
        """Load a gold-standard outcome from a csv file into a Pandas DataFrame and add each label to CLAOs.

        """
        outcomes = pd.read_csv(self.outcome_file)
        for clao in self.dc.claos:
            if DATA_SOURCE in outcomes.columns:
                try:
                    data_source = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, DATA_SOURCE].item()
                except ValueError as e:
                    # TODO Better handling for repeat lines
                    logger.warning(f"Repeat line for file {clao.name} in outcomes csv. Inserting 0")
                    logger.info(e)
                    data_source = ''
                clao.insert_annotation(ActualDataSource, ActualDataSource(data_source))
            if ACTUAL_LABEL in outcomes.columns:
                try:
                    actual_label = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, ACTUAL_LABEL].item()
                    # logger.info(actual_label)
                except ValueError as e:
                    # TODO Better handling for repeat lines
                    logger.warning(f"Repeat line for file {clao.name} in outcomes csv. Inserting 0")
                    logger.info(e)
                    actual_label = 0
                clao.insert_annotation(ActualLabel, ActualLabel(actual_label))
            if PREDICTION in outcomes.columns:
                predicted_label = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, PREDICTION].item()
                clao.insert_annotation(Predictions, Predictions(predicted_label))
            if PROBABILITY in outcomes.columns:
                probability = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name, PROBABILITY].item()
                clao.insert_annotation(PredictProbabilities, PredictProbabilities(probability))

    def load_outcome_multi_from_csv(self):
        """Load a gold-standard outcome from a csv file into a Pandas DataFrame and add each label to CLAOs.
        """
        outcomes = pd.read_csv(self.outcome_file)
        for clao in self.dc.claos:
            try:
                data_set = outcomes.loc[outcomes['doc_name'].astype(str) == clao.name]
                d = data_set.to_dict(orient='records')
                d[0].pop('doc_name')
            except ValueError as e:
                # TODO Better handling for repeat lines
                logger.warning(f"Repeat line for file {clao.name} in outcomes csv. Inserting 0")
                logger.info(e)
                data_set = ''
            clao.insert_annotation(ActualMultiLabels, ActualMultiLabels(d[0]))

    def load_entity_from_json(self):
        """Load a gold-standard entity from the annotation file into a dictionary format and add to CLAOs.
        """
        with open(self.outcome_file, 'r') as f:
            outcomes = json.load(f)
        for clao in self.dc.claos:
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
            self.load_outcome_from_csv()
        if self.outcome_type == 'multi_labels':
            self.load_outcome_from_csv()
        if self.outcome_type == 'multi_cols':
            self.load_outcome_multi_from_csv()
        if self.outcome_type == 'entity':
            self.load_entity_from_json()
        if self.outcome_type == 'span':
            raise NotImplementedError(f"Loading span (start index and end index) is not implemented."
                                      f"for outcome '{self.outcome_type}'")
        if self.outcome_type == 'span_linking':
            raise NotImplementedError(f"Loading span (start index and end index) and linking is not implemented."
                                      f"for outcome '{self.outcome_type}'")
