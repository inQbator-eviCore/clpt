"""Initial thoughts / Rough draft of CLAO skeleton"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Generic, TypeVar, Union

from lxml import etree

from src.clao.annotations import Annotations

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class ClinicalLanguageAnnotationObject(ABC, Generic[T]):
    def __init__(self, data: Union[T, Annotations], config=None):
        """

        Args:
            data: Object of type T or pre-annotated dict
            config:
        """
        if config is None:
            # TODO: set default config
            config = None
        if isinstance(data, Annotations):
            self.annotations = data
        else:
            self.annotations = self.ingest_data(data, config)

    @classmethod
    @abstractmethod
    def from_file(cls, input_path: str, config=None):
        pass

    @classmethod
    def from_xml(cls, input_path: str, config=None):
        # TODO
        with open(input_path, 'r'):
            return None

    @abstractmethod
    def ingest_data(self, data: T, config) -> Annotations:
        pass

    @abstractmethod
    def insert_annotation(self) -> bool:
        pass

    @abstractmethod
    def remove_annotation(self) -> bool:
        pass

    def search_annotation(self, key: Union[int, str]) -> Dict:
        if isinstance(key, int):
            return self._search_by_id(key)
        if isinstance(key, str):
            return self._search_by_val(key)
        raise ValueError(f"Parameter 'key' must be of type int or str. Got {type(key).__name__}.")

    def _search_by_id(self, annot_id: int) -> Dict:
        pass

    @abstractmethod
    def _search_by_val(self, annot_val: T) -> Dict:
        pass

    def print_clao(self):
        print(self._dump_clao())

    def _dump_clao(self):
        return json.dumps(self.annotations)

    def write_as_xml(self, output_path) -> bool:
        try:
            xml = self.annotations.to_xml()
            etree.indent(xml, space='    ')
            etree.ElementTree(xml).write(output_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        except Exception as e:
            LOGGER.exception(f"Failed so save CLAO. "
                             f"Error of type {type(e).__name__} encountered with arguments '{e.args}'")
            return False
        return True

    def __hash__(self):
        # TODO
        return hash(self.annotations)


class TextCLAO(ClinicalLanguageAnnotationObject[str]):
    def __init__(self, data: Union[str, Annotations], config=None):
        """add docstring here"""
        super(TextCLAO, self).__init__(data, config)

    @classmethod
    def from_file(cls, input_path: str, config=None):
        with open(input_path, 'r') as f:
            return cls(f.read())

    def ingest_data(self, data: str, config) -> Annotations:
        # TODO
        return Annotations(None)

    def insert_annotation(self) -> bool:
        pass

    def remove_annotation(self) -> bool:
        pass

    def _search_by_val(self, annot_val: str) -> Dict:
        pass


# class VisualCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass
#
#
# class AudioCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass
