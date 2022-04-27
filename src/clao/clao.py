"""Initial thoughts / Rough draft of CLAO skeleton"""

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, TypeVar, Union

from lxml import etree
from omegaconf import DictConfig

from src.clao.annotations import Annotations

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class CLAODataType(Enum):
    TEXT = 'text'
    # AUDIO = 'audio'
    # VISUAL = 'visual'


class ClinicalLanguageAnnotationObject(ABC, Generic[T]):
    def __init__(self, raw_text: str, name: str, cfg: DictConfig = None):
        """

        Args:
            raw_text: pre-annotated dict
            name: CLAO name for serialization purposes. Usually the base name of the input file
            cfg:
        """

        self.cfg = cfg if cfg else None  # TODO: get default cfg
        self.annotations = Annotations(raw_text)
        self.name = name

    @classmethod
    @abstractmethod
    def from_file(cls, input_path: str, cfg: DictConfig = None):
        pass

    @classmethod
    def from_xml(cls, input_path: str, cfg: DictConfig = None):
        # TODO
        with open(input_path, 'r'):
            return None

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

    def write_as_json(self, output_path: str, filename: str = None) -> bool:
        try:
            json_dict = self.annotations.to_json()
            file_path = os.path.join(output_path, filename if filename else self.name) + '.json'
            with open(file_path, 'w') as json_out:
                json.dump(json_dict, json_out, indent=True)
        except Exception as e:
            LOGGER.exception(f"Failed to save CLAO as JSON. "
                             f"Error of type {type(e).__name__} encountered with arguments '{e.args}'")
            return False
        return True

    def write_as_xml(self, output_path: str, filename: str = None) -> bool:
        try:
            xml = self.annotations.to_xml()
            etree.indent(xml, space='    ')
            file_path = os.path.join(output_path, filename if filename else self.name) + '.xml'
            etree.ElementTree(xml).write(file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        except Exception as e:
            LOGGER.exception(f"Failed to save CLAO as XML. "
                             f"Error of type {type(e).__name__} encountered with arguments '{e.args}'")
            return False
        return True

    def __hash__(self):
        # TODO
        return hash(self.annotations)


class TextCLAO(ClinicalLanguageAnnotationObject[str]):
    def __init__(self, raw_text: str, name: str, cfg: DictConfig = None):
        """add docstring here"""
        super(TextCLAO, self).__init__(raw_text, name, cfg)

    @classmethod
    def from_file(cls, input_path: str, cfg: DictConfig = None):
        name = os.path.splitext(os.path.basename(input_path))[0]
        with open(input_path, 'r') as f:
            return cls(f.read(), name)

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
