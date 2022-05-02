"""Initial thoughts / Rough draft of CLAO skeleton"""

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Generic, Optional, Tuple, TypeVar, Union

from blist import blist
from lxml import etree
from omegaconf import DictConfig

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class CLAODataType(Enum):
    TEXT = 'text'
    # AUDIO = 'audio'
    # VISUAL = 'visual'


class ClinicalLanguageAnnotationObject(ABC, Generic[T]):
    def __init__(self, raw_data, name: str, cfg: DictConfig = None):
        """

        Args:
            raw_data: pre-annotated dict
            name: CLAO name for serialization purposes. Usually the base name of the input file
            cfg:
        """

        self.cfg = cfg if cfg else None  # TODO: get default cfg
        self.annotations = self._init_annotations_(raw_data)
        self.name = name

    @abstractmethod
    def _init_annotations_(self, raw_data):
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, input_path: str, cfg: DictConfig = None):
        pass

    @classmethod
    def from_xml(cls, input_path: str, cfg: DictConfig = None):
        # TODO
        with open(input_path, 'r'):
            return None

    def insert_annotation(self, element_type: str, value, element_type_is_list: bool = True):
        if element_type_is_list:
            self.insert_annotations(element_type, [value])
        else:
            self.annotations.elements[element_type] = value

    def insert_annotations(self, element_type: str, values):
        element_annotations = self.get_all_annotations_for_element(element_type)
        element_annotations.extend(values)
        self.annotations.elements[element_type] = element_annotations

    def remove_annotations(self, element_type: str):
        if element_type in self.annotations.elements:
            del(self.annotations.elements[element_type])

    def get_all_annotations_for_element(self, element_type):
        return self.annotations.elements.get(element_type, blist())

    def search_annotations(self, element_type: str, key: Optional[Union[Tuple[int, int], int, str]] = None):
        if key is None:
            return self.get_all_annotations_for_element(element_type)
        if isinstance(key, (tuple, int)):
            return self._search_by_id(element_type, key)
        if isinstance(key, str):
            return self._search_by_val(element_type, key)
        raise ValueError(f"Parameter 'key' must be of type int or str. Got {type(key).__name__}.")

    def _search_by_id(self, element_type: str, key: Union[Tuple[int, int], int]):
        element_annotations = self.get_all_annotations_for_element(element_type)
        if isinstance(key, int):
            return element_annotations[key]
        else:
            return element_annotations[slice(*key)]

    @abstractmethod
    def _search_by_val(self, element_type: str, value: T):
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

# class VisualCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass
#
#
# class AudioCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass
