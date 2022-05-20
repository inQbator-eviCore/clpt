"""ClinicalLanguageAnnotationObject"""

import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, List, Optional, Tuple, TypeVar, Union

from blist import blist
from lxml import etree

from src.constants.annotation_constants import ELEMENT, ID

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class CLAODataType(Enum):
    TEXT = 'text'
    # AUDIO = 'audio'
    # VISUAL = 'visual'


class CLAOElement:
    """Basic element of a CLAO object

    Class Attributes:
        element_name: Name of element as it should appear in an XML tag
    """
    element_name = ELEMENT

    def to_json(self) -> Dict:
        return {}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        if attribs is None:
            attribs = self.to_json()
        if parent is None:
            return etree.Element(self.element_name, **attribs)
        else:
            return etree.SubElement(parent, self.element_name, **attribs)


class IdCLAOElement(CLAOElement):
    """CLAO element with an ID attribute

    Attributes:
        element_id: The id of this Span (starts at 0).
    """

    def __init__(self, element_id: int, **kwargs):
        super(IdCLAOElement, self).__init__(**kwargs)
        self.element_id = element_id

    def to_json(self) -> Dict:
        return {ID: str(self.element_id), **super().to_json()}


class ClinicalLanguageAnnotationObject(ABC, Generic[T]):
    """
    TODO: Add basic CLAO description
    """
    def __init__(self, raw_data: CLAOElement, name: str, *args, **kwargs):
        """
        Args:
            raw_data: CLAOElement representing the raw data of the document represented by the CLAO
            name: CLAO name for serialization purposes. Usually the base name of the input file
        """
        self.elements = {raw_data.element_name: raw_data}
        self.name = name

    @classmethod
    @abstractmethod
    def from_file(cls, input_path: str):
        """Create a new CLAO from a data file"""
        pass

    @classmethod
    def from_xml(cls, input_path: str):
        """Load a CLAO from an XML file"""
        # TODO
        with open(input_path, 'r'):
            return None

    def insert_annotation(self, element_type: str, value, element_type_is_list: bool = True):
        """Insert a single annotation of a specific type into the CLAO. Annotation will be appended to its collection

        Args:
            element_type: Name of element type
            value: Value to be appended
            element_type_is_list: True if the CLAO is meant to contain a collection of {element_type}s rather than a
                                  singleton {element_type}

        Returns: None
        """
        if element_type_is_list:
            self.insert_annotations(element_type, [value])
        else:
            self.elements[element_type] = value

    def insert_annotations(self, element_type: str, values):
        """Insert annotations of a specific type into the CLAO. Annotations will be appended to their collections

        Args:
            element_type: Name of element type
            values: List of values to be appended

        Returns: None
        """
        element_annotations = self.get_annotations(element_type)
        element_annotations.extend(values)
        self.elements[element_type] = element_annotations

    def remove_annotations(self, element_type: str):
        """Remove an entire collection of annotations from the CLAO

        Args:
            element_type: Type to be removed

        Returns: None
        """
        if element_type in self.elements:
            del(self.elements[element_type])

    def get_all_annotations_for_element(self, element_type) -> Union[CLAOElement, List[CLAOElement]]:
        """Get the entire collection of annotations for the given type

        Args:
            element_type: Type to be returned

        Returns: List of List of CLAOElements of element_type, or single CLAOElement of element_type

        """
        return self.elements.get(element_type, blist())

    def get_annotations(self, element_type: str, key: Optional[Union[Tuple[int, int], int, str]] = None
                        ) -> Union[CLAOElement, List[CLAOElement]]:
        """Get specific annotation(s) for an element type based on key. If key is int or tuple of int, will fetch by
        element ID. A key of string will search annotations by value (currently not implemented. TODO).

        To return all annotations for a type key should be None
        To return a single annotation for a type key should be in int representing the element id
        To return a range of annotations key should be a tuple of ints (x,y) representing a range of element ids where x
            is inclusive and y exclusive

        Args:
            element_type: Type of element to return
            key: element id or range of ids to return

        Returns: List of element_type, or single CLAOElement of element_type

        """
        if key is None:
            return self.get_all_annotations_for_element(element_type)
        if isinstance(key, (tuple, int)):
            return self._search_by_id(element_type, key)
        if isinstance(key, str):
            return self._search_by_val(element_type, key)
        raise ValueError(f"Parameter 'key' must be of type int or str. Got {type(key).__name__}.")

    def _search_by_id(self, element_type: str, key: Union[Tuple[int, int], int]):
        """Get specific annotation(s) for an element type based on key.
        To return a single annotation for a type key should be in int representing the element id
        To return a range of annotations key should be a tuple of ints (x,y) representing a range of element ids where x
            is inclusive and y exclusive

        Args:
            element_type: Type of element to return
            key: element id or range of ids to return

        Returns: List of element_type, or single CLAOElement of element_type

        """
        element_annotations = self.get_all_annotations_for_element(element_type)
        if isinstance(key, int):
            return element_annotations[key]
        else:
            return element_annotations[slice(*key)]

    def _search_by_val(self, element_type: str, value: T):
        raise NotImplementedError()

    @abstractmethod
    def to_json(self):
        pass

    @abstractmethod
    def to_xml(self):
        pass

    def write_as_json(self, output_directory: str, filename: str = None) -> bool:
        """Save CLAO to a JSON file

        Args:
            output_directory: directory to save file to
            filename: name of file (sans extension)

        Returns: True if save was successful

        """
        try:
            json_dict = self.to_json()
            file_path = os.path.join(output_directory, filename if filename else self.name) + '.json'
            with open(file_path, 'w') as json_out:
                json.dump(json_dict, json_out, indent=True)
        except Exception as e:
            LOGGER.exception(f"Failed to save CLAO as JSON. "
                             f"Error of type {type(e).__name__} encountered with arguments '{e.args}'")
            return False
        return True

    def write_as_xml(self, output_directory: str, filename: str = None) -> bool:
        """Save CLAO to an XML file

        Args:
            output_directory: directory to save file to
            filename: name of file (sans extension)

        Returns: True if save was successful

        """
        try:
            xml = self.to_xml()
            etree.indent(xml, space='    ')
            file_path = os.path.join(output_directory, filename if filename else self.name) + '.xml'
            etree.ElementTree(xml).write(file_path, pretty_print=True, xml_declaration=True, encoding='UTF-8')
        except Exception as e:
            LOGGER.exception(f"Failed to save CLAO as XML. "
                             f"Error of type {type(e).__name__} encountered with arguments '{e.args}'")
            raise e
        return True

    def __hash__(self):
        # TODO
        return hash(self.elements)

# class VisualCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass
#
#
# class AudioCLAO(ClinicalLanguageAnnotationObject[?]):
#     pass


class CLAOElementContainer(ABC):
    """Base class for defining a sub-element container for a CLAOElement class.

    Add subclasses of this abstract class to any CLAOElement class to create a container for a sub CLAOElement

    Subclasses should contain an element id and a property method to pull that id from CLAO"""
    def __init__(self, clao: ClinicalLanguageAnnotationObject, **kwargs):
        self.clao = clao
