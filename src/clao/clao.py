"""Create a general Clinical Language Annotation Object (CLAO)."""
import json
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

from blist import blist
from lxml import etree

from src.constants.annotation_constants import ELEMENT, ID

LOGGER = logging.getLogger(__name__)

T = TypeVar("T")


class CLAODataType(Enum):
    """CLAO supporting data type."""
    TEXT = 'text'
    # AUDIO = 'audio'
    # VISUAL = 'visual'


class CLAOElement:
    """Basic element of a CLAO object.

    Class Attributes:
        element_name: Name of element as it should appear in an XML tag.
    """
    element_name = ELEMENT

    def to_json(self) -> Dict:
        """Return dict of element."""
        return {}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element to XML tree.

        Args:
            parent: etree.Element, parent XML element. Optional
            attribs (dict[str, str]): attributes of CLAO element in the format of dictionary

        Returns:
            etree.Element with element name and attributes added; or etree.SubElement with parent, element name and
            attributes added
        """
        if attribs is None:
            attribs = self.to_json()
        if parent is None:
            return etree.Element(self.element_name, **attribs)
        else:
            return etree.SubElement(parent, self.element_name, **attribs)

    @classmethod
    def from_json(cls, json_dict: dict):
        """Load CLAO element from a data file in the json format."""
        pass

    @classmethod
    def from_xml(cls, xml_element: etree._Element):
        """Load CLAO element from a XML file."""
        pass


class IdCLAOElement(CLAOElement):
    """CLAO element with an ID attribute.

    Attributes:
        element_id: The id of this Span (starts at 0).
    """

    def __init__(self, element_id: int, **kwargs):
        super(IdCLAOElement, self).__init__(**kwargs)
        self.element_id = element_id

    def to_json(self) -> Dict:
        """Add CLAO element with an ID attribute."""
        return {ID: str(self.element_id), **super().to_json()}


class ClinicalLanguageAnnotationObject(ABC, Generic[T]):
    """Clinical Language Annotation Object (the CLAO).

    The CLAO supports addition, deletion, and update operations via either element name of indexing. The CLAO(s) are
    human-readable and could be easily shared across projects as they could be exported to JSON format or XML format.
    """
    def __init__(self, raw_data: CLAOElement, name: str, *args, **kwargs):
        """
        Args:
            raw_data (CLAOElement): CLAOElement representing the raw data of the document represented by the CLAO
            name (str): CLAO name for serialization purposes. Usually the base name of the input file
        """
        self.elements = {raw_data.element_name: blist([raw_data])}
        self.name = name

    @classmethod
    @abstractmethod
    def from_file(cls, input_path: str):
        """Create a new CLAO from a data file"""
        pass

    @classmethod
    def from_xml_file(cls, input_path: str):
        """Load a CLAO from an XML file."""
        # TODO
        with open(input_path, 'r'):
            return None

    def insert_annotation(self, element_class: Union[Type[CLAOElement], str], value):
        """Insert a single annotation of a specific type into the CLAO. Annotation will be appended to its collection.

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be inserted
            value: A element of class with value(s) to be appended

        Returns:
            None
        """
        self.insert_annotations(element_class, [value])

    def insert_annotations(self, element_class: Union[Type[CLAOElement], str], values):
        """Insert annotations of a specific type into the CLAO. Annotations will be appended to their collections.

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be inserted
            values: List of elements of class object with values to be appended

        Returns:
            None
        """
        element_annotations = self.get_annotations(element_class)
        element_annotations.extend(values)
        self.elements[self._get_element_name(element_class)] = element_annotations

    def remove_annotations(self, element_class: Union[Type[CLAOElement], str]):
        """Remove an entire collection of annotations from the CLAO.

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be removed

        Returns:
            None
        """
        element_name = self._get_element_name(element_class)
        if element_name in self.elements:
            del(self.elements[element_name])

    def get_all_annotations_for_element(self, element_class: Union[Type[CLAOElement], str]
                                        ) -> Union[CLAOElement, List[CLAOElement]]:
        """Get the entire collection of annotations for the given type.

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be fetched

        Returns:
            List of CLAOElements of type element_class, or single CLAOElement of type element_class
        """
        return self.elements.get(self._get_element_name(element_class), blist())

    def get_annotations(self,
                        element_class: Union[Type[CLAOElement], str],
                        key: Optional[Union[Tuple[int, int], int, dict]] = None
                        ) -> Union[CLAOElement, List[CLAOElement]]:
        """Get specific annotation(s) for an element type based on key. If key is int or tuple of int, will fetch by
        element ID. A key of dict will search annotations by value.

        To return all annotations for a type, key should be None.
        To return a single annotation for a type, key should be in int representing the element id.
        To return a range of annotations, key should be a tuple of ints (x,y) representing a range of element ids where
            x is inclusive and y exclusive.
        To return a single annotation matching specific attributes, key should be a dict. Will return the first matching
            annotation. E.g. to return the first matching Entity with Entity.type == MENTION and Entity.label ==
            MEDICATION use clao.get_annotations(Entity, {'type': 'MENTION', 'label': 'MEDICATION'})

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be fetched
            key (Union[Tuple[int, int], int, dict]): key used to find proper annotation (optional)

        Returns:
            List of type element_class, or single CLAOElement of type element_class
        """
        if key is None:
            return self.get_all_annotations_for_element(element_class)
        if isinstance(key, (tuple, int)):
            return self._search_by_id(element_class, key)
        if isinstance(key, dict):
            return self._search_by_val(element_class, key)
        raise ValueError(f"Parameter 'key' must be of type int or str. Got {type(key).__name__}.")

    def _search_by_id(self, element_class: Union[Type[CLAOElement], str], key: Union[Tuple[int, int], int]):
        """Get specific annotation(s) for an element type based on key.

        To return a single annotation for a type, key should be in int representing the element id.
        To return a range of annotations, key should be a tuple of ints (x,y) representing a range of element ids where
            x is inclusive and y exclusive

        Args:
            element_class (Union[Type[CLAOElement], str]): Type of element to return
            key (Union[Tuple[int, int], int])): element id or range of ids to return

        Returns:
            List of element_type, or single CLAOElement of element_type

        """
        element_annotations = self.get_all_annotations_for_element(element_class)
        if isinstance(key, int):
            return element_annotations[key]
        else:
            return element_annotations[slice(*key)]

    def _search_by_val(self, element_class: Union[Type[CLAOElement], str], key: dict):
        """Get specific annotation(s) for an element type based on key.

        Will return the first matching annotation. E.g. to return the first matching Entity with Entity.type == MENTION
            and Entity.label == MEDICATION use self._search_by_val(Entity, {'type': 'MENTION', 'label': 'MEDICATION'})

        Args:
            element_class (Union[Type[CLAOElement], str]): Subclass of CLAOElement (or its element_name) to be fetched
            key (dict): key used to find proper annotation

        Returns:
            List of type element_class, or single CLAOElement of type element_class
        """
        element_annotations = self.get_all_annotations_for_element(element_class)
        for elem in element_annotations:
            if key.items() <= elem.__dict__.items():
                return elem
        return None

    @staticmethod
    def _get_element_name(element_class: Union[Type[CLAOElement], str]) -> str:
        """Get the proper element name of a given class.

        Args:
            element_class: Subclass of CLAOElement (or its element_name)

        Returns:
            element_name for element_class
        """
        return element_class.element_name if issubclass(element_class, CLAOElement) else element_class

    @abstractmethod
    def to_json(self):
        """Add element(s)."""
        pass

    @abstractmethod
    def to_xml(self):
        """Add element(s)."""
        pass

    def write_as_json(self, output_directory: str, filename: str = None) -> bool:
        """Save CLAO to a JSON file.

        Args:
            output_directory (str): directory to save file to
            filename (str): name of file (sans extension)

        Returns:
            True if save was successful

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
        """Save CLAO to an XML file.

        Args:
            output_directory (str): directory to save file to
            filename (str): name of file (sans extension)

        Returns:
            True if save was successful

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

    Add subclasses of this abstract class to any CLAOElement class to create a container for a sub CLAOElement.

    Subclasses should contain an element id and a property method to pull that id from CLAO."""
    def __init__(self, clao: ClinicalLanguageAnnotationObject, **kwargs):
        self.clao = clao
