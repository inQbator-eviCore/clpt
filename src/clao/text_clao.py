"""Create a TextCALO.

Initial version of the text annotation objects, in line with XML schema illustrated in 2022 NaaCL paper."""
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
from lxml import etree
from omegaconf import DictConfig

from src.clao.clao import CLAOElement, CLAOElementContainer, ClinicalLanguageAnnotationObject, IdCLAOElement
from src.constants.annotation_constants import ANNOTATION, CLEANED_TEXT, DESCRIPTION, EMBEDDING, EMBEDDING_ID, \
    ENTITIES, ENTITY, ENTITY_GROUP, EntityType, HEADING, LITERAL, PARAGRAPH, PARAGRAPHS, RAW_TEXT, SECTION, SENTENCE, \
    SENTENCES, SPAN, TEXT, TEXT_ELEMENT, TOKEN, TOKENS, VECTOR, ACTUAL_LABEL, PREDICTION, PROBABILITY


class TextCLAOElement(CLAOElement):
    """Basic element of a Text CLAO object."""
    def __init__(self, *args, **kwargs):
        """Generic CLAOElement specifically for use with TextCLAOs"""
        super(TextCLAOElement, self).__init__(*args, **kwargs)

    @classmethod
    def from_json(cls, json_dict: dict, clao: ClinicalLanguageAnnotationObject):
        pass

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: ClinicalLanguageAnnotationObject):
        pass


class Span(TextCLAOElement):
    """TextCLAOElement representing a span of text.

    Attributes:
        element_name: Name of element as it should appear in an XML tag, default is span
        start_offset(int): start offset for a Span
        end_offset(int): ending offset for a Span
        span_map: arbitrary dict of extra properties for this element. Default is None
    """
    element_name = SPAN

    def __init__(self, start_offset: int, end_offset: int, span_map=None, *args, **kwargs):
        """Create a Span."""
        super(Span, self).__init__(*args, **kwargs)
        self.start_offset = start_offset
        self.end_offset = end_offset
        self.map = {} if span_map is None else span_map

    def adjust_offsets(self, delta: int) -> None:
        """Adjust the start and end offsets of this Span by the provided delta.

        Args:
            delta (int): How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns:
            None
        """
        self.start_offset += delta
        self.end_offset += delta

    def to_json(self) -> Dict:
        """Add CLAO element with Span start and end attributes."""
        span_map = {k: str(v) for k, v in self.map.items()}
        return {**super(Span, self).to_json(),
                'start': str(self.start_offset),
                'end': str(self.end_offset),
                **span_map}

    def get_text_from(self, text_clao: 'TextCLAO') -> str:
        """Get the text for this Span from the TextCLAO that contains it.

        Args:
            text_clao: The TextCLAO object that contains this Span

        Returns:
            The text encompassed by this Span
        """
        return text_clao.get_text_for_span(self)

    def overlaps(self, other):
        """Determine if this Span overlaps another.

        Args:
            other: a Span
        Returns:
            Whether or not these two spans overlap at all
        """
        return self.start_offset < other.end_offset and self.end_offset > other.start_offset

    def contains(self, other):
        """Determine whether this Span contains another.

        Args:
            other: a Span
        Returns:
            Whether or not `other` is wholly contained within this Span
        """
        return self.start_offset <= other.start_offset and self.end_offset >= other.end_offset

    def get_span_embedding(self, method: Callable[[List[np.ndarray], Any], np.ndarray] = np.mean, **method_args
                           ) -> np.ndarray:
        """Get embedding vector for a span.

        Span must either be a EmbeddingContainer or TokenContainer. If span is a TokenContainer, embedding will be
        calculated using `method` argument on all sub-embeddings. `method` must take a list of numpy.ndarrays as its
        first argument and return a single numpy.ndarray.

        Args:
            method: method used to calculate Span embedding from sub embeddings. Defaults to numpy.mean()
            **method_args: extra arguments to be passed in to `method`

        Returns:
            embedding vector in np.ndarray format
        """
        if not method_args:
            method_args = {'axis': 0}
        if isinstance(self, EmbeddingContainer) and self.embedding is not None:
            return self.embedding.vector
        elif isinstance(self, TokenContainer):
            embedding_vectors = [t.embedding.vector for t in self.tokens]
            return method(embedding_vectors, **method_args)
        else:
            raise NotImplementedError('Embedding vector cannot be calculated for Spans that are not Tokens or do not '
                                      'contain Tokens with pre-calculated embeddings')

    def __len__(self):
        """Return the length of the Span."""
        return self.end_offset - self.start_offset

    def __eq__(self, other):
        """Compare the instances of the class."""
        return self.start_offset == other.start_offset and self.end_offset == other.end_offset and self.map == other.map

    def __str__(self):
        """Return the Span class object as a string."""
        return f'Span({self.start_offset}, {self.end_offset})'

    def __repr__(self):
        """Compute the string representation of a Span object."""
        return str(self)


class TextCLAO(Span, ClinicalLanguageAnnotationObject[str]):
    """The CLAO object for handling textual data.

    Class Attributes:
        element_name: Name of element as it should appear in an XML tag; default is annotation
        _top_level_elements: List of elements to be serialized in the top level of the annotation schema. All other CLAO
        elements to be serialized should be contained within one of these
    """
    element_name = ANNOTATION
    _top_level_elements = [TEXT_ELEMENT, SENTENCE, EMBEDDING, ENTITY_GROUP, ACTUAL_LABEL, PREDICTION, PROBABILITY]
    _text_clao_element_dict = None

    def __init__(self, raw_text: str, name: str, cfg: DictConfig = None, *args, **kwargs):
        """Create a TextCLAO."""
        super(TextCLAO, self).__init__(start_offset=0, end_offset=len(raw_text), raw_data=Text(raw_text, RAW_TEXT),
                                       name=name, cfg=cfg, *args, **kwargs)

    @classmethod
    def from_file(cls, input_path: str):
        """Create a new TextCLAO from a data file"""
        name = TextCLAO.get_base_name_from_path(input_path)
        with open(input_path, 'r') as f:
            return cls(f.read(), name)

    def to_json(self) -> Dict:
        """Add CLAO element with Span start and end attributes."""
        json_dict = super(TextCLAO, self).to_json()
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                json_dict[element_type] = [e.to_json() for e in self.elements[element_type]]
        return json_dict

    def to_xml(self) -> etree.Element:
        """Add CLAO element with annotation attributes to XML."""
        annotation = super(TextCLAO, self).to_xml(parent=None, attribs={})
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                for element in self.elements[element_type]:
                    element.to_xml(parent=annotation)
        return annotation

    @classmethod
    def from_xml_file(cls, file_path: str):
        """Load a CLAO element from an XML file"""
        xml = etree.parse(file_path)
        annotation = xml.getroot()
        xml_elements = annotation.getchildren()
        raw_text = xml_elements.pop(0)
        clao = cls(raw_text.text, cls.get_base_name_from_path(file_path))
        for xml_element in xml_elements:
            element_cls = cls.text_clao_element_dict()[xml_element.tag]
            clao_element = element_cls.from_xml(xml_element, clao)
            clao.insert_annotation(xml_element.tag, clao_element)
        return clao

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: ClinicalLanguageAnnotationObject):
        raise NotImplementedError("Method not implemented for class TextCLAO. Use from_xml_file() instead")

    def get_text_for_offsets(self, start: int, end: int) -> str:
        """Get the text between two offsets.

        Offsets should not be adjusted to account for the document's start_offset, as that is handled within this
        method.

        Args:
            start (int): Starting (absolute) offset
            end (int): Ending (absolute) offset

        Returns:
            the text between the two specified offsets
        """
        text = (self.get_annotations(Text, {'description': CLEANED_TEXT})
                or self.get_annotations(Text, {'description': RAW_TEXT})).raw_text
        return text[start - self.start_offset:end - self.start_offset]

    def get_text_for_span(self, span: Span) -> str:
        """Get the text for a given Span contained within this document.

           clao.get_text_for_span(span) and span.get_text_from(clao) are equivalent calls.

           Args:
               span: the span class object

           Returns:
               the text between the two specified offsets
        """
        return self.get_text_for_offsets(span.start_offset, span.end_offset)

    @classmethod
    def text_clao_element_dict(cls) -> Dict[str, Type[TextCLAOElement]]:
        """Get dict of TextCLAOElement.element_name -> TextCLAOElement. Used for recreating CLAOs from files."""
        if not cls._text_clao_element_dict:
            cls._text_clao_element_dict = {c.element_name: c for c in TextCLAOElement.__subclasses__()}
            for c in list(cls._text_clao_element_dict.values()):
                cls._text_clao_element_dict.update(TextCLAO._get_subclass_element_dict(c))
        return cls._text_clao_element_dict

    @classmethod
    def _get_subclass_element_dict(cls, subclass) -> Dict[str, Type[TextCLAOElement]]:
        """Helper method of cls.text_clao_element_dict to make recursive calls into subclasses."""
        classes = {c.element_name: c for c in subclass.__subclasses__()}
        for c in list(classes.values()):
            classes.update(cls._get_subclass_element_dict(c))
        return classes

    @staticmethod
    def get_base_name_from_path(file_path):
        """Get the base name in a specified path.

        Args:
            file_path: a specified path

        Returns:
            a base name
        """
        return os.path.splitext(os.path.basename(file_path))[0]


class IdSpan(IdCLAOElement, Span):
    """A basic Span with an id"""
    element_name = 'id_span'

    def __init__(self, start_offset: int, end_offset: int, element_id: int, span_map=None, **kwargs):
        """Create an IdSpan."""
        super().__init__(element_id=element_id, start_offset=start_offset, end_offset=end_offset, span_map=span_map,
                         **kwargs)

    @classmethod
    def from_span(cls, span: Span, elemend_id: int) -> 'IdSpan':
        """Given a Span and an id, create an IdSpan"""
        return IdSpan(span.start_offset, span.end_offset, elemend_id, span.map)

    def __eq__(self, other):
        return super().__eq__(other) and self.element_id == other.element_id


class Embedding(IdCLAOElement, TextCLAOElement):
    """CLAO element representing an embedding vector"""
    element_name = EMBEDDING

    def __init__(self, element_id: int, vector: np.ndarray, **kwargs):
        """Create an Embedding."""
        super(Embedding, self).__init__(element_id=element_id, **kwargs)
        self.vector = vector

    def to_json(self) -> Dict:
        """Add CLAO element with Embedding attributes."""
        return {**super(Embedding, self).to_json(), VECTOR: list(self.vector)}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Embedding attributes to XML."""
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        vector = attribs.pop(VECTOR)
        embedding = super(Embedding, self).to_xml(parent, attribs)
        embedding.text = str(vector)
        return embedding

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: ClinicalLanguageAnnotationObject):
        """Get Embedding element from CLAO."""
        return cls(xml_element.attrib['id'], np.fromstring(xml_element.text[1:-1], sep=', '))

    def __eq__(self, other):
        return super(Embedding, self).__eq__() and all(self.vector == other.vector)

    def __str__(self):
        return f'Embedding({len(self.vector)})'


class TextCLAOElementContainer(CLAOElementContainer):
    """Base class to create mix-ins that give container functionality to TextCLAOElements. i.e. Paragprahs contain
    Sentences, and Sentences contain Tokens"""
    def __init__(self, clao: TextCLAO, **kwargs):
        super(TextCLAOElementContainer, self).__init__(clao=clao, **kwargs)

    @classmethod
    def _from_xml_process_children(cls, xml_element: etree._Element, clao: TextCLAO) -> Dict[str, Tuple[int, int]]:
        """Process any child elements of XML tree, and return element id ranges for each child type.

        Args:
            any child elements of XML tree

        Returns:
            element id ranges for each child type
        """
        children_ranges = {}
        for child_elem in xml_element.getchildren():
            tag = child_elem.tag
            element_cls = clao.text_clao_element_dict()[tag]
            element = element_cls.from_xml(child_elem, clao)
            if tag not in children_ranges:
                children_ranges[tag] = [element.element_id, 0]
            children_ranges[tag][1] = element.element_id + 1
            clao.insert_annotation(tag, element)
        return {k: tuple(v) for k, v in children_ranges.items()}


class EmbeddingContainer(TextCLAOElementContainer):
    """Add class to any CLAOElement class to create a container for an embedding.

    Args:
        embedding_id (int): embedding id
        clao: the CLAO information to process
    """
    def __init__(self, embedding_id: int, clao: TextCLAO, **kwargs):
        """Create an EmbeddingContainer."""
        self._embedding_id = embedding_id
        super(EmbeddingContainer, self).__init__(clao=clao, **kwargs)

    @property
    def embedding(self) -> Optional[Embedding]:
        """Return a list of Embedding element class, or single CLAOElement of Embedding element class."""
        if self._embedding_id is not None:
            return self.clao.get_annotations(Embedding, self._embedding_id)
        else:
            return None


class Token(IdSpan, EmbeddingContainer):
    """CLAO element representing a token.

    Properties:
        embedding: Optional word embedding for this token

    Args:
        start_offset (int):  start offset for a Token
        end_offset (int): ending offset for a Token
        element_id (int): element id for Token
        clao (TextCLAO): the CLAO information to process
        text (str): the text for the Token
        embedding_id (int): embedding id. The default is None
        span_map: arbitrary dict of extra properties for this element. Default is None. The default is None
    """
    element_name = TOKEN

    def __init__(self, start_offset: int, end_offset: int, element_id: int, clao: TextCLAO, text: str,
                 embedding_id: int = None, span_map=None, **kwargs):
        """Create a Token."""
        super(Token, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id, clao=clao,
                                    embedding_id=embedding_id, span_map=span_map, **kwargs)
        self.text = text

    @classmethod
    def from_id_span(cls, span: IdSpan, text: str, clao: TextCLAO) -> 'Token':
        """Given an IdSpan, create a Token.

        Args:
            span: IdSpan element class
            text (str): the text for the Token
            clao: the CLAO information to process

        Returns:
            Token element class
        """
        return Token(span.start_offset, span.end_offset, span.element_id, clao, text, span.map)

    def to_json(self) -> Dict:
        """Add CLAO element with Token attributes.

        Returns:
            a json dictionary with embedding id
        """
        json_dict = {**super(Token, self).to_json(), TEXT: self.text}
        if self._embedding_id is not None:
            json_dict[EMBEDDING_ID] = self._embedding_id
        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Token attributes to XML.

        Returns:
            a token with attributes in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        if EMBEDDING_ID in attribs:
            attribs[EMBEDDING_ID] = str(attribs[EMBEDDING_ID])

        text = attribs.pop(TEXT)
        token = super(Token, self).to_xml(parent, attribs)
        token.text = text
        return token

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Token element from CLAO in a XML format.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Token element class
        """
        start_offset = int(xml_element.attrib.pop('start'))
        end_offset = int(xml_element.attrib.pop('end'))
        element_id = int(xml_element.attrib.pop('id'))
        text = xml_element.text
        try:
            embedding_id = int(xml_element.attrib.pop('embedding_id'))
        except KeyError:
            embedding_id = None

        return cls(start_offset, end_offset, element_id, clao, text, embedding_id, xml_element.attrib)

    def __str__(self):
        return f'Token({self.start_offset}, {self.end_offset}, {self.text})'

    def __eq__(self, other):
        return super().__eq__(other) and self._embedding_id == other._embedding_id


class TokenContainer(TextCLAOElementContainer):
    """Create a TokenContainer."""
    def __init__(self, token_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._token_id_range = token_id_range
        super(TokenContainer, self).__init__(clao=clao, **kwargs)

    @property
    def tokens(self) -> List[Token]:
        """Return a list of Token element class, or single CLAOElement of Token element class."""
        if self._token_id_range:
            return self.clao.get_annotations(Token, self._token_id_range)
        else:
            return []


class Heading(Span):
    """CLAO element representing a heading."""
    element_name = HEADING

    def __init__(self, start_offset: int, end_offset: int, text: str):
        """Create a Heading.

        Args:
            start_offset (int):  start offset for a Heading
            end_offset (int): ending offset for a Heading
            text:  the text for the Heading
        """
        super(Heading, self).__init__(start_offset, end_offset)
        self.text = text

    def to_json(self) -> Dict:
        """Add CLAO element with Heading attributes.

        Returns:
            a dictionary with Heading and Heading attributes.
        """
        return {**super(Heading, self).to_json(), TEXT: self.text}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Heading attributes to XML.

        Returns:
            a heading with attributes in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        heading = super(Heading, self).to_xml(parent, attribs)
        heading.text = text
        return heading

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Heading element from CLAO in a XML format.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Heading element class
        """
        start_offset = int(xml_element.attrib.pop('start'))
        end_offset = int(xml_element.attrib.pop('end'))
        text = xml_element.text

        return cls(start_offset, end_offset, text)

    def __str__(self):
        return f'Heading({self.start_offset}, {self.end_offset}, {self.text})'

    def __eq__(self, other):
        return super().__eq__(other) and self.text == other.text


class Entity(IdSpan, TokenContainer):
    """CLAO element representing an entity.

    Properties:
        tokens: List of tokens that make up the entity
    """
    element_name = ENTITY

    def __init__(self, element_id: int, entity_type: EntityType, confidence: float, literal: str, label: str,
                 token_id_range: Tuple[int, int], clao: TextCLAO, start_offset: int = None, end_offset: int = None,
                 span_map=None):
        """Create an Entity CLAOElement.

        Args:
            element_id: unique id for this element
            entity_type: Type of entity represented (e.g NER or FACT)
            confidence: percentage of confidence in accuracy of entity
            literal: name of the concept represented by this entity
            label: type of concept represented by this entity
            token_id_range: range [x,y) of token ids contained within this entity, where x is inclusive and y exclusive
            clao: TextCLAO this Entity is contained within
            start_offset: beginning index of this element in text (only needs to be included if Tokens in token_id_range
                          do not yet exist
            end_offset: ending index (non-inclusive) of this element in text (only needs to be included if Tokens in
                        token_id_range do not yet exist
            span_map: arbitrary dict of extra properties for this element
        """
        self.entity_type = entity_type
        self.confidence = confidence
        self.literal = literal
        self.label = label

        start_offset = start_offset if start_offset else clao.get_annotations(Token, token_id_range[0]).start_offset
        end_offset = end_offset if end_offset else clao.get_annotations(Token, token_id_range[1]-1).end_offset
        super(Entity, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id, clao=clao,
                                     token_id_range=token_id_range, span_map=span_map)

    def to_json(self) -> Dict:
        """Add CLAO element with Entity attributes.

        Returns:
            a dictionary with Entity attributes
        """
        return {**super(Entity, self).to_json(),
                'token_ids': f"[{self._token_id_range[0]}, {self._token_id_range[1]})",
                'type': self.entity_type.name,
                'confidence': str(self.confidence),
                'label': str(self.label),
                LITERAL: self.literal}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Entity attributes to XML.

        Returns:
            a entity element class
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        literal = attribs.pop(LITERAL)
        entity = super(Entity, self).to_xml(parent, attribs)
        entity.text = literal
        return entity

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Entity element from CLAO in a XML format.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Entity element class
        """
        element_id = int(xml_element.attrib.pop('id'))
        start_offset = int(xml_element.attrib.pop('start'))
        end_offset = int(xml_element.attrib.pop('end'))
        entity_type = EntityType(xml_element.attrib.pop('type'))
        confidence = float(xml_element.attrib.pop('confidence'))
        label = xml_element.attrib.pop('label')
        literal = xml_element.text
        token_id_range = xml_element.attrib.pop('token_ids')[1:-1].split(', ')
        token_id_range = tuple(int(i) for i in token_id_range)

        return cls(element_id, entity_type, confidence, literal, label, token_id_range, clao, start_offset, end_offset,
                   xml_element.attrib)

    def __str__(self):
        return f'Entity({self.start_offset}, {self.end_offset}, {self.entity_type.name}, tokens: {len(self.tokens)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self.entity_type == other.entity_type \
               and self.confidence == other.confidence \
               and self._token_id_range == other._token_id_range


class EntityContainer(TextCLAOElementContainer):
    """Create an EntityContainer."""
    def __init__(self, entity_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        self._entity_id_range = entity_id_range
        super(EntityContainer, self).__init__(clao=clao, **kwargs)

    @property
    def entities(self) -> List[Entity]:
        """Return a list of Entity element class, or single CLAOElement of Entity element class."""
        if self._entity_id_range:
            return self.clao.get_annotations(Entity, self._entity_id_range)
        else:
            return []


class EntityGroup(IdCLAOElement, TextCLAOElement):
    """CLAO element representing an entity group."""
    element_name = ENTITY_GROUP

    def __init__(self, element_id: int, entity_type: EntityType, literal: str, label: str, **kwargs):
        """Create an EntityGroup."""
        super(EntityGroup, self).__init__(element_id=element_id, **kwargs)
        self.entity_type = entity_type
        self.literal = literal
        self.label = label

    def to_json(self) -> Dict:
        """Add CLAO element with Entity Group attributes."""
        return {**super(EntityGroup, self).to_json(),
                'entity_type': self.entity_type.name,
                'literal': self.literal,
                'label': self.label
                }

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Entity Group attributes to XML."""
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        literal = attribs.pop(LITERAL)
        entity_group = super(EntityGroup, self).to_xml(parent, attribs)
        entity_group.text = literal
        return entity_group

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get EntityGroup element from CLAO in a XML format.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Entity element class
        """
        element_id = int(xml_element.attrib.pop('id'))
        entity_type = EntityType(xml_element.attrib.pop('entity_type'))
        label = EntityType(xml_element.attrib.pop('label'))
        literal = xml_element.text

        return cls(element_id, entity_type, label, literal)
        return cls(element_id, entity_type, literal)

    def __str__(self):
        return f'EntityGroup({self.entity_type.name}, {self.literal})'

    def __eq__(self, other):
        return super().__eq__(other) and self.entity_type == other.entity_type and self.literal == other.literal \
               and self.label == other.label


class Sentence(IdSpan, TokenContainer, EntityContainer, EmbeddingContainer):
    """CLAO element representing a sentence.

    Properties:
        tokens: List of tokens contained in the sentence
        entities: List of entities contained in the sentence
        embedding: Optional embedding for this Sentence
    """
    element_name = SENTENCE

    def __init__(self, start_offset: int, end_offset: int, element_id: int, clao: TextCLAO,
                 entity_id_range: Optional[Tuple[int, int]] = None, token_id_range: Optional[Tuple[int, int]] = None,
                 embedding_id: int = None, span_map=None, **kwargs):
        """Create a Sentence.

        Args:
            start_offset: beginning index of this element in text (only needs to be included if Tokens in token_id_range
                          do not yet exist
            end_offset: ending index (non-inclusive) of this element in text (only needs to be included if Tokens in
                        token_id_range do not yet exist
            element_id: unique id for this element
            clao: TextCLAO this Entity is contained within
            entity_id_range: range [x,y) of entity ids contained within this entity group, where x is inclusive and
                y exclusive
            entity_id_range: range [x,y) of token ids contained within this entity, where x is inclusive and
                y exclusive
            embedding_id: id for the embedding
            span_map: arbitrary dict of extra properties for this element
        """
        super(Sentence, self).__init__(start_offset=start_offset, end_offset=end_offset, element_id=element_id,
                                       clao=clao, entity_id_range=entity_id_range, token_id_range=token_id_range,
                                       embedding_id=embedding_id, span_map=span_map, **kwargs)

    def all_subspans(self) -> List[Span]:
        """Get all Spans contained within this document in a single list.

        Returns:
            a unified list with all the members of this object that are Spans
        """
        all_spans: List[Span] = []
        all_spans.extend(self.tokens)
        all_spans.extend(self.entities)
        return all_spans

    def adjust_offsets(self, delta: int) -> None:
        """Adjust the start and end offsets of this Sentence (and any existing spans therein) by the provided delta.

        Args:
            delta: How much to shift the offsets. To reduce the offsets, this should be negative.

        Returns:
            None
        """
        super().adjust_offsets(delta)
        for span in self.all_subspans():
            span.adjust_offsets(delta)

    @classmethod
    def from_id_span(cls, span: IdSpan, clao: TextCLAO, entity_id_range: Optional[Tuple[int, int]] = None,
                     token_id_range: Optional[Tuple[int, int]] = None, embedding_id: int = None) -> 'Sentence':
        """Given an IdSpan and lists of Tokens, Entities, etc create a new Sentence"""
        return cls(span.start_offset, span.end_offset, span.element_id, clao, entity_id_range, token_id_range,
                   embedding_id, span.map)

    def to_json(self) -> Dict:
        """Add CLAO element with Sentence attributes.

        Returns:
            a json dictionary with Entities and Tokens attributes
        """
        json_dict = {**super(Sentence, self).to_json(),
                     ENTITIES: [e.to_json() for e in self.entities],
                     TOKENS: [t.to_json() for t in self.tokens]}

        if self._embedding_id is not None:
            json_dict[EMBEDDING_ID] = self._embedding_id

        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Sentence attributes to XML.

        Returns:
            Sentence element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        if EMBEDDING_ID in attribs:
            attribs[EMBEDDING_ID] = str(attribs[EMBEDDING_ID])
        attribs.pop(ENTITIES)
        attribs.pop(TOKENS)
        sentence = super(Sentence, self).to_xml(parent, attribs)
        for entity in self.entities:
            entity.to_xml(parent=sentence)
        for token in self.tokens:
            token.to_xml(sentence)
        return sentence

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Sentence element from CLAO in a XML format.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Sentence element class
        """
        children_ranges = cls._from_xml_process_children(xml_element, clao)
        start_offset = int(xml_element.attrib.pop('start'))
        end_offset = int(xml_element.attrib.pop('end'))
        element_id = int(xml_element.attrib.pop('id'))
        entity_id_range = children_ranges.get(Entity.element_name)
        token_id_range = children_ranges.get(Token.element_name)

        try:
            embedding_id = int(xml_element.attrib.pop('embedding_id'))
        except KeyError:
            embedding_id = None

        return cls(start_offset, end_offset, element_id, clao, entity_id_range, token_id_range, embedding_id,
                   xml_element.attrib)

    def get_tokens_for_span(self, span: Span, partial=False) -> List[Token]:
        """Get the Tokens in this sentence that are covered by a given Span.

        Args:
            span: the Span to retrieve Tokens for
            partial: if True, will return Tokens that are only partially covered by the Span.
                     Otherwise, a Token will only be included if it is fully covered by the Span.
        Returns:
            A list of Tokens covered by the Span
        """
        if span.contains(self):
            return self.tokens
        elif partial:
            return [t for t in self.tokens if t.overlaps(span)]
        else:
            return [t for t in self.tokens if span.contains(t)]

    def get_entity_spans_for_span(self, span, partial=False) -> List[Entity]:
        """Get the Entities in this sentence that are covered by a given Span.

        Args:
            span: the Span to retrieve Entities for
            partial: if True, will return Entities that are only partially covered by the Span.
                     Otherwise, a Entity will only be included if it is fully covered by the Span.
        Returns:
            A list of Entities covered by the Span
        """
        if span.contains(self):
            return self.entities
        elif partial:
            return [es for es in self.entities if es.overlaps(span)]
        else:
            return [es for es in self.entities if span.contains(es)]

    def __str__(self):
        return f'Sentence({self.start_offset}, {self.end_offset}, tokens: {len(self.tokens)}, ' \
               f'entities: {len(self.entities)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self._token_id_range == other._token_id_range \
               and self._entity_id_range == other._entity_id_range \
               and self._embedding_id == other._embedding_id


class SentenceContainer(TextCLAOElementContainer):
    """Add class to any CLAOElement class to create a container for sentences"""
    def __init__(self, sentence_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        """Create a SentenceContainer."""
        self._sentence_id_range = sentence_id_range
        super(SentenceContainer, self).__init__(clao=clao, **kwargs)

    @property
    def sentences(self) -> List[Sentence]:
        """Return a list of Sentence element class, or single CLAOElement of Sentence element class."""
        if self._sentence_id_range:
            return self.clao.get_annotations(Sentence, self._sentence_id_range)
        else:
            return []


class Paragraph(TextCLAOElement, SentenceContainer):
    """CLAO element representing a paragraph

    Properties:
        sentences: List of sentences contained in the paragraph
    """
    element_name = PARAGRAPH

    def __init__(self, clao: TextCLAO, sentence_id_range: Optional[Tuple[int, int]] = None, **kwargs):
        """Create a Paragraph.

        Args:
            clao (TextCLAO): the CLAO information to process
            sentence_id_range (Tuple[int, int]): sentence id range. Optional, the default is None
        """
        super(Paragraph, self).__init__(clao=clao, sentence_id_range=sentence_id_range, **kwargs)

    def to_json(self) -> Dict:
        """Add CLAO element with Paragraph attributes.

        Returns:
            a json dictionary with Paragraph attributes
        """
        return {**super(Paragraph, self).to_json(), SENTENCES: [s.to_json() for s in self.sentences]}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Paragraph attributes to XML.

        Returns:
            Paragraph element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        attribs.pop(SENTENCES)
        paragraph = super(Paragraph, self).to_xml(parent, attribs)
        for sentence in self.sentences:
            sentence.to_xml(parent=paragraph)
        return paragraph

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Paragraph element from CLAO in a XML.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Paragraph element class
        """
        children_ranges = cls._from_xml_process_children(xml_element, clao)
        sentence_id_range = children_ranges.get(Sentence.element_name)
        return cls(clao, sentence_id_range)

    def __str__(self):
        return f'Paragraph(sentences: {len(self.sentences)})'

    def __eq__(self, other):
        return super().__eq__(other) and self._sentence_id_range == other._sentence_id_range


class ParagraphContainer(TextCLAOElementContainer):
    """Add class to any CLAOElement class to create a container for paragraphs"""
    def __init__(self, paragraph_id_range: Tuple[int, int], clao: TextCLAO, **kwargs):
        """Create a ParagraphContainer."""
        self._paragraph_id_range = paragraph_id_range
        super(ParagraphContainer, self).__init__(clao=clao, **kwargs)

    @property
    def paragraphs(self) -> List[Paragraph]:
        """Return a list of Paragraph element class, or single CLAOElement of Paragraph element class."""
        if self._paragraph_id_range:
            return self.clao.get_annotations(Paragraph, self._paragraph_id_range)
        else:
            return []


class Section(Span, ParagraphContainer):
    """CLAO element representing a section

    Properties:
        paragraphs: List of paragraphs contained in the section
        heading: Optional heading object for the section
    """
    element_name = SECTION

    def __init__(self, start_offset: int, end_offset: int, clao: TextCLAO,
                 paragraph_id_range: Optional[Tuple[int, int]] = None, heading_id: Optional[int] = None, **kwargs):
        """Create a Section."""
        super(Section, self).__init__(start_offset=start_offset, end_offset=end_offset, clao=clao,
                                      paragraph_id_range=paragraph_id_range, **kwargs)
        self._heading_id = heading_id

    @property
    def heading(self) -> Optional[Heading]:
        """Return a list of Heading element class, or single CLAOElement of Heading element class."""
        if self._heading_id is not None:
            return self.clao.get_annotations(Heading, self._heading_id)
        else:
            return None

    def to_json(self) -> Dict:
        """Add CLAO element with Section attributes.

        Returns:
            a json dictionary with Section attributes
        """
        json_dict = {**super(Section, self).to_json(), PARAGRAPHS: [p.to_json() for p in self.paragraphs]}
        if self.heading is not None:
            json_dict[HEADING] = self.heading.to_json()
        return json_dict

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Section attributes to XML.

        Returns:
            Section element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        attribs.pop(PARAGRAPHS)
        if HEADING in attribs:
            attribs.pop(HEADING)
        section = super(Section, self).to_xml(parent, attribs)

        if self.heading is not None:
            self.heading.to_xml(parent=section)
        for paragraph in self.paragraphs:
            paragraph.to_xml(parent=section)
        return section

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Section element from CLAO in a XML.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Section element class
        """
        children_ranges = cls._from_xml_process_children(xml_element, clao)

        start_offset = int(xml_element.attrib.pop('start'))
        end_offset = int(xml_element.attrib.pop('end'))
        paragraph_id_range = children_ranges.get(Paragraph.element_name)

        try:
            heading_id = int(xml_element.attrib.pop('heading_id'))
        except KeyError:
            heading_id = None

        return cls(start_offset, end_offset, clao, paragraph_id_range, heading_id)

    def __str__(self):
        return f'Section({self.start_offset}, {self.end_offset}, paragraphs: {len(self.paragraphs)})'

    def __eq__(self, other):
        return super().__eq__(other) \
               and self._paragraph_id_range == other._paragraph_id_range \
               and self._heading_id == other._heading_id


class Text(Span):
    """CLAO object representing unannotated text."""
    element_name = TEXT_ELEMENT

    def __init__(self, rax_text: str, description: str):
        """Create a Text."""
        super(Text, self).__init__(0, len(rax_text))
        self.raw_text = rax_text
        self.description = description

    def to_json(self) -> Dict:
        """Add CLAO element with Text attributes.

        Returns:
            a json dictionary with Text attributes
        """
        return {**super(Text, self).to_json(), TEXT: self.raw_text, DESCRIPTION: self.description}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with Text attributes to XML.

        Returns:
            Text element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        text = attribs.pop(TEXT)
        raw_text = super(Text, self).to_xml(parent, attribs)
        raw_text.text = text
        return raw_text

    @classmethod
    def from_xml(cls, xml_element: etree._Element, clao: TextCLAO):
        """Get Text element from CLAO in a XML.

        Args:
            xml_element (etree._Element): Element from etree
            clao (TextCLAO): the CLAO information to process

        Returns:
            a Text element class
        """
        description = xml_element.attrib.pop(DESCRIPTION)
        text = xml_element.text
        return cls(text, description)

    def __str__(self):
        return f'RawText({len(self.raw_text)})'

    def __eq__(self, other):
        return super().__eq__(other) and self.raw_text == other.raw_text


class ActualLabel(TextCLAOElement):
    """CLAO element representing an actual label."""
    element_name = ACTUAL_LABEL

    def __init__(self, actual_label_value, *args, **kwargs):
        """Create a ActualLabel."""
        super(ActualLabel, self).__init__(*args, **kwargs)
        self.actual_label_value = actual_label_value

    def to_json(self) -> Dict:
        """Add CLAO element with actual label attributes.

        Returns:
            a json dictionary with actual label attributes
        """
        return {**super(ActualLabel, self).to_json(), ACTUAL_LABEL: self.actual_label_value}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with actual label attributes to XML.

        Returns:
            actual label element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        actual_label_value = attribs.pop(ACTUAL_LABEL)
        actual_label = super(ActualLabel, self).to_xml(parent, attribs)
        actual_label.text = str(actual_label_value)
        return actual_label


class PredictProbabilities(TextCLAOElement):
    """CLAO element representing predicted probabilities."""
    element_name = PROBABILITY

    def __init__(self, probability: float, *args, **kwargs):
        """Create a PredictProbability."""
        super(PredictProbabilities, self).__init__(*args, **kwargs)
        self.probability = probability

    def to_json(self) -> Dict:
        """Add CLAO element with predicted probabilities attributes.

        Returns:
            a json dictionary with predicted probabilities attributes
        """
        return {**super(PredictProbabilities, self).to_json(), PROBABILITY: self.probability}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with predicted probabilities to XML.

        Returns:
            predicted probabilities element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        predicted_prob = attribs.pop(PROBABILITY)
        predicted_probs = super(PredictProbabilities, self).to_xml(parent, attribs)
        predicted_probs.text = str(predicted_prob)
        return predicted_probs


class Predictions(TextCLAOElement):
    """CLAO element representing predictions."""
    element_name = PREDICTION

    def __init__(self, prediction: float, *args, **kwargs):
        """Create a Prediction."""
        super(Predictions, self).__init__(*args, **kwargs)
        self.prediction = prediction

    def to_json(self) -> Dict:
        """Add CLAO element with predictions attributes

        Returns:
            a json dictionary with predictions attributes
        """
        return {**super(Predictions, self).to_json(), PREDICTION: self.prediction}

    def to_xml(self, parent: Optional[etree.Element], attribs: Dict[str, str] = None):
        """Add CLAO element with predictions to XML.

        Returns:
            predictions element in XML format
        """
        if attribs:
            attribs.update(self.to_json())
        else:
            attribs = self.to_json()

        prediction = attribs.pop(PREDICTION)
        predictions = super(Predictions, self).to_xml(parent, attribs)
        predictions.text = str(prediction)
        return predictions
