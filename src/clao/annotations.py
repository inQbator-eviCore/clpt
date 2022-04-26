"""Initial thoughts / rough draft of some internal text annotation objects, in line with XML schema illustrated in 2022
NaaCL paper"""

import itertools
from typing import Dict, Iterable, Optional, Type, Union

from lxml import etree

from src.constants import ANNOTATION, ELEMENT, ENTITIES, ENTITY, HEADING, PARAGRAPH, PARAGRAPHS, RAW_TEXT, \
    SECTION, SENTENCE, SENTENCES, SPAN, TEXT, TOKEN, TOKENS


class CLAOIdCounter(object):
    _ids = {}

    @classmethod
    def get_id_for_class(cls, class_to_id: Type['CLAOElement']):
        cls._ids.setdefault(class_to_id, itertools.count())
        return next(cls._ids[class_to_id])


class CLAOElement:
    element_name = ELEMENT

    def __init__(self):
        """add docstring here"""
        self.id = CLAOIdCounter.get_id_for_class(self.__class__)

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        return {'id': str(self.id)}

    def to_json(self) -> Dict:
        return self.serializable_attributes

    def to_xml(self, parent: Optional[etree.Element]):
        if parent is None:
            return etree.Element(self.element_name, **self.serializable_attributes)
        else:
            return etree.SubElement(parent, self.element_name, **self.serializable_attributes)


class Span(CLAOElement):
    element_name = SPAN

    def __init__(self, start: int, end: int):
        """add docstring here"""
        super(Span, self).__init__()
        self.start = start
        self.end = end

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        attrs = super(Span, self).serializable_attributes
        attrs.update({'start': str(self.start),
                      'end': str(self.end)})
        return attrs


class Token(Span):
    element_name = TOKEN

    def __init__(self, start: int, end: int, lemma: str, stem: str, pos: str, text: str):
        """add docstring here"""
        super(Token, self).__init__(start, end)
        self.lemma = lemma
        self.stem = stem
        self.pos = pos
        self.text = text

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        attrs = super(Token, self).serializable_attributes
        attrs.update({'lemma': self.lemma,
                      'stem': self.stem,
                      'pos': self.pos})
        return attrs

    def to_json(self) -> Dict:
        json_dict = super(Token, self).to_json()
        json_dict[TEXT] = self.text
        return json_dict

    def to_xml(self, parent: etree.Element):
        token = super(Token, self).to_xml(parent)
        token.text = self.text
        return token


class Heading(Span):
    element_name = HEADING

    def __init__(self, start: int, end: int, text: str):
        """add docstring here"""
        super(Heading, self).__init__(start, end)
        self.text = text

    def to_json(self) -> Dict:
        json_dict = super(Heading, self).to_json()
        json_dict[TEXT] = self.text
        return json_dict

    def to_xml(self, parent: etree.Element):
        heading = super(Heading, self).to_xml(parent)
        heading.text = self.text
        return heading


class Entity(CLAOElement):
    element_name = ENTITY

    def __init__(self, tokens: Iterable[Token], entity_type: str, confidence: float, text: str):
        """add docstring here"""
        super(Entity, self).__init__()
        self.tokens = tokens
        self.entity_type = entity_type
        self.confidence = confidence
        self.text = text

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        attrs = super(Entity, self).serializable_attributes
        token_ids = [token.id for token in self.tokens]
        attrs.update({'tokens': str(token_ids),
                      'type': self.entity_type,
                      'confidence': str(self.confidence)})
        return attrs

    def to_json(self) -> Dict:
        json_dict = super(Entity, self).to_json()
        json_dict[TEXT] = self.text
        return json_dict

    def to_xml(self, parent: etree.Element):
        entity = super(Entity, self).to_xml(parent)
        entity.text = self.text
        return entity


class Sentence(Span):
    element_name = SENTENCE

    def __init__(self, entities: Iterable[Entity], tokens: Iterable[Token]):
        """add docstring here"""
        super(Sentence, self).__init__()
        self.entities = entities
        self.tokens = tokens

    def to_json(self) -> Dict:
        json_dict = super(Sentence, self).to_json()
        json_dict[ENTITIES] = [e.to_json() for e in self.entities]
        json_dict[TOKENS] = [t.to_json() for t in self.tokens]
        return json_dict

    def to_xml(self, parent: etree.Element):
        sentence = super(Sentence, self).to_xml(parent)
        for entity in self.entities:
            entity.to_xml(parent=sentence)
        for token in self.tokens:
            token.to_xml(sentence)
        return sentence


class Paragraph(CLAOElement):
    element_name = PARAGRAPH

    def __init__(self, sentences: Iterable[Sentence]):
        """add docstring here"""
        super(Paragraph, self).__init__()
        self.sentences = sentences

    def to_json(self) -> Dict:
        json_dict = super(Paragraph, self).to_json()
        json_dict[SENTENCES] = [s.to_json() for s in self.sentences]
        return json_dict

    def to_xml(self, parent: etree.Element):
        paragraph = super(Paragraph, self).to_xml(parent)
        for sentence in self.sentences:
            sentence.to_xml(parent=paragraph)
        return paragraph


class Section(Span):
    element_name = SECTION

    def __init__(self, start: int, end: int, paragraphs: Iterable[Paragraph],
                 heading: Optional[Heading] = None):
        """add docstring here"""
        super(Section, self).__init__(start, end)
        self.paragraphs = paragraphs
        self.heading = heading

    def to_json(self) -> Dict:
        json_dict = super(Section, self).to_json()
        if self.heading is not None:
            json_dict[HEADING] = self.heading.to_json()
        json_dict[PARAGRAPHS] = [p.to_json() for p in self.paragraphs]
        return json_dict

    def to_xml(self, parent: etree.Element):
        section = super(Section, self).to_xml(parent)
        if self.heading is not None:
            self.heading.to_xml(parent=section)
        for paragraph in self.paragraphs:
            paragraph.to_xml(parent=section)
        return section


class Annotations(CLAOElement):
    element_name = ANNOTATION

    def __init__(self, elements: Dict[str, Union[CLAOElement, Iterable[CLAOElement]]]):
        """add docstring here"""
        super(Annotations, self).__init__()
        self.elements = elements

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        return {}

    def to_json(self) -> Dict:
        json_dict = super(Annotations, self).to_json()
        for element_type, element_value in self.elements.items():
            if isinstance(element_value, list):
                json_dict[element_type] = [e.to_json() for e in element_value]
            else:
                json_dict.update(element_value.to_json())
        return json_dict

    def to_xml(self) -> etree.Element:
        annotation = super(Annotations, self).to_xml(parent=None)
        for element_type, element_list in self.elements.items():
            for element in element_list:
                element.to_xml(parent=annotation)
        return annotation


class RawText(CLAOElement):
    element_name = RAW_TEXT

    def __init__(self, rax_text: str):
        """add docstring here"""
        super(RawText, self).__init__()
        self.raw_text = rax_text

    @property
    def serializable_attributes(self) -> Dict[str, str]:
        return {}

    def to_json(self) -> Dict:
        json_dict = super(RawText, self).to_json()
        json_dict[RAW_TEXT] = self.raw_text
        return json_dict

    def to_xml(self, parent: Optional[etree.Element]):
        raw_text = super(RawText, self).to_xml(parent)
        raw_text.text = self.raw_text
        return raw_text
