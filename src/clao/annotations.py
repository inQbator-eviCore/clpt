"""Initial thoughts / rough draft of some internal text annotation objects, in line with XML schema illustrated in 2022
NaaCL paper"""

import itertools
from typing import Dict, Iterable, Optional, Type

from lxml import etree


class CLAOIdCounter(object):
    _ids = {}

    @classmethod
    def get_id_for_class(cls, class_to_id: Type['CLAOElement']):
        cls._ids.setdefault(class_to_id, itertools.count())
        return next(cls._ids[class_to_id])


class CLAOElement:
    xml_element_name = 'element'

    def __init__(self):
        """add docstring here"""
        self.id = CLAOIdCounter.get_id_for_class(self.__class__)

    @property
    def xml_attrs(self) -> Dict[str, str]:
        return {'id': str(self.id)}

    def to_xml(self, parent: Optional[etree.Element]):
        if parent is None:
            return etree.Element(self.xml_element_name, **self.xml_attrs)
        else:
            return etree.SubElement(parent, self.xml_element_name, **self.xml_attrs)


class Span(CLAOElement):
    xml_element_name = 'span'

    def __init__(self, start: int, end: int):
        """add docstring here"""
        super(Span, self).__init__()
        self.start = start
        self.end = end

    @property
    def xml_attrs(self) -> Dict[str, str]:
        attrs = super(Span, self).xml_attrs
        attrs.update({'start': str(self.start),
                      'end': str(self.end)})
        return attrs


class Token(Span):
    xml_element_name = 'token'

    def __init__(self, start: int, end: int, lemma: str, stem: str, pos: str, text: str):
        """add docstring here"""
        super(Token, self).__init__(start, end)
        self.lemma = lemma
        self.stem = stem
        self.pos = pos
        self.text = text

    @property
    def xml_attrs(self) -> Dict[str, str]:
        attrs = super(Token, self).xml_attrs
        attrs.update({'lemma': self.lemma,
                      'stem': self.stem,
                      'pos': self.pos})
        return attrs

    def to_xml(self, parent: etree.Element):
        token = super(Token, self).to_xml(parent)
        token.text = self.text
        return token


class Heading(Span):
    xml_element_name = 'heading'

    def __init__(self, start: int, end: int, text: str):
        """add docstring here"""
        super(Heading, self).__init__(start, end)
        self.text = text

    def to_xml(self, parent: etree.Element):
        heading = super(Heading, self).to_xml(parent)
        heading.text = self.text
        return heading


class Entity(CLAOElement):
    xml_element_name = 'entity'

    def __init__(self, tokens: Iterable[Token], entity_type: str, confidence: float, text: str):
        """add docstring here"""
        super(Entity, self).__init__()
        self.tokens = tokens
        self.entity_type = entity_type
        self.confidence = confidence
        self.text = text

    @property
    def xml_attrs(self) -> Dict[str, str]:
        attrs = super(Entity, self).xml_attrs
        token_ids = [token.id for token in self.tokens]
        attrs.update({'tokens': str(token_ids),
                      'type': self.entity_type,
                      'confidence': str(self.confidence)})
        return attrs

    def to_xml(self, parent: etree.Element):
        entity = super(Entity, self).to_xml(parent)
        entity.text = self.text
        return entity


class Sentence(CLAOElement):
    xml_element_name = 'sentence'

    def __init__(self, entities: Iterable[Entity], tokens: Iterable[Token]):
        """add docstring here"""
        super(Sentence, self).__init__()
        self.entities = entities
        self.tokens = tokens

    def to_xml(self, parent: etree.Element):
        sentence = super(Sentence, self).to_xml(parent)
        for entity in self.entities:
            entity.to_xml(parent=sentence)
        for token in self.tokens:
            token.to_xml(sentence)
        return sentence


class Paragraph(CLAOElement):
    xml_element_name = 'paragraph'

    def __init__(self, sentences: Iterable[Sentence]):
        """add docstring here"""
        super(Paragraph, self).__init__()
        self.sentences = sentences

    def to_xml(self, parent: etree.Element):
        paragraph = super(Paragraph, self).to_xml(parent)
        for sentence in self.sentences:
            sentence.to_xml(parent=paragraph)
        return paragraph


class Section(Span):
    xml_element_name = 'section'

    def __init__(self, start: int, end: int, paragraphs: Iterable[Paragraph],
                 heading: Optional[Heading] = None):
        """add docstring here"""
        super(Section, self).__init__(start, end)
        self.paragraphs = paragraphs
        self.heading = heading

    def to_xml(self, parent: etree.Element):
        section = super(Section, self).to_xml(parent)
        if self.heading is not None:
            self.heading.to_xml(parent=section)
        for paragraph in self.paragraphs:
            paragraph.to_xml(parent=section)
        return section


class Annotations(CLAOElement):
    xml_element_name = 'annotation'

    def __init__(self, sections: Iterable[Section]):
        """add docstring here"""
        super(Annotations, self).__init__()
        self.sections = sections

    @property
    def xml_attrs(self) -> Dict[str, str]:
        return {}

    def to_xml(self) -> etree.Element:
        annotation = super(Annotations, self).to_xml(parent=None)
        for section in self.sections:
            section.to_xml(parent=annotation)
        return annotation
