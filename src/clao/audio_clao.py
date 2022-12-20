"""Create a AudioCALO.

Initial version of the audio objects."""
from typing import Dict
import numpy as np
from lxml import etree
from omegaconf import DictConfig
from src.clao.clao import CLAOElement, ClinicalLanguageAnnotationObject
from src.constants.annotation_constants import ANNOTATION, META_INFO, RAW_AUDIO
import soundfile


class AudioCLAO(ClinicalLanguageAnnotationObject[str]):
    """The CLAO object for handling audio data.

    Class Attributes:
        element_name: Name of element as it should appear in an XML tag; default is annotation
        _top_level_elements: List of elements to be serialized in the top level of the annotation schema. All other CLAO
        elements to be serialized should be contained within one of these
    """
    element_name = ANNOTATION
    _top_level_elements = [META_INFO]
    _text_clao_element_dict = {}

    def __init__(self, raw_audio: np.ndarray, name: str, cfg: DictConfig = None, *args, **kwargs):
        """Create a AudioCLAO."""
        super(AudioCLAO, self).__init__(start_offset=0, end_offset=len(raw_audio),
                                        raw_data=Audio(raw_audio, RAW_AUDIO),
                                        name=name, cfg=cfg, *args, **kwargs)

    @classmethod
    def from_file(cls, path):
        speech, rate = soundfile.read(path)
        return cls(speech, path)

    def to_json(self) -> Dict:
        """Add CLAO element with Span start and end attributes."""
        json_dict = {}
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                json_dict[element_type] = [e.to_json() for e in self.elements[element_type]]
        return json_dict

    def to_xml(self) -> etree.Element:
        """Add CLAO element with annotation attributes to XML."""
        annotation = super(AudioCLAO, self).to_xml()
        for element_type in self._top_level_elements:
            if element_type in self.elements:
                for element in self.elements[element_type]:
                    element.to_xml(parent=annotation)
        return annotation


class Audio(CLAOElement):
    """Base Audio class"""

    element_name = RAW_AUDIO

    def __init__(self, raw_audio, description) -> None:
        super(Audio).__init__()
        self.raw_data = raw_audio
        self.description = description

    def __str__(self):
        return f'RawAudio({len(self.raw_data)})'

    def __eq__(self, other):
        return super().__eq__(other) and self.raw_data == other.raw_data and self.description == other.description
