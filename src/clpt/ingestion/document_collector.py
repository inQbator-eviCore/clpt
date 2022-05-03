import os
from typing import List, Union
import glob

from src.clao.clao import CLAODataType, ClinicalLanguageAnnotationObject
from src.clao.text_clao import TextCLAO


class DocumentCollector:
    def __init__(self, input_dir, data_type: Union[str, CLAODataType]):
        """add docstring here"""
        if isinstance(data_type, str):
            data_type = CLAODataType(data_type)
        self.claos = self.ingest(input_dir, data_type)

    @staticmethod
    def ingest(input_dir: str, data_type: CLAODataType) -> List[ClinicalLanguageAnnotationObject]:
        # Select the appropriate CLAO class for the given data type. Text is the only type currently accepted.
        # More types to be implemented in the future
        if data_type is CLAODataType.TEXT:
            clao_cls = TextCLAO
        else:
            raise NotImplementedError(f"DocumentCollector not implemented for data type '{data_type.name}'")

        return [clao_cls.from_file(os.path.join(input_dir, file))
                for file in glob.glob(os.path.join(input_dir, '*'))
                if os.path.isfile(file)]

    def serialize_all(self, output_dir: str):
        for clao in self.claos:
            clao.write_as_xml(output_dir)
