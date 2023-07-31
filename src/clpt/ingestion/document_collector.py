"""DocumentCollector is a document reader to read in data and store in the CLAO(s)."""
import os
from typing import List, Union
import glob
from src.clao.audio_clao import AudioCLAO
from src.clao.clao import CLAODataType, ClinicalLanguageAnnotationObject
from src.clao.text_clao import TextCLAO, METAINFO


class DocumentCollector:
    """DocumentCollector to read in data and ingest the data in the CLAO(s)."""
    def __init__(self, input_dir, project_name, project_desc, creation_date,
                 project_input_link, project_version,
                 data_type: Union[str, CLAODataType],
                 files_to_ignore):
        """Ingest raw data to a list of CLAOs.

        Args:
            input_dir: the path to the input file(s)
            data_type(Union[str, CLAODataType]): the type of raw data to ingest into CLAO
            file_to_ignore: files that will not be ingested with DocumentCollector()
        """
        if isinstance(data_type, str):
            data_type = CLAODataType(data_type)
        self.claos = self.ingest(input_dir, data_type, files_to_ignore)
        textclaos = self.claos
        self.ingest_attributes_clao(textclaos, project_name, project_desc,
                                    creation_date, project_input_link,
                                    project_version)

    @staticmethod
    def ingest_attributes_clao(textclaos, project_name, project_desc,
                               creation_date, project_input_link,
                               project_version):
        metdata = METAINFO(project_name, project_desc, creation_date,
                           project_input_link, project_version)
        for clao in textclaos:
            clao.insert_annotation(METAINFO, metdata)

    @staticmethod
    def ingest(input_dir: str, data_type: CLAODataType, files_to_ignore) -> List[ClinicalLanguageAnnotationObject]:
        """Ingest raw data into CLAO(s).

        Args:
            input_dir: the path to the input file(s)
            data_type(Union[str, CLAODataType]): the type of raw data to ingest into CLAO
            file_to_ignore: files that will not be ingested with DocumentCollector()

        Returns:
            A list of CLAOs with raw data ingested into each of the CLAOs
        """
        # Select the appropriate CLAO class for the given data type. Text is the only type currently accepted.
        # More types to be implemented in the future
        if data_type is CLAODataType.TEXT:
            clao_cls = TextCLAO
        elif data_type is CLAODataType.AUDIO:
            clao_cls = AudioCLAO
        else:
            raise NotImplementedError(f"DocumentCollector not implemented for data type '{data_type.name}'")

        return [clao_cls.from_file(os.path.join(input_dir, file))
                for file in glob.glob(os.path.join(input_dir, '*'))
                if os.path.isfile(file) and not file.endswith(tuple(files_to_ignore))]

    def serialize_all(self, output_dir: str):
        """Serialize to the XML format.

        Args:
            output_dir(str): an output location for the serialized CLAO(s)
        """
        for clao in self.claos:
            # clao.write_as_xml(output_dir)
            clao.write_as_json(output_dir)
