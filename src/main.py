import logging
import os

import hydra
from omegaconf import DictConfig

from src.clpt.ingestion.document_collector import DocumentCollector
from src.clpt.pipeline.pipeline_processor import NlpPipelineProcessor
from src.constants.constants import CONFIG_FILE, CONFIG_FILEPATH
from src.utils import add_new_key_to_cfg

logger = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_FILEPATH, config_name=CONFIG_FILE)
def main(cfg: DictConfig) -> None:
    cfg.ingestion.input_dir = os.path.realpath(cfg.ingestion.input_dir)
    add_new_key_to_cfg(cfg, os.getcwd(), 'ingestion', 'output_dir')
    logger.info(f"Ingesting documents from {cfg.ingestion.input_dir}")
    dc = DocumentCollector(cfg.ingestion.input_dir, cfg.ingestion.data_type)
    dc.serialize_all(cfg.ingestion.output_dir)

    logger.info("Building pipeline")
    pipeline = NlpPipelineProcessor.from_stages_config(cfg)

    logger.info("Running pipeline stages over individual CLAOs")
    for clao in dc.claos:
        pipeline.process(clao)
    logger.info("Running pipeline stages over entire corpus")
    pipeline.process_multiple(dc.claos)

    logger.info("Serializing CLAOs")
    dc.serialize_all(cfg.ingestion.output_dir)


if __name__ == '__main__':
    main()
