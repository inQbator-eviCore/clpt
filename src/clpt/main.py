import os

import hydra
from omegaconf import DictConfig

from src.clpt.constants import CONFIG_FILE, CONFIG_FILEPATH
from src.clpt.ingestion.document_collector import DocumentCollector
from src.utils import add_new_key_to_cfg


@hydra.main(config_path=CONFIG_FILEPATH, config_name=CONFIG_FILE)
def main(cfg: DictConfig) -> None:
    cfg.ingestion.input_dir = os.path.realpath(cfg.ingestion.input_dir)
    add_new_key_to_cfg(cfg, os.getcwd(), 'ingestion', 'output_dir')
    dc = DocumentCollector(cfg.ingestion.input_dir, cfg.ingestion.data_type)
    dc.serialize_all(cfg.ingestion.output_dir)


if __name__ == '__main__':
    main()
