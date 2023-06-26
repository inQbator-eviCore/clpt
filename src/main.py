"""Run CLPT based on configuration specified in src/clpt/conf"""

import logging
import os
import hydra
from omegaconf import DictConfig
from src.clpt.ingestion.document_collector import DocumentCollector
from src.clpt.ingestion.truth_collector import TruthCollector
from src.clpt.pipeline.pipeline_processor import NlpPipelineProcessor
from src.clpt.evaluation.evaluator import Evaluator
from src.clpt.pipeline.stages.classification.Unsupervised import Embedding_Distance
from src.constants.constants import CONFIG_FILE, CONFIG_FILEPATH
from src.utils import add_new_key_to_cfg
from datetime import datetime

logger = logging.getLogger(__name__)


@hydra.main(config_path=CONFIG_FILEPATH, config_name=CONFIG_FILE)
def main(cfg: DictConfig) -> None:
    # ingest document file(s)
    cfg.ingestion.input_dir = os.path.realpath(cfg.ingestion.input_dir)
    add_new_key_to_cfg(cfg, os.getcwd(), 'ingestion', 'output_dir')
    logger.info(f"Ingesting documents from {cfg.ingestion.input_dir}")
    logger.info(cfg.ingestion.project_name)
    dc = DocumentCollector(cfg.ingestion.input_dir, cfg.ingestion.project_name, cfg.ingestion.project_desc,
                           datetime.now(), cfg.ingestion.project_input_link, cfg.ingestion.project_version,
                           cfg.ingestion.data_type, ['.csv', '.json'],)

    # add gold-standard to each CLAO object
    cfg.ingestion.outcome_file_name = os.path.join(cfg.ingestion.input_dir, cfg.ingestion.outcome_file_name)
    gold_standard_outcome = TruthCollector(dc=dc, outcome_file=cfg.ingestion.outcome_file_name,
                                           outcome_type=cfg.ingestion.outcome_type)
    gold_standard_outcome.ingest()
    dc.serialize_all(cfg.ingestion.output_dir)
    # add taxonomy
    if("taxonomy_dir" in cfg.ingestion):
        cfg.ingestion.input_dir = os.path.realpath(cfg.ingestion.taxonomy_dir)
        logger.info(f"Ingesting documents from {cfg.ingestion.taxonomy_dir}")
        logger.info(cfg.ingestion.project_name)
        td = DocumentCollector(cfg.ingestion.taxonomy_dir, cfg.ingestion.project_name, cfg.ingestion.project_desc,
                               datetime.now(), cfg.ingestion.project_input_link, cfg.ingestion.project_version,
                               cfg.ingestion.data_type, ['.csv', '.json'],)

        td.serialize_all(cfg.ingestion.output_dir)
    # add stages to the pipeline
    logger.info("Building pipeline")
    pipeline = NlpPipelineProcessor.from_stages_config(cfg)
    i = 0
    # process each CLAO
    logger.info("Running pipeline stages over individual CLAOs")
    for clao in dc.claos:
        pipeline.process(clao)
        i += 1
        logger.info(i)
    if("taxonomy_dir" in cfg.ingestion):
        j = 0
        logger.info("Running pipeline stages over individual CLAOs")
        for clao in td.claos:
            pipeline.process(clao)
            j += 1
            logger.info(j)
    logger.info("Running pipeline stages over entire corpus")

    if("taxonomy_dir" in cfg.ingestion):
        unsup = Embedding_Distance(model_name=cfg.classification.pipeline_stages[0].model_name)
        unsup.process(dc.claos, td.claos)
    else:
        pipeline.process_multi_labels(dc.claos)

    # evaluate the performance
    # to do better way to handle evaluation
    if "evaluation" in cfg:
        eval = Evaluator(outcome_type=cfg.ingestion.outcome_type, target_dir=cfg.ingestion.output_dir, claos=dc.claos,
                         threshold=cfg.evaluation.threshold)
    eval = Evaluator(outcome_type=cfg.ingestion.outcome_type, target_dir=cfg.ingestion.output_dir, claos=dc.claos,
                     threshold=None)
    eval.calculate_metrics(claos=dc.claos)

    # serialize CLAO to xml format or json format
    logger.info("Serializing CLAOs")
    dc.serialize_all(cfg.ingestion.output_dir)


if __name__ == '__main__':
    main()
