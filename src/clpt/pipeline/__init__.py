"""pipeline consists of stages from Analysis engine module and Classification module.

Users could specify the stages needed for the analysis and classification tasks. Method `build_pipeline_stages` will
add each of the stages from the configuration files into NlpPipelineProcessor and apply the stages to each of the CLAOs.

All stages available are located under dir `src/clpt/pipeline/stages/`.

Several pre-processing techniques are added to it: document cleaning, lemmatization, stemming, tokenization,
sentence detection, part-of-speech (POS) tagging, word embeddings and sentence embedding. We will continue to expand
and improve the pre-processing techniques in the CLPT.

For classification, some of the major components to be released in the CLPT are: (1) acronym expansion
(similar to CARD (Wu et al., 2017)); (2) mention detection; (3) fact extraction, whcih will be used to extract clinical
concepts from the mentions to better disambiguate clinical notes and provide fact-based evidence for classification;
(4) relationship extraction is used to further expand mention detection to allow linking entities and the creation of a
knowledge graph – to be presented as future work.

References:
    - Wu, Yonghui, et al. "A long journey to short abbreviations: developing an open-source framework for
    clinical abbreviation recognition and disambiguation (CARD)." Journal of the American Medical Informatics
    Association 24.e1 (2017): e79-e86."""
