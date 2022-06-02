"""Classification module.

The classification module perform classification task via retrieving information from the CLAO which have been added
frp, the upstream CLPT component(s) in the data ingestion module and analysis engine module. In this classification
module, machine learning and other techniques (e.g., heuristics) could be applied to further augment annotations for
classification tasks before evaluation. Some of the major components to be released in the CLPT are:
(1) acronym expansion (similar to CARD (Wu et al., 2017)); (2) mention detection; (3) fact extraction, whcih will be
used to extract clinical concepts from the mentions to better disambiguate clinical notes and provide fact-based
evidence for classification; (4) relationship extraction is used to further expand mention detection to allow linking
entities and the creation of a knowledge graph – to be presented as future work.

In the current design, a light Mention Detection is provided. We are in active development of a more complex
MentionDetection methods as well as other components for classification, which will be released when they are ready."""
