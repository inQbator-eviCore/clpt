"""Clinical Language Annotation Object.

This model provides functions to create a general Clinical Language Annotation Object and specifically to create text
annotation objects (TextCLAOs), in line with XML schema illustrated in 2022 NaaCL paper. Functions to support and create
CLAOs other than text (e.g. image, audio) will be provided in the future release of the CLPT.

The CLAO objects will be created via specifying in the YAML configuration file the location and data_type with ingestion
module (see configurations files in `src/clpt/conf/config.yaml` and `src/clpt/conf/ingestion/default.yaml`). Based on
the configuration value for `data_type` in `src/clpt/conf/ingestion/default.yaml`, the corresponding CLAO supporting
data type will be returned via `CLAODataType` class.

The CLAO could be created from a data file or reconstituted from a CLAO XML file. Also, processed results from each step
of the pipeline stages will be stored to the corresponding elements. The current elements which are available in Text
CLAO are annotation, text, embedding, entity, entity_group, heading, paragraph, section, sentence, token, actual_label,
prediction, probability.vector, actual_label, prediction, probability.

The CLAO supports addition, deletion, and update operations along with the enhancement of annotations through the use
B-Trees for indexing (Johnson and Sasha, 1993). B-Tree indexing within the CLAO is performed at O(log n) speed for
operations on CLAO elements. To implement B-Trees, we have used python blist package.

The CLAO(s) are human-readable and could be easily shared across projects as they could be exported to JSON format or
XML format."""
