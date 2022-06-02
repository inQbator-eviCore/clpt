"""Ingestion module to ingest raw data and the target outcome/gold standard annotations into CLAO(s).

The CLPT is designed to be able to accept any form of input such as text, speech, video or images. At this
point, we have experimented with text only and will continue to work on functions to support other data type.
DocumentCollector is a document reader to read in data and store in the CLAO(s). OutcomeCollector is another reader
to read in target labels or gold standard annotations and store them in the CLAO(s). User could specify the text files
and outcome file in a configuration file (a .yaml file)

The ingestion module is the initial creation of the CLAO(s) and then passes the CLAO(s) for downstream processing in
the analysis engine."""
