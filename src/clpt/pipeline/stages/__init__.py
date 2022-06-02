"""Stages module contains functions for the analysis module and the classification module to be added to the pipeline.

Users could specify the pre-processing stages and classification tasks needed for the CLAO(s) in the configuration file
(a .yaml). Based on the information in the configuration file, each of CLAOs will go through the corresponding stages
in the pipeline and processed outputs will also be stored in the CLAO(s).

Currently, the following pre-processing stages are supported: document cleaning (remove stop words, convert to lower
case, exclude punctuations), spell check, lemmatization, stemming, tokenization, sentence detection, part-of-speech
(POS) tagging, word embeddings and sentence embedding. Specifically, a default pipeline with customized functions plus
functions from NLTK package is supported and another pipeline which loads some pre-processing methods from spaCy is
also supported.

Currently, a light rule-based MentionDetection based on medspacy is provided for classification. We are in active
development of a more complex MentionDetection methods as well as other components for classification, which will be
released when they are ready."""
