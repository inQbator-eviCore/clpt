"""Analysis engine module.

The analysis engine is designed to further process the CLAO object, adding or modifying any annnotations in the process.
The goal of the analysis engine is to allow initial preparation for classification. Therefore, several pre-processing
techniques are added to it: document cleaning, lemmatization, stemming, tokenization, sentence detection,
part-of-speech (POS) tagging, word embeddings and sentence embedding. We will continue to expand and improve the
pre-processing techniques in the CLPT.

Embeddings are stored in CLAO objects to allow comparison between tokens and spans of tokens. This is done by assigning
a vector to each token or token group where the group returns an average of all of the embeddings within it. The CLPT
also has a configuration mechanism for changing the token group method of calculation.

In the current design, the functions for pre-processing mentioned above are located under the pipeline model."""
