# Primary Indicator
This repository provides code to predict an important step of selecting
what is known as the “primary indicator” – one
of the several heuristics based on [clinical guidelines](https://www.evicore.com/-/media/files/evicore/clinical-guidelines/solution/cardiology-and-radiology/2020/evicore_spine_eff021420_pub101519_upd012920.pdf) that are published and publicly available.

Some of the experiments uses BiowordVec and BioSentVec models to generate embeddings. Both the models can be downloaded from the below [link] (https://github.com/ncbi-nlp/BioSentVec)


## Ingesting data into CLPT
The Data Ingestion Documentation provides a streamlined process for incorporating raw text files into the data processing pipeline. By following the recommended steps to store the files and configure the directory path, users can seamlessly integrate unstructured input data for further processing and analysis.

### Input data 
- The Data Ingestion Documentation provides a concise guide for integrating raw text files into the data processing pipeline.  Input data can be unstructured data typically from an attending physician or nurse that describes the patient's condition, the treatment provided.

### Raw text
Create a designated folder to store the raw text files in `src/resources/` folder.

### Config file
Locate the YAML configuration file in `src/clpt/conf/ingestion/`.
Find and update the parameter value with the path to the folder containing the raw text files.


### gold standard outcomes file
- Gold Standard Outcomes file serves as a comprehensive guide for organizing and structuring annotations or target labels in JSON or CSV format. For classification and prediction tasks, provide the gold standards at the document level.

#### File Structure:
To maintain clarity and ease of use, the gold standard outcomes should be stored in the same folder as the corresponding raw text files. The following file formats are supported:

#### JSON Format:
Each JSON file should be named after its respective raw text file.
The JSON file should contain a single key-value pair.
The key should be the name of the raw text file.
The value should be a list of annotations or target labels associated with the document.

#### CSV Format:
A CSV file named "gold_standard_outcomes.csv" should be created.
The CSV file should contain two columns: "doc_name" and "actual_label".
The "doc_name" column should contain the name of the raw text file.
The "actual_label" column should contain the target label for classification or prediction.

### To run models
To replicate the experiments in the paper. Please refer to the table to run the modules. Experiments 3 and 4 uses pretrained model downloaded from the [link](https://github.com/ncbi-nlp/BioSentVec)
#### Supervised

| SlNo  | Model Name  | Script to run  |
|-----------|-----------|-----------|
|1|BOW + ML | python main.py  --config-name=config_pi_baseline_cv.yaml |
|2|TF-IDF + ML | python main.py  --config-name=config_pi_baseline_tf.yaml |
|3|BioWordVec + RFC|python main.py  --config-name=config_wordvec_ml.yaml | 
|4|BioSentVec + RFC|python main.py  --config-name=config_sent2vec_ml.yaml | 
|5|BioSentVec + CNN|python main.py  --config-name=config_sent2vec_DL.yaml |
|6|BioBERT|Refer to external scripts folder | 
|7|BlueBERT|Refer to external scripts folder | 

#### UnSupervised

| SlNo  | Model Name  | Script to run  |
|-----------|-----------|-----------|
|1|BiosentVec with Taxonomy| python main.py  --config-name=config_sent2vec_taxonomy.yaml |
|2|FastText with Taxonomy| python main.py  --config-name=config_fastText_taxonomy.yaml |