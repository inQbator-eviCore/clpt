import logging
from keras.layers import Input, Dense, MaxPooling1D, \
    concatenate
from keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras.layers import Conv1D
from keras.layers import Dropout, Flatten
from typing import List
from src.clao.text_clao import TextCLAO, Predictions
from src.clpt.pipeline.stages.pipeline_stage import PipelineStage
from src.utils import extract_vector_from_claos_ml, extract_gold_standard_ds_filter_from_claos
import numpy as np
import pandas as pd
logger = logging.getLogger(__name__)


class CNNModel(PipelineStage):
    """ Classification  class to run a sklearn ml model and generate predictions.

        Args:
            model: machine learning model name
            parameters: Model parameters to be passed if any
        """

    def __init__(self, **kwargs):
        super(CNNModel, self).__init__(**kwargs)
        self.single_clao = False

    def process(self, claos: List[TextCLAO]) -> None:
        """Perform classification on the data

        Args:
            clao_info: the CLAO information to process
        """
        train_embeddings = extract_vector_from_claos_ml(claos, 'train')
        train_labels = extract_gold_standard_ds_filter_from_claos(claos, 'train')
        test_embeddings = extract_vector_from_claos_ml(claos, 'test')
        X_train = np.array(list(train_embeddings.values()))
        X_test = np.array(list(test_embeddings.values()))
        y_train = np.array(list(train_labels.values()))
        y_train_pad = to_categorical((y_train))
        # sequence_length = X_train.shape[1]
        drop = 0.5
        input_emb = Input(shape=(700, 1))
        conv_0 = Conv1D(filters=100, kernel_size=3, activation='relu')(input_emb)
        conv_1 = Conv1D(filters=100, kernel_size=4, activation='relu')(input_emb)
        conv_2 = Conv1D(filters=100, kernel_size=5, activation='relu')(input_emb)
        maxpool_0 = MaxPooling1D(pool_size=2, strides=1)(conv_0)
        maxpool_1 = MaxPooling1D(pool_size=2, strides=1)(conv_1)
        maxpool_2 = MaxPooling1D(pool_size=2, strides=1)(conv_2)
        merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2],
                                    axis=1)
        flatten = Flatten()(merged_tensor)
        dropout = Dropout(drop)(flatten)
        output = Dense(units=26, activation='softmax',
                       kernel_regularizer=regularizers.l2(0.01))(dropout)
        model = Model(input_emb, output)
        logger.info(model.summary())
        adam = Adam(learning_rate=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['acc'])
        callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
        model.fit(X_train, y_train_pad, epochs=3,
                  verbose=1, callbacks=callbacks)
        # starts train
        model.save('biosentvec.h5')
        predicted = model.predict(X_test)
        pred = np.argmax(np.array(predicted), axis=1)
        df = pd.DataFrame(test_embeddings.keys(), columns=['doc_name'])
        df['probab'] = pred
        if 'probab' in df.columns:
            for clao in claos:
                try:
                    probability_d = df.loc[df['doc_name'].astype(str) == clao.name, 'probab'].item()
                    clao.insert_annotation(Predictions, Predictions(float(probability_d)))
                except ValueError as e:
                    # TODO Better handling for repeat lines
                    logger.info("Execution error at unsupervised")
                    logger.info(e)
                    pass
