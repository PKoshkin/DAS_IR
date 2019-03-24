from collections import Counter
from sklearn.model_selection import KFold
import numpy as np
from scipy.sparse import csr_matrix
import re
import os
import shutil

from IPython.display import clear_output
from collections import defaultdict

from keras.models import Sequential, Model
from keras.losses import mean_absolute_error
from keras.layers import Embedding, LSTM, Bidirectional, Dense, Input, Masking, Lambda, concatenate
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback

BATCH_SIZE = 32
TEST_DATA_SIZE = 10000
TRAIN_DATA_SIZE = 490000
QUERY_DICT_SIZE = 247074
DOCUMENT_DICT_SIZE = 583954
ACTIVATION = 'relu'
HIDDEN_SIZE = 256

def get_max_lens(filename):
    max_query_len, max_document_len = -1, -1
    with open(filename) as handler:
        for line in handler:
            query, document = line.split("\t")
            query_len = len(query.split())
            document_len = len(document.split())
            if query_len > max_query_len:
                max_query_len = query_len
            if document_len > max_document_len:
                max_document_len = document_len
    return max_query_len, max_document_len

def cycle_file(filename):
    while True:
        with open(filename) as f:
            yield from f
            
def make_seq_batches_generator(filename, batch_size, max_query_len, max_document_len):
    with open(filename) as handler:
        while True:
            query_batch = np.zeros([batch_size, max_query_len])
            docment_batch = np.zeros([batch_size, max_document_len])
            for i in range(batch_size):
                line = next(handler)
                query, document = line.split("\t")
                query = list(map(int, query.split()))
                document = list(map(int, document.split()))
                query_batch[i, :len(query)] = query
                docment_batch[i, :len(document)] = document
            yield query_batch, docment_batch

def make_non_seq_batches_generator(filename, batch_size):
    with open(filename) as handler:
        while True:
            query_batch = np.zeros([batch_size, QUERY_DICT_SIZE])
            docment_batch = np.zeros([batch_size, DOCUMENT_DICT_SIZE])
            for i in range(batch_size):
                line = next(handler)
                query, document = line.split("\t")
                query = list(map(int, query.split()))
                document = list(map(int, document.split()))
                for word in query:
                    query_batch[i, word] += 1
                for word in document:
                    docment_batch[i, word] += 1
            yield query_batch, docment_batch

def make_data_generator(positive_generator, negative_generator):
    while True:
        positive_query_batch, positive_docment_batch = next(positive_generator)
        negative_query_batch, negative_docment_batch = next(negative_generator)
        query_input = np.concatenate([positive_query_batch, negative_query_batch], axis=0)
        document_input = np.concatenate([positive_docment_batch, negative_docment_batch], axis=0)
        labels = np.concatenate(
            [np.ones(len(positive_query_batch)), -1 * np.ones(len(negative_query_batch))
        ]).reshape([-1, 1])
        yield (
            {'query_input': query_input, 'document_input': document_input},
            {'output': labels}
        )

def make_batch(positive_lines_generator, negative_lines_generator, batch_size):
    lines = [next(positive_lines_generator) for i in range(batch_size)]
    lines += [next(negative_lines_generator) for i in range(batch_size)]
    inputs = make_input_from_lines(lines)
    labels = np.concatenate([np.ones(batch_size), -1 * np.ones(batch_size)]).reshape([-1, 1])
    return inputs, {'output': labels}

def make_input_from_lines(lines):
    query_indices_axis_0, document_indices_axis_0 = [], []
    query_indices_axis_1, document_indices_axis_1 = [], []
    query_values, document_values = [], []
    for i, line in enumerate(lines):
        query, document = line.split("\t")
        query_words = Counter(map(int, query.split()))
        document_words = Counter(map(int, document.split()))

        for word in query_words:
            query_indices_axis_0.append(i)
            query_indices_axis_1.append(word)
            query_values.append(query_words[word])
        for word in document_words:
            document_indices_axis_0.append(i)
            document_indices_axis_1.append(word)
            document_values.append(document_words[word])

    query_batch = csr_matrix(
        (query_values, (query_indices_axis_0, query_indices_axis_1)),
        shape=(len(lines), QUERY_DICT_SIZE)
    )
    docment_batch = csr_matrix(
        (document_values, (document_indices_axis_0, document_indices_axis_1)),
        shape=(len(lines), DOCUMENT_DICT_SIZE)
    )

    return {'query_input': query_batch, 'document_input': docment_batch}
    
class DataGenerator:
    def __init__(self, positive_filename, negative_filename, model, batch_size, negative_candidates_num):
        self._positive_generator = cycle_file(positive_filename)
        self._negative_generator = cycle_file(negative_filename)
        self._model = model
        self._batch_size = batch_size
        self._negative_candidates_num = negative_candidates_num

    def make_batch(self):
        lines = [next(self._negative_generator) for _ in range(self._negative_candidates_num)]
        inputs = make_input_from_lines(lines)
        predictions = -self._model.predict(inputs).reshape(-1)
        indices = np.argsort(predictions)[:self._batch_size]
        negative_lines = np.array(lines)[indices]
        return make_batch(self._positive_generator, (line for line in negative_lines), self._batch_size)
    
def make_generator(data_generator):
    while True:
        yield data_generator.make_batch()
        
def my_cosine_proximity(y_true, y_pred):
    mean_pred = K.mean(y_pred)
    return -K.mean((y_pred - mean_pred) * y_true)

def mean_positive_score(y_true, y_pred):
    filter_mult = (y_true + 1) / 2
    return K.mean(y_pred * filter_mult)

def mean_positive_var(y_true, y_pred):
    mean_positive = mean_positive_score(y_true, y_pred)
    filter_mult = (y_true + 1) / 2
    return K.mean((y_pred * filter_mult - mean_positive) ** 2)

def get_pred(y_true, y_pred):
    mean_positive = mean_positive_score(y_true, y_pred)
    mean_negative = mean_negative_score(y_true, y_pred)
    
    threshold = (mean_positive + mean_negative) / 2
    
    positive_mult = (y_true + 1) / 2
    negative_mult = (1 - y_true) / 2

    return K.mean((y_pred * filter_mult - mean_positive) ** 2)

def mean_negative_score(y_true, y_pred):
    filter_mult = (1 - y_true) / 2
    return K.mean(y_pred * filter_mult)

def mean_negative_var(y_true, y_pred):
    mean_negative = mean_negative_score(y_true, y_pred)
    filter_mult = (1 - y_true) / 2
    return K.mean((y_pred * filter_mult - mean_negative) ** 2)

def normalize(embedding):
    return K.l2_normalize(embedding, axis=-1)

def dot_product(embeddings):
    return K.sum(embeddings[0] * embeddings[1], axis=-1)

def reshape_to_prediction(score):
    return K.reshape(score, (-1, 1))

def loss(y_true, y_pred):
    return 0.01 * mean_absolute_error(y_true, y_pred) + my_cosine_proximity(y_true, y_pred)
    
def make_lstm_branch(tensor, lstm_num, dense_num, hidden_size, dict_size, activation):
    tensor = Masking(mask_value=0)(tensor)
    tensor = Embedding(dict_size, hidden_size)(tensor)  # shape: (BATCH_SIZE, dict_size, hidden_size)
    for i in range(lstm_num - 1):
        tensor = Bidirectional(LSTM(hidden_size, return_sequences=True))(tensor)  # shape: (BATCH_SIZE, hidden_size)
    tensor = Bidirectional(LSTM(hidden_size))(tensor)  # shape: (BATCH_SIZE, hidden_size)
    for i in range(dense_num):
        tensor = Dense(hidden_size, activation=activation)(tensor)  # shape: (BATCH_SIZE, hidden_size)
    return Lambda(normalize)(tensor)  # shape: (BATCH_SIZE, hidden_size)

def make_dense_branch(tensor, dense_num, hidden_size, dict_size, activation):
    for i in range(dense_num):
        tensor = Dense(hidden_size, activation=activation)(tensor)  # shape: (BATCH_SIZE, hidden_size)
    return Lambda(normalize)(tensor)  # shape: (BATCH_SIZE, hidden_size)

def make_compiled_model(input_1, input_2, embedding_1, embedding_2):
    score = Lambda(dot_product)([embedding_1, embedding_2])
    prediction = Lambda(reshape_to_prediction, name="output")(score)

    model = Model(inputs=[input_1, input_2], outputs=prediction)
    model.compile(
        Adam(),
        loss=my_cosine_proximity,
        metrics=[mean_positive_score, mean_negative_score, mean_positive_var, mean_negative_var, 'acc']
    )
    return model

def make_dense_model(document_dict_size,
                     document_dense_num,
                     query_dict_size,
                     query_dense_num,
                     activation,
                     hidden_size):
    # shape: (BATCH_SIZE, QUERY_DICT_SIZE)
    query_input = Input(shape=(query_dict_size,), sparse=True, name="query_input")
    query_embedding = make_dense_branch(
        query_input,
        query_dense_num,
        hidden_size,
        query_dict_size,
        activation
    )
    
    # shape: (BATCH_SIZE, document_dict_size)
    document_input = Input(shape=(document_dict_size,), sparse=True, name="document_input")
    document_embedding = make_dense_branch(
        document_input,
        document_dense_num,
        hidden_size,
        document_dict_size,
        activation
    )

    return make_compiled_model(query_input, document_input, query_embedding, document_embedding)
    
def make_dense_not_dssm_model(document_dict_size,
                              document_dense_num,
                              query_dict_size,
                              query_dense_num,
                              common_dense_num,
                              activation,
                              hidden_size):
    # shape: (BATCH_SIZE, QUERY_DICT_SIZE)
    query_input = Input(shape=(query_dict_size,), sparse=True, name="query_input")
    query_tensor = query_input
    for i in range(query_dense_num):
        # shape: (BATCH_SIZE, hidden_size)
        query_tensor = Dense(hidden_size, activation=activation)(query_tensor)
    
    # shape: (BATCH_SIZE, document_dict_size)
    document_input = Input(shape=(document_dict_size,), sparse=True, name="document_input")
    document_tensor = document_input
    for i in range(document_dense_num):
        # shape: (BATCH_SIZE, hidden_size)
        document_tensor = Dense(hidden_size, activation=activation)(document_tensor)

    # shape: (BATCH_SIZE, hidden_size * 2)
    concatenate_tensor = concatenate([query_tensor, document_tensor], axis=1)
    for i in range(common_dense_num):
        # shape: (BATCH_SIZE, hidden_size)
        concatenate_tensor = Dense(hidden_size * 2, activation=activation)(concatenate_tensor)
    prediction = Dense(1, activation='sigmoid', name="output")(concatenate_tensor)

    model = Model(inputs=[query_input, document_input], outputs=prediction)
    model.compile(
        Adam(),
        loss=my_cosine_proximity,
        metrics=[mean_positive_score, mean_negative_score, mean_positive_var, mean_negative_var, 'acc']
    )
    return model

class EvaluateCallback(Callback):
    def __init__(self,
                 model,
                 models_folder,
                 metrics_file,
                 train_generator,
                 test_generator,
                 validation_steps,
                 validation_batch_divider):
        self._model = model
        self._models_folder = models_folder
        self._metrics_file = metrics_file
        self._test_generator = test_generator
        self._train_generator = train_generator
        self._validation_steps = validation_steps
        self._validation_batch_divider = validation_batch_divider
        self._epoch = 0
        

        self.history = {}
        for name in self._model.metrics_names:
            self.history["train_" + name] = []
            self.history["test_" + name] = []
            
    def on_train_begin(self, logs):
        if os.path.exists(self._models_folder):
            shutil.rmtree(self._models_folder)
        os.mkdir(self._models_folder)
        if os.path.exists(self._metrics_file):
            os.remove(self._metrics_file)
        open(self._metrics_file, "w").close()

    def on_batch_end(self, batch, logs):
        if batch % self._validation_batch_divider == 0:
            test_evals = self._model.evaluate_generator(
                self._test_generator,
                steps=self._validation_steps
            )
            train_evals = self._model.evaluate_generator(
                self._train_generator,
                steps=self._validation_steps
            )
            for metric_name, metric in zip(self._model.metrics_names, train_evals):
                self.history["train_" + metric_name].append(metric)
            for metric_name, metric in zip(self._model.metrics_names, test_evals):
                self.history["test_" + metric_name].append(metric)
            short_model_name = "epoch_{}_batch_{}".format(
                self._epoch,
                batch)
            metrics_string = short_model_name + "_train_{}_test_{}".format(
                "_".join(map(str, train_evals)),
                "_".join(map(str, test_evals))
            )
            with open(self._metrics_file, "a") as handler:
                handler.write(metrics_string + "\n")
            
            if os.path.exists(self._models_folder):
                shutil.rmtree(self._models_folder)
            os.mkdir(self._models_folder)
            
            self._model.save_weights(os.path.join(self._models_folder, short_model_name))

    def on_epoch_end(self, epoch, logs):
        self._epoch += 1


def train_epoch(model, batches_to_sample_negatives, epoch):
    train_data_generator = make_generator(DataGenerator(
        "positive_train_data.tsv",
        "negative_train_data.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives
    ))

    val_data_generator = make_generator(DataGenerator(
        "positive_train_data.tsv",
        "negative_train_data.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives
    ))

    test_data_generator = make_generator(DataGenerator(
        "positive_train_data_10K.tsv",
        "negative_train_data_10K.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives
    ))
    сallback = EvaluateCallback(
         model,
         "models_epoch_{}".format(epoch),
         "metrics_{}".format(epoch),
         val_data_generator,
         test_data_generator,
         int(TEST_DATA_SIZE / BATCH_SIZE),
         int(TEST_DATA_SIZE / BATCH_SIZE) * 10
    )
    x = next(train_data_generator)
    y = next(val_data_generator)
    z = next(test_data_generator)
    return model.fit_generator(
        train_data_generator,
        steps_per_epoch=int(TRAIN_DATA_SIZE / BATCH_SIZE),
        epochs=1,
        verbose=1,
        initial_epoch=0,
        callbacks=[сallback]
    )