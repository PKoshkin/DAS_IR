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

BATCH_SIZE = 64
TEST_DATA_SIZE = 10000
TRAIN_DATA_SIZE = 490000
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
            
def make_dense_input_from_lines(lines, query_dict_size, document_dict_size):
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
        shape=(len(lines), query_dict_size)
    )
    docment_batch = csr_matrix(
        (document_values, (document_indices_axis_0, document_indices_axis_1)),
        shape=(len(lines), document_dict_size)
    )

    return {'query_input': query_batch, 'document_input': docment_batch}
    

def make_lstm_dense_input_from_lines(lines, max_query_len, document_dict_size):
    document_indices_axis_0, document_indices_axis_1, document_values = [], [], []
    query_batch = np.zeros([len(lines), max_query_len])
    for i, line in enumerate(lines):
        query, document = line.split("\t")
        query_words = list(map(int, query.split()))
        document_words = Counter(map(int, document.split()))
        
        query_batch[i, :len(query_words)] = query_words

        for word in document_words:
            document_indices_axis_0.append(i)
            document_indices_axis_1.append(word)
            document_values.append(document_words[word])

    docment_batch = csr_matrix(
        (document_values, (document_indices_axis_0, document_indices_axis_1)),
        shape=(len(lines), document_dict_size)
    )

    return {'query_input': query_batch, 'document_input': docment_batch}

class DataGenerator:
    def __init__(self, positive_filename, negative_filename, model, positives_num, negatives_num, negative_candidates_num, make_input_from_lines_function):
        self._positive_generator = cycle_file(positive_filename)
        self._negative_generator = cycle_file(negative_filename)
        self._model = model
        self._positives_num = positives_num
        self._negatives_num = negatives_num 
        self._negative_candidates_num = negative_candidates_num
        self._input_from_lines = make_input_from_lines_function

    def make_batch(self):
        negative_lines = [next(self._negative_generator) for _ in range(self._negative_candidates_num)]
        test_inputs = self._input_from_lines(negative_lines)
        predictions = -self._model.predict(test_inputs).reshape(-1)
        indices = np.argsort(predictions)[:self._negatives_num]
        negative_lines = list(np.array(negative_lines)[indices])

        positive_lines = [next(self._positive_generator) for _ in range(self._positives_num)]

        inputs = self._input_from_lines(positive_lines + negative_lines)
        labels = np.concatenate(
            [np.ones(len(positive_lines)), -1 * np.ones(len(negative_lines))],
            axis=0
        )

        return inputs, {'output': labels}
    
def make_generator(data_generator):
    while True:
        yield data_generator.make_batch()
        
def my_cosine_proximity(y_true, y_pred):
    mean_pred = K.mean(y_pred)
    return -K.mean((y_pred - mean_pred) * y_true)

def get_positive_mask(y_true):
    return (y_true + 1) / 2

def get_negative_mask(y_true):
    return (1 - y_true) / 2

def ranking_loss(y_true, y_pred):
    positive_prediction = mean_positive_score(y_true, y_pred)
    positives_num = K.sum(get_positive_mask(y_true))
    negatives_num = K.sum(get_negative_mask(y_true))
    
    return K.mean(y_pred - positive_prediction) * negatives_num / (positives_num + negatives_num)

def mean_positive_score(y_true, y_pred):
    positives_num = K.sum(get_positive_mask(y_true))
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.mean(y_pred * get_positive_mask(y_true)) * positives_num / (positives_num + negatives_num)

def mean_positive_var(y_true, y_pred):
    mean_positive = mean_positive_score(y_true, y_pred)
    positives_num = K.sum(get_positive_mask(y_true))
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.mean((y_pred * get_positive_mask(y_true) - mean_positive) ** 2) * positives_num / (positives_num + negatives_num)

def mean_negative_score(y_true, y_pred):
    positives_num = K.sum(get_positive_mask(y_true))
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.mean(y_pred * get_negative_mask(y_true)) * negatives_num / (positives_num + negatives_num)

def mean_negative_var(y_true, y_pred):
    mean_negative = mean_negative_score(y_true, y_pred)
    positives_num = K.sum(get_positive_mask(y_true))
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.mean((y_pred * get_negative_mask(y_true) - mean_negative) ** 2) * negatives_num / (positives_num + negatives_num)

def normalize(embedding):
    return K.l2_normalize(embedding, axis=-1)

def dot_product(embeddings):
    return K.sum(embeddings[0] * embeddings[1], axis=-1)

def reshape_to_prediction(score):
    return K.reshape(score, (-1, 1))

def loss(y_true, y_pred):
    return my_cosine_proximity(y_true, y_pred)
    
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
        loss=loss,
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
    

def make_lstm_dense_model(document_dict_size,
                     document_dense_num,
                     query_dict_size,
                     query_lstm_num,
                     query_dense_num,
                     max_query_len,
                     activation,
                     hidden_size):
    # shape: (BATCH_SIZE, QUERY_DICT_SIZE)
    query_input = Input(shape=(max_query_len,), name="query_input")
    query_embedding = make_lstm_branch(
        query_input,
        query_lstm_num,
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


def train_epoch(model, batches_to_sample_negatives, epoch, input_from_lines_function):
    train_data_generator = make_generator(DataGenerator(
        "freq_positive_train_data.tsv",
        "freq_negative_train_data.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives,
        input_from_lines_function
    ))

    val_data_generator = make_generator(DataGenerator(
        "freq_positive_train_data.tsv",
        "freq_negative_train_data.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives,
        input_from_lines_function
    ))

    test_data_generator = make_generator(DataGenerator(
        "freq_positive_train_data_10K.tsv",
        "freq_negative_train_data_10K.tsv",
        model,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives,
        input_from_lines_function
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
        callbacks=[сallback][:0]
    )
