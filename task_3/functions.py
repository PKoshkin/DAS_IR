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
from keras.activations import sigmoid
import keras.backend as K
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback, LearningRateScheduler, Callback

BATCH_SIZE = 64
TEST_DATA_SIZE = 10000
TRAIN_DATA_SIZE = 490000
MAX_QUERY_LEN_HARD = 150

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


def make_hash_input_from_lines(lines, dict_size):
    query_indices_words = [], []
    document_indices_words = [], []

    query_indices_words_bigrams = [], []
    document_indices_words_bigrams = [], []

    query_indices_symbol_trigrams = [], []
    document_indices_symbol_trigrams = [], []
    for i, line in enumerate(lines):
        query_words, query_words_bigrams, query_symbol_trigrams, document_words, document_words_bigrams, document_symbol_trigrams = line.split("\t")
        query_words = map(int, query_words.split())
        document_words = map(int, document_words.split())
        query_words_bigrams = map(int, query_words_bigrams.split())
        document_words_bigrams = map(int, document_words_bigrams.split())
        query_symbol_trigrams = map(int, query_symbol_trigrams.split())
        document_symbol_trigrams = map(int, document_symbol_trigrams.split())

        for word in query_words:
            query_indices_words[0].append(i)
            query_indices_words[1].append(word)
        for word in document_words:
            document_indices_words[0].append(i)
            document_indices_words[1].append(word)

        for word in query_words_bigrams:
            query_indices_words_bigrams[0].append(i)
            query_indices_words_bigrams[1].append(word)
        for word in document_words_bigrams:
            document_indices_words_bigrams[0].append(i)
            document_indices_words_bigrams[1].append(word)

        for word in query_symbol_trigrams:
            query_indices_symbol_trigrams[0].append(i)
            query_indices_symbol_trigrams[1].append(word)
        for word in document_symbol_trigrams:
            document_indices_symbol_trigrams[0].append(i)
            document_indices_symbol_trigrams[1].append(word)

    query_words_batch = csr_matrix(
        (np.ones(len(query_indices_words[0])), query_indices_words),
        shape=(len(lines), dict_size)
    )
    document_words_batch = csr_matrix(
        (np.ones(len(document_indices_words[0])), document_indices_words),
        shape=(len(lines), dict_size)
    )

    query_words_bigrams_batch = csr_matrix(
        (np.ones(len(query_indices_words_bigrams[0])), query_indices_words_bigrams),
        shape=(len(lines), dict_size)
    )
    document_words_bigrams_batch = csr_matrix(
        (np.ones(len(document_indices_words_bigrams[0])), document_indices_words_bigrams),
        shape=(len(lines), dict_size)
    )

    query_symbol_trigrams_batch = csr_matrix(
        (np.ones(len(query_indices_symbol_trigrams[0])), query_indices_symbol_trigrams),
        shape=(len(lines), dict_size)
    )
    document_symbol_trigrams_batch = csr_matrix(
        (np.ones(len(document_indices_symbol_trigrams[0])), document_indices_symbol_trigrams),
        shape=(len(lines), dict_size)
    )

    return {
        'query_words': query_words_batch,
        'document_words': document_words_batch,
        'query_words_bigrams': query_words_bigrams_batch,
        'document_words_bigrams': document_words_bigrams_batch,
        'query_symbol_trigrams': query_symbol_trigrams_batch,
        'document_symbol_trigrams': document_symbol_trigrams_batch
    }

def make_lstm_dense_input_from_lines(lines, max_query_len, document_dict_size):
    document_indices_axis_0, document_indices_axis_1, document_values = [], [], []
    query_batch = np.zeros([len(lines), MAX_QUERY_LEN_HARD])
    for i, line in enumerate(lines):
        query, document = line.split("\t")
        query_words = list(map(int, query.split()))[:MAX_QUERY_LEN_HARD]
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

def logloss(y_true, y_pred):
    labels = (y_true + 1) / 2
    probs = (y_pred + 1) / 2
    return -K.mean(labels * K.log(y_pred) + (1 - labels) * K.log(1 - y_pred))

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
    return K.sum(y_pred * get_positive_mask(y_true)) / positives_num

def mean_positive_var(y_true, y_pred):
    mean_positive = mean_positive_score(y_true, y_pred)
    positives_num = K.sum(get_positive_mask(y_true))
    return K.sum((y_pred * get_positive_mask(y_true) - mean_positive) ** 2) / positives_num

def mean_negative_score(y_true, y_pred):
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.sum(y_pred * get_negative_mask(y_true)) / negatives_num

def mean_negative_var(y_true, y_pred):
    mean_negative = mean_negative_score(y_true, y_pred)
    negatives_num = K.sum(get_negative_mask(y_true))
    return K.sum((y_pred * get_negative_mask(y_true) - mean_negative) ** 2) / negatives_num

def normalize(embedding):
    return K.l2_normalize(embedding, axis=-1)

def dot_product(embeddings):
    return K.sum(embeddings[0] * embeddings[1], axis=-1)

def proba_score(embeddings):
    return K.sigmoid(dot_product(embeddings))

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
    return tensor  # shape: (BATCH_SIZE, hidden_size)

def make_compiled_model(inputs, embedding_1, embedding_2):
    embedding_1 = Lambda(normalize)(embedding_1)  # shape: (BATCH_SIZE, hidden_size)
    embedding_2 = Lambda(normalize)(embedding_2)  # shape: (BATCH_SIZE, hidden_size)
    score = Lambda(dot_product)([embedding_1, embedding_2])
    prediction = Lambda(reshape_to_prediction, name="output")(score)

    model = Model(inputs=inputs, outputs=prediction)
    model.compile(
        Adam(),
        loss=logloss,
        metrics=[mean_positive_score, mean_negative_score, mean_positive_var, mean_negative_var, 'acc', 'mse']
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

    return make_compiled_model([query_input, document_input], query_embedding, document_embedding)
    

def make_hash_model(dict_size,
                    dense_num,
                    activation,
                    hidden_size):
    # shape: (BATCH_SIZE, dict_size)
    query_words_input = Input(shape=(dict_size,), sparse=True, name="query_words")
    query_words_embedding = make_dense_branch(query_words_input, dense_num, hidden_size, dict_size, activation)
    # shape: (BATCH_SIZE, hidden_size)
    document_words_input = Input(shape=(dict_size,), sparse=True, name="document_words")
    document_words_embedding = make_dense_branch(document_words_input, dense_num, hidden_size, dict_size, activation)

    # shape: (BATCH_SIZE, dict_size)
    query_words_bigrams_input = Input(shape=(dict_size,), sparse=True, name="query_words_bigrams")
    query_words_bigrams_embedding = make_dense_branch(query_words_bigrams_input, dense_num, hidden_size, dict_size, activation)
    # shape: (BATCH_SIZE, hidden_size)
    document_words_bigrams_input = Input(shape=(dict_size,), sparse=True, name="document_words_bigrams")
    document_words_bigrams_embedding = make_dense_branch(document_words_bigrams_input, dense_num, hidden_size, dict_size, activation)

    # shape: (BATCH_SIZE, dict_size)
    query_symbol_trigrams_input = Input(shape=(dict_size,), sparse=True, name="query_symbol_trigrams")
    query_symbol_trigrams_embedding = make_dense_branch(query_symbol_trigrams_input, dense_num, hidden_size, dict_size, activation)
    # shape: (BATCH_SIZE, hidden_size)
    document_symbol_trigrams_input = Input(shape=(dict_size,), sparse=True, name="document_symbol_trigrams")
    document_symbol_trigrams_embedding = make_dense_branch(document_symbol_trigrams_input, dense_num, hidden_size, dict_size, activation)

    query_embedding = concatenate([query_symbol_trigrams_embedding, query_words_bigrams_embedding, query_words_embedding], axis=1)
    document_embedding = concatenate([document_symbol_trigrams_embedding, document_words_bigrams_embedding, document_words_embedding], axis=1)
    
    return make_compiled_model(
        [
            query_words_input,
            document_words_input,
            query_words_bigrams_input,
            document_words_bigrams_input,
            query_symbol_trigrams_input,
            document_symbol_trigrams_input,
        ],
        query_embedding,
        document_embedding
    )

def make_lstm_dense_model(document_dict_size,
                     document_dense_num,
                     query_dict_size,
                     query_lstm_num,
                     query_dense_num,
                     max_query_len,
                     activation,
                     hidden_size):
    # shape: (BATCH_SIZE, QUERY_DICT_SIZE)
    query_input = Input(shape=(MAX_QUERY_LEN_HARD,), name="query_input")
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

    return make_compiled_model([query_input, document_input], query_embedding, document_embedding)


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


def train_epoch(model, batches_to_sample_negatives, epoch, positive_file, negative_file, input_from_lines_function):
    train_data_generator = make_generator(DataGenerator(
        positive_file,
        negative_file,
        model,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives,
        input_from_lines_function
    ))

    val_data_generator = make_generator(DataGenerator(
        positive_file,
        negative_file,
        model,
        BATCH_SIZE,
        BATCH_SIZE,
        BATCH_SIZE * batches_to_sample_negatives,
        input_from_lines_function
    ))

    test_data_generator = make_generator(DataGenerator(
        positive_file,
        negative_file,
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
