from typing import List
import pickle
import os
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, sigmoid

INTS = tf.int32
FLOATS = tf.float32

# TODO add UnitNorm constraint to embedding,
#  https://www.tensorflow.org/api_docs/python/tf/keras/constraints/UnitNorm


class TensorboardOnNthBatch(tf.keras.callbacks.TensorBoard):
    """Same as Tensorboard callback, except logs validation and distribution
        metrics every `log_every_n_steps`."""

    def __init__(self, val_data, log_every_n_steps=1, histogram_freq=1,
                 embedding_freq=1, *args, **kwargs):
        # Force Tensorboard to not write distributions on epoch end. Write every
        # n steps instead if they are passed to __init__. If you write
        # distributions on epoch end as well, it will delete the intra-epoch
        # distributions.
        super().__init__(histogram_freq=0, embeddings_freq=0, *args, **kwargs)
        self.N = log_every_n_steps
        self.val_eval_count = 0
        self.step = 0
        self.val_data = val_data
        # Cannot call these 'histogram_freq' to not override TensorBoard class.
        self.histogram_freq_n_step = histogram_freq
        self.embedding_freq_n_step = embedding_freq

    def on_train_batch_end(self, batch, logs=None):
        self.step += 1

        # Code copied from original TensorBoard.on_train_batch_end
        if self._should_write_train_graph:
            self._write_keras_model_train_graph()
            self._should_write_train_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._batch_start_time
            self.summary.scalar(
                "batch_steps_per_second",
                1.0 / batch_run_time,
                step=self._train_step,
            )

        # Every N steps, generate validation metrics, and also for first step.
        if self.step % self.N == 0 or self.step == 1:
            self.val_eval_count =+ 1
            val_logs = self.model.evaluate(
                x=self.val_data, return_dict=True, verbose=0)
            val_logs = {"val_" + name: val for name, val in val_logs.items()}
            logs.update(val_logs)

            # From tensorboard source code.
            if self.histogram_freq_n_step:
                self._log_weights(self.step)
            if self.embedding_freq_n_step:
                self._log_embeddings(self.step)

        # More code copied from TensorBoard source code.
        # `logs` isn't necessarily always a dict
        if isinstance(logs, dict):
            for name, value in logs.items():
                self.summary.scalar(
                    "batch_" + name, value, step=self.step
                )
        if not self._should_trace:
            return
        if self._is_tracing and self._global_train_batch >= self._stop_batch:
            self._stop_trace()

class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1],), initializer="random_normal",
            trainable=True, dtype=tf.float32)
        self.b = self.add_weight(
            shape=(input_shape[0][-1],), initializer="random_normal",
            trainable=True, dtype=tf.float32)

    def call(self, inputs):
        x_0 = inputs[0]
        x_l = inputs[1]
        cross_term = tf.matmul(
            tf.matmul(x_l, x_0, transpose_a=True), tf.expand_dims(self.w, 1))
        return tf.transpose(cross_term, (0, 2, 1)) + self.b + x_0


class CrossLayerV2(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w = None
        self.b = None

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1],input_shape[0][-1]),
            initializer="random_normal", trainable=True, dtype=tf.float32)
        self.b = self.add_weight(
            shape=(input_shape[0][-1],), initializer="random_normal",
            trainable=True, dtype=tf.float32)

    def call(self, inputs):
        x_0 = inputs[0]
        x_l = inputs[1]
        return layers.Multiply()([x_0, tf.matmul(x_l, self.w) + self.b]) + x_l


class CrossLayerBlock(tf.keras.Model):
    def __init__(self, n_layers: int = 6, use_v2: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.cross_layers = []
        self.use_v2 = use_v2

    def build(self):
        for i in range(self.n_layers):
            if self.use_v2:
                self.cross_layers.append(CrossLayerV2(name=f'cross_v2_{i}'))
            else:
                self.cross_layers.append(CrossLayer(name=f'cross{i}'))

    def call(self, inputs):
        x = inputs
        for layer in self.cross_layers:
            x = layer((inputs, x))
        return x


class MLPBlock(tf.keras.Model):
    """Block of (dense, activation, batchNorm, dropout) layers.

    Dropout layers not applied when drop_rate==0 (default).
    """
    def __init__(
            self, n_layers: int = 5, units: int = 1024, drop_rate: float = 0,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.units = units
        self.drop_rate = drop_rate
        self.dense_layers = []
        self.batch_norm_layers = []
        self.drop_layers = []
        for i in range(self.n_layers):
            self.dense_layers.append(layers.Dense(
                self.units, activation=relu, name=f'dense{i}', use_bias=False))
            self.batch_norm_layers.append(layers.BatchNormalization(
                name=f'BN{i}'))
            # Note, if drop_rate == 0, this won't get used.
            self.drop_layers.append(layers.Dropout(
                rate=self.drop_rate, name=f'drop{i}'))

    def call(self, inputs):
        x = inputs
        for dense, batch_norm, dropout in zip(
                self.dense_layers, self.batch_norm_layers, self.drop_layers):
            x = dense(x)
            x = batch_norm(x)
            if self.drop_rate > 0:
                x = dropout(x)
        return x


class DeepCrossNetwork:
    """Creates the Deep Cross Network in the 2017 paper by Wang et al.

    See https://arxiv.org/pdf/1708.05123
    """
    def __init__(
            self, vocab_file: str = 'data/vocab/vocab_all_data_thresh10.pkl',
            save_path='model/my_model'
    ):
        """Creates the max vocab size for each categorical feature."""
        self.save_path = save_path
        self.vocab_file = vocab_file
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        with open(vocab_file, 'rb') as fp:
            self.vocabs = pickle.load(fp)

        self.embed_dims = {}
        for col, vocab in self.vocabs.items():
            self.embed_dims[col] = 6 * int(len(vocab) ** 0.25)

        self.numeric_cols = [f'I{i + 1}' for i in range(13)]
        self.categorical_cols = [*self.vocabs]  # dict.keys() prevents pickle.
        self.label_col = 'label'

        self.submodels = {}

        self.model = None
        self.history = None
        self.input_layer = []
        self.output_layer = []

    def clear_model(self):
        """Required step if creating model more than once."""
        self.model = None
        self.history = None
        self.input_layer = []
        self.output_layer = []
        self.submodels = {}

    def create_feature_embedding(self, feat_name: str):
        """Creates graph from input to embedding for categorical features."""
        input_layer = layers.Input(shape=(1,), name=feat_name)
        # this is needed when specifying model.
        self.input_layer.append(input_layer)
        encode_layer = layers.IntegerLookup(
            vocabulary=self.vocabs[feat_name], num_oov_indices=1,
            output_mode='int', name=f'idx_{feat_name}', dtype=INTS)
        embed_layer = layers.Embedding(
            input_dim=len(self.vocabs[feat_name]) + 1,
            output_dim=self.embed_dims[feat_name],
            name=f'emb_{feat_name}')

        # connect the layers
        x = encode_layer(input_layer)  # first input is integer_inputs
        x = embed_layer(x)
        return x

    def create_mlp_model(self, dense_layers: int = 5, units: int = 1024,
                         dropout_rate: float = 0.5, plot_file: str = None):
        """Vanilla MLP. 5 dense layers, size of 1024."""
        self.clear_model()

        # create feature inputs, numeric and categorical
        numeric_inputs = layers.Input(
            shape=(1, len(self.numeric_cols),), name='numeric_inputs',
            dtype=INTS)
        self.input_layer.append(numeric_inputs)

        # Create embeddings for categorical features
        cat_embeddings = []
        for col in self.categorical_cols:
            cat_embeddings.append(self.create_feature_embedding(col))
        # numeric features go first, then all cat features
        cat_embeddings.insert(0, numeric_inputs)
        feature_stack_layer = layers.Concatenate(name='concat_layer')(
            cat_embeddings)

        # Create MLP block
        self.submodels['mlp_block'] = MLPBlock(
            n_layers=dense_layers, units=units, drop_rate=dropout_rate,
            name=f'mlp_block_n_{dense_layers}'
        )
        mlp_layer_output = self.submodels['mlp_block'](feature_stack_layer)
        # output layer
        output_layer = layers.Dense(
            units=1, activation=sigmoid, name='output_layer')(mlp_layer_output)

        self.output_layer = output_layer
        self.model = tf.keras.Model(
            inputs=self.input_layer, outputs=output_layer, name='MLP_model')

        if plot_file:
            tf.keras.utils.plot_model(
                self.model, to_file=plot_file, show_shapes=True,
                expand_nested=True, show_layer_names=True)

    def create_dcn_model(
            self, cross_layers: int = 6, dense_layers: int = 2,
            units: int = 1024, dropout_rate: float = 0.5, use_v2: bool = True,
            plot_file: str = None):
        """Cross network in parallel with MLP"""
        self.clear_model()
        # Numeric and embedding concatenation feeds into both cross and MLP
        # streams
        numeric_inputs = layers.Input(
            shape=(1, len(self.numeric_cols),), name='numeric_inputs',
            dtype=INTS)
        self.input_layer.append(numeric_inputs)

        cat_embeddings = []
        for col in self.categorical_cols:
            cat_embeddings.append(self.create_feature_embedding(col))

        # numeric features go first, then all cat features
        cat_embeddings.insert(0, numeric_inputs)
        feature_stack_layer = layers.Concatenate(name='concat_layer')(
            cat_embeddings)

        # Create MLP block
        self.submodels['mlp_block'] = MLPBlock(
            n_layers=dense_layers, units=units, drop_rate=dropout_rate,
            name=f'mlp_block_n_{dense_layers}')
        mlp_layer_output = self.submodels['mlp_block'](feature_stack_layer)

        # cross network block
        self.submodels['cross_block'] = CrossLayerBlock(
            n_layers=cross_layers, use_v2=use_v2,
            name=f'cross_layer_block_n_{cross_layers}')
        cross_layer_output = self.submodels['cross_block'](feature_stack_layer)
        concat2 = layers.Concatenate(name='concat_layer2')(
            [cross_layer_output, mlp_layer_output])

        # output layer
        output_layer = layers.Dense(
            units=1, activation=None, name='output_layer')(concat2)
        # Using float16 on output layer can load to instability. Need to ensure
        # output layer is float32, even if upstream variables are not.
        output_layer = layers.Activation(
            'sigmoid', dtype=tf.float32)(output_layer)
        self.output_layer = output_layer
        self.model = tf.keras.Model(
            inputs=self.input_layer, outputs=output_layer, name='DCN_model')

        if plot_file:
            if not os.path.exists(os.path.dirname(plot_file)):
                os.makedirs(os.path.dirname(plot_file))
            tf.keras.utils.plot_model(
                self.model, to_file=plot_file, show_shapes=True,
                expand_nested=True, show_layer_names=True)

    def compile_model(
            self, learning_rate: float = 0.001, clipnorm: float = 1.0):
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=learning_rate, clipnorm=clipnorm),
            metrics=[tf.keras.metrics.AUC(),
                     tf.keras.metrics.BinaryAccuracy(threshold=0.25)],
        )

    def save_model(self):
        # saves the model separately from the DeepCrossNetwork object.
        keras_path = self.save_path+'/model.keras'
        if os.path.exists(keras_path):
            print(f'overwriting: {keras_path}')
        else:
            print(f'saving tf.keras.model to {keras_path}')
        self.model.save(keras_path)

    def load_model(self):
        # saves the model separately from the DeepCrossNetwork object.
        keras_path = self.save_path+'/model.keras'
        if not os.path.exists(keras_path):
            print(f'no file found at: {keras_path}')
            print(f'Cannot load model.')
        else:
            self.model = tf.keras.models.load_model(keras_path)

    def save_history(self):
        csv_path = self.save_path + '/history.csv'
        hist_df = pd.DataFrame(self.history.history)
        if os.path.exists(csv_path):
            print(f'overwriting: {csv_path}')
        else:
            print(f'saving tf.keras.model to {csv_path}')

        with open(csv_path, mode='w') as f:
            hist_df.to_csv(f, index_label='epoch')


def create_input_data(
        df: pd.DataFrame,
        model: DeepCrossNetwork,
        label: str = 'train') -> (List[tf.Tensor], tf.Tensor):
    """Creates tf input data from pandas DF"""
    target_df = df.pop('label')
    target = tf.expand_dims(tf.convert_to_tensor(
        target_df, name=f'label_{label}', dtype=INTS), 1)
    # 1 input for integer features, 26 inputs for categorical features
    input_data = [tf.expand_dims(tf.convert_to_tensor(
        df[model.numeric_cols], dtype=FLOATS, name=f'numeric_{label}'), 1)]

    for col in model.categorical_cols:
        input_data.append(
            tf.convert_to_tensor(df[col], name=f'{col}_{label}',
                                 dtype=INTS))
    return input_data, target
