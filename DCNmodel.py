from typing import List
import pickle
import os

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.activations import relu, sigmoid, tanh, linear

INTS = tf.int32
FLOATS = tf.float32


class CrossLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[0][-1],),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(input_shape[0][-1],), initializer="random_normal",
            trainable=True)

    def call(self, inputs):
        x_0 = inputs[0]
        x_l = inputs[1]
        cross_term = tf.matmul(
            tf.matmul(x_l, x_0, transpose_a=True), tf.expand_dims(self.w, 1))
        return tf.transpose(cross_term, (0, 2, 1)) + self.b + x_0


class CrossLayerBlock(tf.keras.Model):
    def __init__(self, n_layers: int = 6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.cross_layers = []

    def build(self):
        for i in range(self.n_layers):
            self.cross_layers.append(CrossLayer(name=f'cross{i}'))

    def call(self, inputs):
        x = inputs
        for layer in self.cross_layers:
            x = layer((inputs, x))
        return x


class MLPBlock(tf.keras.Model):
    def __init__(
            self, n_layers: int = 5, units: int = 1024, drop_rate: float = 0.5,
            *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_layers = n_layers
        self.units = units
        self.drop_rate = drop_rate
        self.dense_layers = []
        self.drop_layers = []
        for i in range(self.n_layers):
            self.dense_layers.append(layers.Dense(
                self.units, activation=relu, name=f'dense{i}'))
            self.drop_layers.append(layers.Dropout(
                rate=self.drop_rate, name=f'drop{i}'))

    def call(self, inputs):
        x = inputs
        for dense, dropout in zip(self.dense_layers, self.drop_layers):
            x = dense(x)
            x = dropout(x)
        return x


class CategoricalEmbeddingBlock(tf.keras.layers.Layer):

    def __init__(self, feat_name: str,
                 feat_vocab: List[int],
                 # feat_vocab: npt.NDArray[np.int_],
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feat_name = feat_name
        self.feat_vocab = feat_vocab
        self.embed_dim = int(6*len(feat_vocab)**0.25)
        # self.input_layer = None
        self.encode_layer = None
        self.embed_layer = None

    def build(self, input_shape):
        self.encode_layer = layers.IntegerLookup(
            vocabulary=self.feat_vocab, num_oov_indices=1,
            output_mode='int', name=f'idx_{self.feat_name}', dtype=INTS)
        self.embed_layer = layers.Embedding(
            input_dim=len(self.feat_vocab) + 1,
            output_dim=self.embed_dim,
            name=f'emb_{self.feat_name}')

    def call(self, inputs):
        x = self.encode_layer(inputs)
        x = self.embed_layer(x)
        return x

    # def get_config(self):
    #     config = super().get_config()
    #     config.update({
    #         'feat_name': self.feat_name,
    #         'feat_vocab': self.feat_vocab
    #     })
    #     return config


class DeepCrossNetwork:

    def __init__(
            self, vocab_file: str = 'data/vocab/vocab_all_data_thresh10.pkl',
            save_path='model/my_model'
    ):
        """Creates the max vocab size for each categorical feature."""
        self.save_path = save_path
        with open(vocab_file, 'rb') as fp:
            # Dict[List[int]]
            self.vocabs = pickle.load(fp)
        # convert from Dict[str: npt.NDArray[np.int_]] to # Dict[List[int]]
        if isinstance(list(self.vocabs.values())[0], np.ndarray):
            for col, vocab in self.vocabs.items():
                self.vocabs[col] = vocab.tolist()

        self.embed_dims = {}
        for col, vocab in self.vocabs.items():
            self.embed_dims[col] = 6 * int(len(vocab) ** 0.25)

        self.numeric_cols = [f'I{i + 1}' for i in range(13)]
        self.categorical_cols = [*self.vocabs]  # dict.keys() prevents pickle.
        self.label_col = 'label'

        self.submodels = {}

        self.model = None
        self.input_layer = []
        self.output_layer = []

    def clear_model(self):
        """Required step if creating model more than once."""
        self.model = None
        self.input_layer = []
        self.output_layer = []

    def create_mlp_model(self, dense_layers: int, units: int,
                         dropout_rate: float = 0.5, plot_file: str = None):
        """Vanilla MLP. 5 dense layers, size of 1024."""
        self.clear_model()

        # create feature inputs, numeric and categorical
        numeric_inputs = layers.Input(
            shape=(1, len(self.numeric_cols),), name='numeric_inputs',
            dtype=INTS)
        self.input_layer.append(numeric_inputs)
        for col in self.categorical_cols:
            self.input_layer.append(layers.Input(shape=(1,), name=col))
        feature_stack = [numeric_inputs]

        # Create embeddings for categorical features
        self.submodels['cat_features'] = {}
        for col, inputs in zip(self.categorical_cols, self.input_layer[1:]):
            self.submodels['cat_features'][col] = CategoricalEmbeddingBlock(
                feat_name=col, feat_vocab=self.vocabs[col])
            feature_stack.append(self.submodels['cat_features'][col](inputs))
        feature_stack_layer = layers.Concatenate(name='concat_layer')(
            feature_stack)

        # Create MLP block
        self.submodels['mlp_block'] = MLPBlock(
            n_layers=5, units=1024, drop_rate=0.5)
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
                show_layer_names=True)

    def create_DCN_model(
            self, cross_layers: int, dense_layers: int, units: int,
            dropout_rate: float = 0.5, plot_file: str = None):
        """Cross network in parallel with MLP"""
        self.clear_model()

        # create feature inputs, numeric and categorical
        numeric_inputs = layers.Input(
            shape=(1, len(self.numeric_cols),), name='numeric_inputs',
            dtype=INTS)
        self.input_layer.append(numeric_inputs)
        for col in self.categorical_cols:
            self.input_layer.append(layers.Input(shape=(1,), name=col))

        feature_stack = [numeric_inputs]
        # Create embeddings for categorical features
        self.submodels['cat_features'] = {}
        for col, inputs in zip(self.categorical_cols, self.input_layer[1:]):
            self.submodels['cat_features'][col] = CategoricalEmbeddingBlock(
                feat_name=col, feat_vocab=self.vocabs[col])
            feature_stack.append(self.submodels['cat_features'][col](inputs))
        feature_stack_layer = layers.Concatenate(name='concat_layer')(
            feature_stack)

        # Create MLP block
        self.submodels['mlp_block'] = MLPBlock(
            n_layers=5, units=1024, drop_rate=0.5)
        mlp_layer_output = self.submodels['mlp_block'](feature_stack_layer)

        # cross network block
        self.submodels['cross_block'] = CrossLayerBlock(
            n_layers=6, name=f'cross_layer_block_n_{cross_layers}')
        cross_block = self.submodels['cross_block'](feature_stack_layer)
        cross_layer_output = self.submodels['cross_block'](feature_stack_layer)
        concat2 = layers.Concatenate(name='concat_layer2')(
            [cross_layer_output, mlp_layer_output])

        # output layer
        output_layer = layers.Dense(
            units=1, activation=sigmoid, name='output_layer')(concat2)
        self.output_layer = output_layer
        self.model = tf.keras.Model(
            inputs=self.input_layer, outputs=output_layer, name='DCN_model')

        if plot_file:
            tf.keras.utils.plot_model(
                self.model, to_file=plot_file, show_shapes=True,
                show_layer_names=True)

    def compile_model(self):
        self.model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            metrics=[tf.keras.metrics.AUC(),
                     tf.keras.metrics.BinaryAccuracy(threshold=0.25)],
        )

    def save_model(self):
        # saves the model separately from the DeepCrossNetwork object.
        keras_path = self.save_path+'.keras'
        if os.path.exists(keras_path):
            print(f'overwriting: {keras_path}')
        else:
            print(f'saving tf.keras.model to {keras_path}')
        self.model.save(keras_path)

    def load_model(self):
        # saves the model separately from the DeepCrossNetwork object.
        keras_path = self.save_path+'.keras'
        if not os.path.exists(keras_path):
            print(f'no file found at: {keras_path}')
            print(f'Cannot load model.')
        else:
            self.model = tf.keras.models.load_model(keras_path)


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
