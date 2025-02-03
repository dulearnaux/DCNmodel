import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import subprocess
import tensorflow as tf
import pandas as pd
from collections import OrderedDict

df = pd.read_csv('data/Criteo_x1/train.csv', nrows=1)
cols = df.columns.values.tolist()
numeric_cols = [col for col in cols if col.startswith('I')]


def concat_numeric(elem: OrderedDict[str:tf.Tensor], label: tf.Tensor):
    # Create the concatenated single numeric input feature
    nums = tf.concat(values=[elem[col] for col in numeric_cols], axis=-1,
                     name='numeric_inputs')
    nums = tf.reshape(nums, (1, -1))
    elem.update({'numeric_inputs': nums})
    elem.move_to_end('numeric_inputs', last=False)
    # delete individual numeric features
    for col in numeric_cols:
        del elem[col]
    return elem, label


def shuffle_csv(file_in: str):
    os.rename(file_in, 'tmp.csv')
    subprocess.run(
        f'( head -n 2 tmp.csv ; tail -n +2 tmp.csv|shuf ) > {file_in}',
        shell=True)


if __name__ == '__main__':
    for file, shards in zip(['sample', 'test', 'valid', 'train'],
                            [2, 10, 20, 80]):
        def shard_func(features, label):
            # choose to split shards using C2 since it seems the most evenly
            # distributed and is an int.
            _ = label
            return tf.cast(features['C2'][0] % shards, tf.int64)

        read_path = f'data/Criteo_x1/{file}.csv'
        save_path = f'data/tfdata/{file}.data'

        # At start of each epoch, batch loss jumps a lot. We need to randomize
        # across the entire epoch, which is best done offline.
        print(f'Shuffling file {read_path}.')
        shuffle_csv(read_path)
        print(f'Shuffling complete')

        data_ts = tf.data.experimental.make_csv_dataset(
            read_path, 1, label_name='label', num_epochs=1)
        print(f'concatenating inputs on dataset')
        new_data = data_ts.map(
            concat_numeric, num_parallel_calls=tf.data.AUTOTUNE)
        print(f'saving to {save_path=}')
        new_data.save(save_path, shard_func=shard_func)
