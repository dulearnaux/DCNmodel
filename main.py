import shutil
import os
import pickle

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Mixed float16 used float32 on the variables, but float16 for compute in the
# GPU. This can lead to minor speed-ups due to lower memory requirements and
# major speed-ups, 3x, on some compatible NVIDIA GPUs.
# policy = tf.keras.mixed_precision.Policy('mixed_float16')
# tf.keras.mixed_precision.set_global_policy(policy)

from DCNmodel import DeepCrossNetwork

if __name__ == '__main__':

    # DCN = DeepCrossNetwork(
    #     vocab_file='data/vocab/vocab_all_data_thresh0.pkl',
    #     save_path='model/MLP_vocab00')
    # DCN.create_mlp_model(dense_layers=5, units=1024, dropout_rate=0.5,
    #                      plot_file='./docs/images/MLP_vocab00.png')
    DCN = DeepCrossNetwork(
        vocab_file='data/vocab/vocab_all_data_thresh100.pkl',
        save_path='model/DCN_shuffle')
    DCN.create_dcn_model(
        cross_layers=6, dense_layers=2, units=1024, dropout_rate=0.5,
        plot_file='./docs/images/DCN_shuffle.png')

    # create input data
    valid = tf.data.Dataset.load('data/tfdata/valid.data')
    train = tf.data.Dataset.load('data/tfdata/train.data')

    logdir = f'./logs/tensorboard/{DCN.save_path}'
    print(f'make sure you started tensorboard in a separate process. E.g. in'
          f'shell')
    print(f'tensorboard --logdir={logdir}')
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.mkdir(logdir)

    # Define callbacks
    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, update_freq=3000,
        embeddings_freq=1, write_graph=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{DCN.save_path}/checkpoint.model'+'.{epoch:02d}.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        initial_value_threshold=None
    )
    # Save metrics to CSV file for easy plotting.
    csv_logger = tf.keras.callbacks.CSVLogger(
        f'{DCN.save_path}/training_log_metrics.csv')
    print(f'Training model')
    BATCH_SIZE = 128
    # LR is halfway between SQRT and linear increase.
    LR = 0.001*((BATCH_SIZE/128)**0.75)
    DCN.compile_model(learning_rate=LR)
    DCN.model.summary()
    DCN.history = DCN.model.fit(
        x=train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        epochs=10,
        callbacks=[tb_callback, checkpoint, csv_logger],
        validation_data=valid.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE),
        validation_steps=100_000//BATCH_SIZE
    )

    DCN.save_model()  # save tf.keras.Model object
    DCN.save_history()  # save history.history as csv.
    # save the rest of the DeepCrossNetwork custom object.
    with open(DCN.save_path+'/model.pkl', 'wb') as fp:
        pickle.dump(DCN, fp)

    # # load pickle, then load model into the DeepCrossNetwork object.
    # with open('model/MLP_vocab00/model.pkl', 'rb') as fp:
    #     NEW = pickle.load(fp)
    #
    # NEW.model = tf.keras.models.load_model(
    #     NEW.save_path + '/checkpoint.model.04.keras')
    # NEW.model.summary()
    # NEW.save_history()

    # Create a plot of training history with 3 panels
    df = pd.read_csv(DCN.save_path + '/history.csv')
    metrics = [col for col in df.columns if
               not col.startswith('epoch') and not col.startswith('val_')]
    plt.figure(figsize=(15, 12))
    for i, metric in enumerate(metrics):
        plt.subplot(3, 1, i + 1)
        plt.plot(df['epoch'], df[metric], marker='o', label=f'{metric}')
        plt.plot(df['epoch'], df[f'val_{metric}'], marker='x',
                 label=f'val_{metric}')
        plt.title(f'{metric} vs epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.ylim(0, 1)  # Set y-axis scale between 0 and 1
        # Ensure x-axis labels show integers
        plt.xticks(df['epoch'])
        plt.grid(axis='x', which='both')  # Add grid only for x-axis
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(DCN.save_path + '/training_history_plot.png')
