import shutil
import os
import pickle

import tensorflow as tf

from DCNmodel import DeepCrossNetwork

if __name__ == '__main__':

    # DCN = DeepCrossNetwork(
    #     vocab_file='data/vocab/vocab_all_data_thresh0.pkl',
    #     save_path='model/MLP_vocab00')
    # DCN.create_mlp_model(dense_layers=5, units=1024, dropout_rate=0.5,
    #                      plot_file='./docs/images/MLP_vocab00.png')
    DCN = DeepCrossNetwork(
        vocab_file='data/vocab/vocab_all_data_thresh0.pkl',
        save_path='model/DCN_vocab00')
    DCN.create_dcn_model(
        cross_layers=6, dense_layers=2, units=1024, dropout_rate=0.5,
        plot_file='./docs/images/DCN_vocab00.png')

    # create input data
    valid = tf.data.Dataset.load('data/tfdata/valid.data')
    train = tf.data.Dataset.load('data/tfdata/sample.data')

    logdir = f'./logs/tensorboard/{DCN.save_path}'
    print(f'make sure you started tensorboard in a separate process. E.g. in'
          f'shell')
    print(f'tensorboard --logdir={logdir}')
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.mkdir(logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, update_freq=3000,
        embeddings_freq=1)
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

    print(f'Training model')
    BATCH_SIZE = 128
    # LR is halfway between SQRT and linear increase.
    LR = 0.001*((BATCH_SIZE/128)**0.75)
    DCN.compile_model(learning_rate=LR)
    DCN.model.summary()
    DCN.history = DCN.model.fit(
        x=train.take(10_000).batch(BATCH_SIZE),
        epochs=10,
        callbacks=[tb_callback, checkpoint],
        validation_data=train.batch(BATCH_SIZE),
        validation_steps=100_000//BATCH_SIZE
    )

    # save tf.keras.Model object
    DCN.save_model()
    # save the rest of the DeepCrossNetwork custom object.
    with open(DCN.save_path+'/model.pkl', 'wb') as fp:
        pickle.dump(DCN, fp)

    # load pickle, then load model into the DeepCrossNetwork object.
    # with open('model/DCN_vocab00/model.pkl', 'rb') as fp:
    #     NEW = pickle.load(fp)
    #
    # NEW.model = tf.keras.models.load_model(
    #     NEW.save_path+'/checkpoint.model.04.keras')
    # NEW.model.summary()
    # NEW.model
