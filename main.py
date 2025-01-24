import shutil
import os
import pickle

import pandas as pd
import tensorflow as tf

from DCNmodel import DeepCrossNetwork, create_input_data

if __name__ == '__main__':

    # DCN = DeepCrossNetwork(
    #     vocab_file='data/vocab/vocab_all_data_thresh0.pkl',
    #     save_path='model/MLP_vocab00')
    # DCN.create_mlp_model(dense_layers=5, units=1024, dropout_rate=0.5,
    #                      plot_file='./docs/images/MLPmodel.png')

    DCN = DeepCrossNetwork(
        vocab_file='data/vocab/vocab_all_data_thresh0.pkl',
        save_path='model/DCN_vocab00')
    DCN.create_DCN_model(
        cross_layers=6, dense_layers=2, units=1024, dropout_rate=0.5,
        plot_file='./docs/images/DCN_vocab00.png')
    DCN.compile_model()
    DCN.model.summary()

    # create input data
    train_df = pd.read_csv('data/Criteo_x1/train.csv', nrows=33_003_326 // 2)
    valid_df = pd.read_csv('data/Criteo_x1/valid.csv', nrows=33_003_326 // 20)

    input_train, target_train = create_input_data(train_df, DCN, 'train')
    input_valid, target_valid = create_input_data(valid_df, DCN, 'valid')

    logdir = f'./logs/tensorboard/{DCN.save_path}'
    # logdir = f'./logs/tensorboard/{time.strftime("%Y.%m.%d_%H.%M.%S", time.gmtime())}'
    if os.path.exists(logdir):
        shutil.rmtree(logdir)
        os.mkdir(logdir)

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1, update_freq=3000,
        embeddings_freq=1)
    print(f'Running model')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f'{DCN.save_path}/checkpoint.model.keras',
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        save_freq='epoch',
        initial_value_threshold=None
    )
    DCN.history = DCN.model.fit(
        x=input_train,
        y=target_train,
        batch_size=256,
        # batch_size=256,
        epochs=10,
        callbacks=[tb_callback, checkpoint],
        # validation_split=0.1,
        validation_data=(input_valid, target_valid)
    )
    DCN.save_model()
    with open(DCN.save_path+'.pkl', 'wb') as fp:
        pickle.dump(DCN, fp)

