import subprocess
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':

    num_lines = int(subprocess.check_output(
        "/usr/bin/wc -l data/Criteo_x1/train.csv", shell=True).split()[0])
    # Generate sample file.
    sample_size = 100_000
    subprocess.run(
        f'shuf -n {sample_size} data/Criteo_x1/train.csv > data/Criteo_x1/sample.csv',
        shell=True)
    df_head = pd.read_csv('data/Criteo_x1/train.csv', nrows=0)
    df = pd.read_csv('data/Criteo_x1/sample.csv',
                     header=None, names=df_head.columns.values)
    # shuf does not preserve the header row.
    df.to_csv('data/Criteo_x1/sample.csv', index=None)

    int_cols = df.columns[1:14]
    cat_cols = df.columns[14:(26+14)]
    # the [min, max] are already [0, 1]. So the data is normalized.1
    df[int_cols].describe().transpose()

    # Histograms of integer variables. They appear normalized.
    fig, axis = plt.subplots(3, 5, figsize=(12*1.5, 8*1.5))
    fig.delaxes(axis[2, 4])
    df.iloc[:, :14].hist(ax=axis.flatten()[:14], bins=50)
    fig.suptitle('Histogram of Integer Variables', fontsize=25)
    plt.savefig('docs/images/hist_int_vars.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    # Histograms of categorical variables. They have different ranges across
    # each feature
    fig, axis = plt.subplots(5, 6, figsize=(15*1.5, 12*1.5))
    fig.delaxes(axis[4, 5])
    fig.delaxes(axis[4, 4])
    fig.delaxes(axis[4, 3])
    fig.delaxes(axis[4, 2])
    df.iloc[:, 14:].hist(ax=axis.flatten()[:26], bins=50)
    fig.suptitle('Histogram of Categorical Variables', fontsize=25)
    plt.savefig('docs/images/hist_cat_vars.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    # Quite a large range in cardinality of the categorical variables
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    df[cat_cols].nunique().plot.bar()
    axis.bar_label(axis.containers[0], rotation=90)
    fig.suptitle('Cardinality of Categorical Variables', fontsize=25)
    plt.savefig('docs/images/cardinality_bar_chart.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    # Calculate embedding dims
    embed_dims = (6*df[cat_cols].nunique()**0.25).apply(int)
    embed_dims.to_csv('data/embed_dims.csv', header=['embed_dim'],
                      index_label='feature')
    fig, axis = plt.subplots(1, 1, figsize=(8, 8))
    embed_dims.plot.bar()
    axis.bar_label(axis.containers[0], rotation=90)
    fig.suptitle('Embedding Dimensions of Categorical Variables', fontsize=25)
    plt.savefig('docs/images/embed_dims_bar_chart.png', bbox_inches='tight')
    # plt.show()
    plt.close()

    # individual embedding matrix for each feature
    (embed_dims.apply(lambda x: int(6 * (x ** 0.25)))*embed_dims).sum()
    # single embedding for all features combined
    int(int(6*(embed_dims.sum()**0.25))*embed_dims.sum())

    # Save vocab sizes for various samples and thresholds
    df[cat_cols].nunique().to_csv('data/vocabs_sample.csv')
    # check against full file
    vocab_size = {}
    # col_dtypes = [np.float32]*len(int_cols) + [np.int16]*len(cat_cols)
    thresholds = [0, 10, 100, 1000, 10000]
    vocab = {}
    import time

    for col in cat_cols:  # skip labels
        # remove
        tic = time.time()
        df_col = pd.read_csv('data/Criteo_x1/train.csv', usecols=[col], header=0, engine="c", dtype=np.int64)
        counts = df_col[col].value_counts()
        for THRESHOLD in thresholds:
            values = counts[counts > THRESHOLD].index.values
            vocab_size[col] = len(np.unique(values))
            vocab[col] = np.unique(values)
            print(f'Column {col} cardinality = {vocab_size[col]}, t={time.time()/60 - tic/60: .4}')
            pd.Series(vocab_size).to_csv(f'data/vocab/vocabs_all_data_thresh{THRESHOLD}.csv')
            with open(f'data/vocab/vocab_all_data_thresh{THRESHOLD}.pkl', 'wb') as fp:
                pickle.dump(vocab, fp)
        print(f'saved data for {col=}')
