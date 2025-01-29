#!/bin/bash

# Handy script when training on virtual machines.

sudo apt update

# install pipx and poetry. Used to manage python dependencies and virtual env.
sudo apt install pipx
pipx ensurepath
sudo pipx ensurepath --global
pipx install poetry

# Graphviz is required to render the neural network DAG plots.
sudo apt-get install graphviz

# Run from same directory as `poetry.lock` file. This installs all python deps.
poetry install
# poetry install --without nvidia

# Download CSV data and unzip. About 3Gb zipped, 11Gb unzipped.
mkdir ./data
wget -q -O ./data/tmp.zip https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip --show-progress && unzip ./data/tmp.zip -d ./data/Criteo_x1 && rm ./data/tmp.zip
# data/Criteo_x1/
#  |---train.csv  8.2Gb
#  |---valid.csv  2.1Gb
#  |---test.csv   1.1Gb


# Generate some plots, vocabulary files, and tensorflow dataset files. The ts
# data set files allow for lazy reading of batches as needed to reduce memory
# use. They use about 4x HDD space. So around 40Gb.
poetry run time python preprocessing.py
poetry run time python create_tf_datasets.py
# data/Criteo_x1/
#  |---train.csv  8.2Gb
#  |---valid.csv  2.1Gb
#  |---test.csv   1.1Gb
# data/tfdata/   # 41Gb total
#  |---sample.data/
#  |---train.data/
#  |---valid.data/
#  |---test.data/
# docs/images   # These are committed to git already, but will be overwritten.
#  |---cardinality_bar_chart.png
#  |---embed_dims_bar_chart.png
#  |---hist_cat_vars.png
#  |---hist_int_vars.png
