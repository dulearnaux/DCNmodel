#!/bin/bash
# Handy script when training on virtual machines.
# This has been tested on lambdalabs

# Assumes you are running from the project root.
if [ ! -f 'poetry.lock' ]; then
    echo "Running from wrong directory. Run from project root!"
    exit 0
fi


# Some of the optional steps to run can be sped up by syncing processed data
# from persistent files storage, rather than processing from scratch. Approx
# run times are in the comments on the right.
while getopts pdn opt; do
   case $opt in
     p ) run_preprocessing=1                  ;;  # 60 min
     d ) run_create_tf_datasets=1             ;;  # 240 min
     n ) install_nvidia=1                     ;;  # pass flag if you have a GPU.
    \? ) echo "${0##*/} [ -erw ]" >&2; exit 1 ;;
  esac
done
shift $((OPTIND-1))
sudo apt update

# install pyenv to manage python versions
curl -fsSL https://pyenv.run | bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
echo 'eval "$(pyenv init - bash)"' >> ~/.profile
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile
eval "$(cat ~/.bashrc | tail -n +10)"
source ~./profile
source ~./bash_profile

# install python deps
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
# Install python 3.12
pyenv install 3.12
pyenv global 3.12

# install pipx and poetry. Used to manage python dependencies and virtual env.
sudo apt install -y pipx
pipx ensurepath
pipx install poetry

# Graphviz is required to render the neural network DAG plots.
sudo apt-get install -y graphviz

# Run from same directory as `poetry.lock` file. This installs all python deps.
if [[ "${install_nvidia}" -eq 1 ]] ; then
    poetry install
else
    poetry install --without nvidia
fi


if [[ "${run_preprocessing}" -eq 1 ]] ; then
    # Download CSV data and unzip. About 3Gb zipped, 11Gb unzipped.
    mkdir ./data
    wget -q -O ./data/tmp.zip https://huggingface.co/datasets/reczoo/Criteo_x1/resolve/main/Criteo_x1.zip --show-progress && unzip ./data/tmp.zip -d ./data/Criteo_x1 && rm ./data/tmp.zip
    # data/Criteo_x1/
    #  |---train.csv  8.2Gb
    #  |---valid.csv  2.1Gb
    #  |---test.csv   1.1Gb

    # Generate some plots, vocabulary files, and tensorflow dataset files.
    poetry run time python preprocessing.py
    # data/vocab
    #  |---vocab_all_data_thresh{0,10,100,1000,10000}.pkl
    #  |---vocab_all_data_thresh{0,10,100,1000,10000}.csv
    # docs/images   # These are committed to git already, but will be overwritten.
    #  |---cardinality_bar_chart.png
    #  |---embed_dims_bar_chart.png
    #  |---hist_cat_vars.png
    #  |---hist_int_vars.png
fi

if [[ "${run_create_tf_datasets}" -eq 1 ]] ; then
    # The ts data set files this creates allow for lazy reading of batches in
    # parallel, as needed to reduce memory use. They use about 4x HDD space.
    # So around 40Gb.
    poetry run time python create_tf_datasets.py
fi
# data/Criteo_x1/
#  |---train.csv  8.2Gb
#  |---valid.csv  2.1Gb
#  |---test.csv   1.1Gb
# data/tfdata/   # 41Gb total
#  |---sample.data/
#  |---train.data/
#  |---valid.data/
#  |---test.data/
