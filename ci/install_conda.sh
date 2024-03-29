set -xe

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --name=${ENV_NAME}  python=$PYTHON --quiet
conda env update -n ${ENV_NAME} -f ci/${ENV_NAME}.yaml

source ${HOME}/miniconda/etc/profile.d/conda.sh

conda activate ${ENV_NAME}

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/miniconda/lib

conda list -n ${ENV_NAME}
# check that the python version matches the desired one; exit immediately if not
PYVER=`python -c "from __future__ import print_function; import sys; print('{:d}.{:d}'.format(sys.version_info.major, sys.version_info.minor))"`
if [[ $PYVER != $PYTHON ]]; then
  exit 1;
fi
