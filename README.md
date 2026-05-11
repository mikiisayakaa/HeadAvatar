A mesh-based human head avatar reconstruction from multiview photos

Environment setup:

conda env create -f environment.yml
pip install git+https://github.com/NVlabs/nvdiffrast
git clone --recursive https://github.com/mikiisayakaa/cholespy_multiGPU.git
pip install cholespy_multiGPU
pip install ./gsface-shader

Train and Test:
batch_train.sh
batch_test.sh

Data:
Dataset from GaussianAvatars