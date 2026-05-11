# Mesh-based Human Head Avatar Reconstruction

> Reconstruct a human head avatar from multiview photos using mesh-based methods.

---

## Environment Setup

```bash
conda env create -f environment.yml
pip install git+https://github.com/NVlabs/nvdiffrast
git clone --recursive https://github.com/mikiisayakaa/cholespy_multiGPU.git
pip install cholespy_multiGPU
pip install ./gsface-shader
```

---

## Training and Testing

```bash
sh batch_train.sh
sh batch_test.sh
```

---

## Data

- Dataset from GaussianAvatars