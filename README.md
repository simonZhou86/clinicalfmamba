# ClinicalFMamba
PyTorch Implementation of ClinicalFMamba: Advancing Clinical Assessment using Mamba-based Multimodal Neuroimaging Fusion, MICCAI MLMI 2025

Arxiv Link: [here](https://arxiv.org/abs/2508.03008)

## Setup (Run on Google Colab)
### Install Mamba
```bash
cd causal-conv1d
python setup.py install

cd ../mamba
python setup.py install
```
### Other useful dependencies
```bash
pip install fvcore
pip install einops
pip install torchmetrics
pip install pyfftw
pip install phasepack
```
Test if Mamba have installed successfully
```bash
python ./causal-conv1d/tests/test_causal_conv1d.py
```

## Sample Training pipeline
```bash
python ./my_train.py --dataset "MRI-SPECT" --lr 0.0005 --epoch 200 --exp_folder_name "dep5_epo200_exp"
```

## Sample Inference pipeline
```bash
python validation.py --model_pt ./model/dep5_epo200_exp.pt --test_folder ./my_data --folder_name "dep5_epo200_exp" --exp 0
```
