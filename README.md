# Mol2DNMR
Official code and data of paper: Mol2DNMR: An Annotated Experimental Dataset for Advancing Molecular Representation Learning in 2D NMR Analysis


## Requirements and Installation
### 1. Create virtual environment
```
conda create -n orientationfinder python=3.9 
conda activate orientationfinder
```

### 2. Install pytorch with CUDA 11.7
```
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117 
pip install pytorch_lightning 
pip install pandas 
pip install nmrglue 
pip install transformers
```
## Train and Evaluation 
