# WhisperX

# Set-Up

## 1. Create Environment

`conda create --name NAME python=3.10`

`conda activate NAME`

## 2. CUDA or CPU

- If CUDA (obecuda)

`pip install ctranslate2`

`conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia`

- If CPU only (obe)

`conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch`

## 3. Packages

``pip install git+https://github.com/m-bain/whisperx.git``

``pip install ipykernel``
