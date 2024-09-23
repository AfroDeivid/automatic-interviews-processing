# Meditation-Interviews

conda create --name NAME python=3.10
conda activate NAME

pip install ctranslate2

# CUDA
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# CPU only
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch

pip install git+https://github.com/m-bain/whisperx.git
pip install ipykernel 
pip install nvidia-<library>