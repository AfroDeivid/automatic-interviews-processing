# Whisper Diarization

# Set-Up

## 1. Create Environment

```
conda create --name wd python=3.9 --yes
conda activate wd
```

## 2. Prerequisite Installations
 This section outlines the essential tools and libraries that need to be installed before proceeding with the main package installations.

- Download ``FFMPEG`` from [here](https://ffmpeg.org/download.html) or/& follow a guide like [this](https://phoenixnap.com/kb/ffmpeg-windows) for Windows installation.
Ensure that FFMPEG is added to your system’s PATH.

- ``Visual C++ Build Tools`` If you encounter build errors during installation, install the Visual C++ Build Tools by following this [guide](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools) or/& download directly from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

- Install ``Strawberry Perl`` for additional dependencies, [here](https://strawberryperl.com/).

## 3. Install Required Packages

After ensuring the prerequisites are set up, proceed with installing the necessary packages.

Run the following comand inside the *Whisper_Diarization* folder.

```
pip install cython
pip install ipykernel
pip install -r requirements.txt
pip uninstall huggingface_hub
pip install huggingface-hub==0.20.3
``` 

# How to use

```
python Whisper_Diariazation\diarize.py -a "..\data\OBE1\Id 15.m4a" --whisper-model large-v3 --language en
```

# Allow CUDA use (TO-DO)
```
conda create --name wdcuda python=3.9 --yes
conda activate wdcuda
pip install cython
pip uninstall torch --yes
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia --yes
pip install -r requirements.txt
pip uninstall huggingface_hub --yes
pip install huggingface-hub==0.20.3
```