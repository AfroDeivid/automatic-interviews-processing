# Automatic Interviews processing

This repository provides a scalable and automated pipeline for transcribing and diarizing audio interviews, tailored to handle real-world challenges such as noisy recordings, overlapping speakers, and multi-language scenarios. It leverages cutting-edge open-source tools, including **Whisper** and **NeMo MSDD**, to deliver accurate transcription and speaker diarization outputs in structured formats like text and CSV files.

# Instalation

## 1. Prerequisite Installations
Essential tools and libraries that need to be installed before proceeding with the main package installations.

- Install ``FFMPEG`` from [here](https://ffmpeg.org/download.html), you can follow a guide like [this](https://phoenixnap.com/kb/ffmpeg-windows) for Windows installation.  
Ensure that FFMPEG is added to your system’s PATH.

- Install ``Strawberry Perl`` from [here](https://strawberryperl.com/).

- ``Visual C++ Build Tools``: If you encounter build errors during installation, install the Visual C++ Build Tools by following this [guide](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools) or download directly from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


## 2. Install Required Packages

After ensuring the prerequisites are set up, proceed with creating the folowing environement :

```
conda create --name wd python=3.10 --yes
conda activate wd
pip install -c constraints.txt -r requirements.txt
pip install ipykernel
pip install pandas
pip install pydub
``` 
### TODO:

- Lock to a Specific Commit to put in the requierements:
``git+https://github.com/username/repo_name.git@commit_hash
``

I specified the last commit that I used in the requierements from: put everything in one requierement file

# Usage

## Transcription & Diarization (Audio-to-Text)
[`run_diarize.py`](run_diarize.py)

``conda activate wd1``

- **Transcribe the audio into his original language :** *(specified with --language)* 
```bash
python run_diarize.py -d .\data\OBE1 --whisper-model large-v3 --language en
```

- **Transcribe and translate the audio into english :** *(e.g. from french to english)*
```bash
python run_diarize.py -d .\data\OBE1 --whisper-model large-v3 --language fr --task translate
```

If only ``language`` is specified, the model will attempt to translate any detected language into the specified language.

To improve performance, specify the task as ``translate`` if you know in advance that the audio is in a certain language (e.g., French) and want to translate it into English.

# Mentions

This work relies heavily on the **Whisper-Diarization** framework to handle transcription and diarization of audio files into structured text formats.

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}
}
```
For additional details, visit the [Whisper-Diarization GitHub repository](https://github.com/MahmoudAshraf97/whisper-diarization).