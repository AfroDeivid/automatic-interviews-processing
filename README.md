# Automatic Interviews processing
short description

# Instalation

# Transcription & Diarization (Audio-to-Text)

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
git+https://github.com/username/repo_name.git@commit_hash

I specified the last commit that I used in the requierements from the

- Put everything in one requierement file
- Give credits to the repo used

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}
}
```

# Usage

## Transcription & Diarization (Audio-to-Text)
[`run_diarize.py`](run_diarize.py)

``conda activate wd1``

- **Transcribe the audio into his original language :** *(specified with --language)* 
```bash
python run_diarize.py -d ".\data\OBE1" --whisper-model large-v3 --language en
```

- **Transcribe and translate the audio into english :** *(e.g. from french to english)*
```bash
python run_diarize.py -d ".\data\OBE1" --whisper-model large-v3 --language fr --task translate
```

If only the language is specified, the model will attempt to translate any detected language into the specified language.

To improve performance, specify the task as "translate" if you know in advance that the audio is in a certain language (e.g., French) and want to translate it into English.

## Translation & parsing from words to csv
[`run_translation.py`](run_translation.py)

``conda activate seam``

- **With CUDA (default):**

```bash
python run_translation.py -d ".\data\Parkinson" --type "docx" --source-lang fra --target-lang eng 
```

- **With CPU:**

```bash
python run_translation.py -d ".\data\Parkinson\fr" --type "docx" --source-lang fra --target-lang eng --use-cpu
```

- **Without Translation:**

```bash
python run_translation.py -d ".\data\Parkinson\en" --type "docx" --no-translate
```

# Mentions

WD 


# Guidelines

## Excel tip

Select an entire row: "Shift" + "Space"

Select an entire column: "Ctrl" + "Space"

Insert a new row: "Ctrl" + "+" (below the actual row)

Enter a cell wihout overide: "F2"

Select a column of continous data: Ctrl + Shift + Down Arrow 

Copy paste a row and inserted below: 
- "Shift" + "Space"
- "Ctrl" + "C"
- "Ctrl" + "+"