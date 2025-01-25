# Automatic Interviews Processing: From Transcriptions to Insights

This repository provides a scalable and automated pipeline for transcription and diarization of audio interviews, tailored to handle real-world challenges such as noisy recordings, overlapping speakers, and multi-language scenarios. It leverages open-source tools, including **Whisper** and **NeMo MSDD**, to deliver accurate transcription and speaker diarization outputs in text and CSV format.

Developed by [David Friou](https://github.com/AfroDeivid) as part of a semester project at LNCO Lab.

![Project Workflow](src/images/readme/WD_pipeline.png)

## Table of Contents
1. [Installation](#installation)  
   1. [Prerequisites](#1-prerequisites)  
   2. [Installing Required Packages](#2-installing-required-packages)  
2. [Usage](#usage)  
   1. [Preparing Your Data](#preparing-your-data)  
   2. [Transcription & Diarization (Audio-to-Text)](#transcription--diarization-audio-to-text)  
   3. [Outputs](#outputs)  
3. [File Structure](#file-structure)  
   1. [Audio-to-Text Processing](#audio-to-text-processing)  
   2. [Transcript Evaluation](#transcript-evaluation)  
   3. [Text and Topic Analysis](#text-and-topic-analysis) 
4. [Mentions](#mentions)

## Features

- **End-to-End Pipeline:** From raw audio (multiple formats) to cleaned, diarized transcripts in `.txt` and `.csv`.  
- **Multi-Language Support:** Enabling automatic language detection or user-defined translation tasks.  
- **Scalable Processing:** Processes nested folder structures, allowing large numbers of audio files spread across multiple experiments.  
- **Post-Processing:** Includes scripts and workflows for text cleaning, removing fillers, and predicting speaker roles (e.g., interviewer vs. participant) to prepare transcripts for analysis.  
- **Analysis & Topic Modeling:** Provides notebooks for detailed **text** analysis, including word count, keyword extraction, and topic modeling to uncover themes and patterns within the transcripts. 

# Installation

## 1. Prerequisites
Essential tools and libraries that need to be installed before proceeding with the main package installations.

- Install ``FFMPEG`` from [here](https://ffmpeg.org/download.html), you can follow a guide like [this](https://phoenixnap.com/kb/ffmpeg-windows) for Windows installation.  
Ensure that FFMPEG is added to your system’s PATH.

- Install ``Strawberry Perl`` from [here](https://strawberryperl.com/).

- ``Visual C++ Build Tools``: If you encounter build errors during installation, install the Visual C++ Build Tools by following this [guide](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools) or download directly from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).


## 2. Installing Required Packages

After ensuring the prerequisites are set up, you can proceed with creating the folowing environement :

**Locked Environment Installation**  
This setup recreates the *exact environment* used during my semester project:
```
conda create --name tti python=3.10 --yes
conda activate tti
pip install cython
pip install -r freeze_requirements.txt
python -m spacy download en_core_web_sm
``` 

**Flexible / Adaptive Installation**  
If you need more flexibility, like updating certain packages or adapting the repository replace the ``pip install -r freeze_requirements.txt`` step with:
```
pip install -c constraints.txt -r requirements.txt 
``` 

# Usage

## Preparing Your Data

The pipeline supports nested folder structures, making it easy to process multiple experiments and interviews. To use the pipeline:

- Simply upload your *main folder(s)* with your audio files into the ``data`` folder.
  - Each *main folders*, can corresponds to an experiment (e.g., OBE1, OBE2, Compassion).
- The pipeline recursively processes all audio files within these folders and subfolders.

## Transcription & Diarization (Audio-to-Text)
Use the [run_diarize.py](./run_diarize.py) script to transcribe and diarize audio:

- **Transcribe the audio in his original language :** *(specified with --language)* 
```bash
python run_diarize.py -d .\data\OBE1 --whisper-model large-v3 --language en
```

- **Transcribe and translate the audio to english :** *(e.g. from french to english)*
```bash
python run_diarize.py -d .\data\OBE1 --whisper-model large-v3 --language fr --task translate
```

If only ``language`` is specified, the model will attempt to translate any detected language into the specified language.

To improve performance, specify the task as ``translate`` if you know in advance that the audio is in a certain language (e.g., French) and want to translate it into English.

- You can view the list of all supported languages along with their corresponding language codes just here: [Languages](src/whisper_diarization/helpers.py)

| Parameter         | Description                                         | Default                         |
|-------------------|-----------------------------------------------------|---------------------------------|
| **`-d, --directory`** | Path to the directory containing audio files.       | None                 |
| **`--whisper_model`** | Name of the Whisper model used for transcription.   | None                      |
| **``--language ``**       | Language code for transcription (e.g., `fra` for French, `eng` for English). | None                            |
| **``--task ``**           | Task to perform (e.g., "transcribe", "translate").  | None                            |
| **``-e, --extensions``**      | List of allowed audio file extensions.              | [".m4a", ".mp4", ".wav"]        |
| **``--overwrite``**       | Overwrites existing transcriptions if specified.    | False                           |

*See [run_diarize.py](./run_diarize.py) for additional information.*

## Outputs
- **Text Format:** Simplified and easy-to-read files for manual review.
- **CSV Format:** A structured format ideal for analysis, with columns such as:
  - Experiment name (derived from the main folder).
  - File name.
  - Participant ID.
  - Timestamps for each segment.
  - Speaker roles and transcription content.

### Processed folder
Contains the same outpouts after aditional preprocessing steps:

- Removal of vocalized fillers
- Visual cleaning of the text
- Prediction of the speaker role in interview set-up (Participant & Interviewer)

For a more modular approach you can use [preprocessing notebook](./src/preprocessing.ipynb).

# File Structure

## Audio-to-Text Processing  
This section focuses on converting raw audio data into text through transcription and diarization, enabling subsequent analysis.  

- **Preprocessing and Conversion:**  
  - [src/pre_analysis.ipynb](./src/pre_analysis.ipynb): Analyzes audio files and experiment structure.  
  - [MTS_to_audio.py](./MTS_to_audio.py): Converts `.MTS` videos into `.wav` format for processing.  

- **Transcription & Diarization:**  
  - [run_diarize.py](./run_diarize.py): The main script for batch-processing transcription and speaker diarization.  
  - [/src/whisper_diarization](./src/whisper_diarization/): Source code from the Whisper-Diarization framework. (See [Mentions](#mentions))  
  - [nemo_msdd_configs/](./nemo_msdd_configs/): YAML configuration files for diarization tasks.  

- **Transcript Preprocessing:**  
  - [src/preprocessing.ipynb](./src/preprocessing.ipynb): Modular workflow for cleaning and preparing transcripts for further analysis.  

``/src/utils/`` [format_helpers.py](./src/utils/format_helpers.py) and [preprocessing_helpers.py](./src/utils/preprocessing_helpers.py): Assist with structured formatting and transcript preprocessing.  

## Transcript Evaluation  
This section focuses on validating transcription quality and ensuring the accuracy of the processed data.  
 
- [src/evaluation.ipynb](./src/evaluation.ipynb): Assesses transcription accuracy after manual verification, ensuring reliable input for downstream analyses.  

``/src/utils/`` [evaluation_helpers.py](./src/utils/evaluation_helpers.py) and [text_html.py](./src/utils/text_html.py): Provides functions for transcription/diarization performance evaluations.

## Text and Topic Analysis  
This section delves into analyzing text for patterns and extracting thematic insights through topic modeling.  

- **Text Analysis:**  
  - [src/analysis_text.ipynb](./src/analysis_text.ipynb): Explores transcript content, including distributions (e.g., conditions, interviewers), word count, and keywords.  

- **Automated Topic Modeling:**  
  - [src/bert_topic.ipynb](./src/bert_topic.ipynb): Performs topic modeling to identify themes within the transcripts.  

- **Topic Analysis & Visualization:**  
  - [src/analysis_topics.ipynb](./src/analysis_topics.ipynb): Provides high-level overviews and detailed analyses of identified topics.  
  - [src/topic_overview.ipynb](./src/topic_overview.py): A Streamlit app for interactively exploring specific topics. 
    - Run via `streamlit run .\src\topic_overview.py`. 

``/src/utils/`` [analysis_helpers.py](./src/utils/analysis_helpers.py): Shared utility functions for text and topic analyses.

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