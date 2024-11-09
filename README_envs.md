# Set-Up for Basics analysis & use of this repo
- audio_and_experiment
- analyis
- evaluation

## Create env

```
conda create --name basic python=3.10 --yes
conda activate basic
pip install ipykernel
pip install pandas
pip install pydub
pip install matplotlib
pip install seaborn
pip install nltk
pip install openpyxl
pip install wordcloud
```
# Clusters
pip install scikit-learn
pip install sentence_transformers

??
pip install ipywidgets --upgrade

### Conventional metric in evaluation folder
pip install jiwer
pip install evaluate


# Set-Up for Translation (Text-to-Text)

## Seamless

```
conda create --name seam python=3.10 --yes
conda activate seam
pip install ipykernel
pip install ipywidgets
pip install python-docx
pip install tqdm
pip install transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install sentencepiece
pip install protobuf
```