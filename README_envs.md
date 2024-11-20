# Set-Up for Basics analysis & use of this repo
- audio_and_experiment
- analyis
- evaluation

conda remove -n MC --all
conda remove -n MC --all

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
pip install bertopic
```

# Topic
conda create --name topic python=3.10 --yes
conda activate topic
conda install ipykernel -y
conda install pandas  -y
conda install matplotlib -y
conda install seaborn -y
conda install wordcloud -y
conda install nltk -y
pip install bertopic
pip install bertopic[spacy]
python -m spacy download en_core_web_sm



conda install -c conda-forge gensim  --yes
conda install numpy  --yes
conda install scikit-learn --yes
conda install sentence_transformers --yes
conda install matplotlib  --yes 
conda install pandas  --yes
conda install seaborn  --yes
conda install nltk  --yes
conda install wordcloud -y
pip install bertopic

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

## translate_eval

```
conda create -n translation_eval python=3.9 -y
conda activate translation_eval
conda install -c conda-forge ipykernel
conda install sacrebleu -yes
conda install pandas nltk -y

pip install --upgrade pip  # ensures that pip is current
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

wget https://storage.googleapis.com/bleurt-oss/20/BLEURT-20.zip
unzip BLEURT-20.zip -d bleurt_checkpoints




```