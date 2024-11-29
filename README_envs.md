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
conda install plotly -y
pip install --upgrade nbformat
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