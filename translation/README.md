# Translation & parsing from .docxs to .csv
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