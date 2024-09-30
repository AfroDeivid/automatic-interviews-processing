# Set-Up for Translation (Text-to-Text)

## Argos

```
conda create --name argos python=3.10 --yes
conda activate argos
pip install argos-translate-files
pip install ipykernel
```

## Seamless

```
conda create --name seam python=3.10 --yes
conda activate seam
pip install ipykernel
pip install tqdm
pip install transformers
```

### With Cuda
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

```
pip install tiktoken
pip install protobuf
pip install blobfile
```

maybe only need-> pip install sentencepiece


pip install python-docx
