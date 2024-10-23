# Meditation-Interviews

# run_diarize.py

``conda activate wd``

```bash
python run_diarize.py -d ".\data\OBE1" --whisper-model large-v3 --language en
```

# run_translation

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


# Excel tip

Select an entire row: "Shift" + "Space"
Select an entire column: "Ctrl" + "Space"
Insert a new row: "Ctrl" + "+" (below the actual row)

Copy paste a row and inserted below: 
- "Shift" + "Space"
- "Ctrl" + "C"
- "Ctrl" + "+"

