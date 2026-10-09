import os
import shutil

# =======================
# Paths
# =======================
BASE = "data"

AUDIO = os.path.join(BASE, "grief_audios")
FR = os.path.join(BASE, "grief_fr")
ENG = os.path.join(BASE, "grief_eng")
MISSING = os.path.join(BASE, "grief_missing")

os.makedirs(MISSING, exist_ok=True)


# =======================
# Utils
# =======================
def move_or_delete_files(src_dir, dst_dir):
    """
    - Move files if they don't exist in destination
    - Delete files if they already exist in destination
    - Remove empty source directories
    """
    os.makedirs(dst_dir, exist_ok=True)

    # Process files
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        target = os.path.join(dst_dir, rel)
        os.makedirs(target, exist_ok=True)

        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(target, f)

            if os.path.exists(dst_file):
                os.remove(src_file)
                print(f"Deleted (already exists): {src_file}")
            else:
                shutil.move(src_file, dst_file)
                print(f"Moved: {src_file} -> {dst_file}")

    # Clean empty directories (bottom-up)
    for root, dirs, files in os.walk(src_dir, topdown=False):
        if not dirs and not files:
            os.rmdir(root)


# =======================
# Main logic
# =======================
audio_folders = sorted(os.listdir(AUDIO))

for folder in audio_folders:
    audio_path = os.path.join(AUDIO, folder)

    if not os.path.isdir(audio_path):
        continue

    fr_path = os.path.join(FR, folder)
    eng_path = os.path.join(ENG, folder)

    exists_fr = os.path.isdir(fr_path)
    exists_eng = os.path.isdir(eng_path)

    if exists_fr:
        print(f"→ Processing French folder: {folder}")
        move_or_delete_files(audio_path, fr_path)

    if exists_eng:
        print(f"→ Processing English folder: {folder}")
        move_or_delete_files(audio_path, eng_path)

    # If still exists and empty → remove it
    if os.path.isdir(audio_path) and not os.listdir(audio_path):
        os.rmdir(audio_path)

    # If no FR and no ENG match → move whole folder
    if not exists_fr and not exists_eng:
        print(f"→ No match for {folder}, moving to grief_missing/")
        shutil.move(audio_path, os.path.join(MISSING, folder))