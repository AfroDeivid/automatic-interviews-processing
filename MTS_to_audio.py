import os
import subprocess

input_directory = "./data/Grief/MTS"
output_directory = "./data/Grief"

for subdir, _, files in os.walk(input_directory):
    for file in files:
        if file.endswith(".MTS"):

            # Create corresponding subdirectory in destination folder
            relative_path = os.path.relpath(subdir, input_directory)
            destination_subdir = os.path.join(output_directory, relative_path)
            os.makedirs(destination_subdir, exist_ok=True)

            input_path = os.path.join(subdir, file)
            output_filename = f"{os.path.splitext(file)[0]}.wav"  # Fixed extension to .wav
            output_path = os.path.join(destination_subdir, output_filename)
            
            # Run the FFmpeg command to extract audio and save as WAV
            command = ["ffmpeg", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", output_path]
            subprocess.run(command)
            
            print(f"Extracted audio from {file} to {output_filename}")
