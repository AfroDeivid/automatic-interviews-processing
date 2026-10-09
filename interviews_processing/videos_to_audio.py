import os
import subprocess
import argparse

def convert_and_backup(raw_root_dir, extensions, output_ext, not_audio_only):
    input_directory = raw_root_dir
    valid_extensions = [ext.lower() for ext in extensions]

    # Create dynamic output root directory: raw_root_dir + "_audios"
    output_root_dir = raw_root_dir.rstrip("/\\") + "_audios"
    os.makedirs(output_root_dir, exist_ok=True)

    print(f"Output directory: {output_root_dir}")

    for subdir, _, files in os.walk(input_directory):
        print(f"Scanning: {subdir}")

        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                print(f"Matched file: {file}")

                # Full input path
                input_path = os.path.join(subdir, file)

                # Preserve folder structure
                relative_path = os.path.relpath(subdir, input_directory)
                output_dir = os.path.join(output_root_dir, relative_path)
                os.makedirs(output_dir, exist_ok=True)

                # Output filename
                output_filename = f"{os.path.splitext(file)[0]}.{output_ext}"
                output_path = os.path.join(output_dir, output_filename)

                # Skip if file already exists ----
                if os.path.exists(output_path):
                    print(f"Skipping (already exists): {output_path}")
                    continue

                # ffmpeg command
                if not_audio_only:
                    command = [
                        "ffmpeg", "-i", input_path,
                        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                        output_path
                    ]
                else:
                    command = ["ffmpeg", "-i", input_path, output_path]

                # Run conversion
                subprocess.run(command)
                print(f"Converted: {input_path} -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert files of given extensions to another audio format."
    )
    parser.add_argument(
        "--raw_root_dir",
        type=str,
        default="./data/grief",
        help="Path to the root directory containing files (default: ./data/grief)"
    )
    parser.add_argument(
        "--not_audio_only",
        action="store_true",
        help="Flag to indicate if the conversion is not audio-only"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mts", ".mp4", ".m4a"],
        help="List of extensions to convert (default: .mts .mp4 .m4a)"
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default="wav",
        help="Extension of output format (default: wav)"
    )

    args = parser.parse_args()

    convert_and_backup(args.raw_root_dir, args.extensions, args.output_ext, args.not_audio_only)