import difflib
import html
import os
import pandas as pd
import csv
from nltk.tokenize import word_tokenize

# WER metrics performance

def flatten_csv_content(file_path):
    """
    Flatten the 'Content' column of a CSV file into a single string.
    """
    content = []
    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            text = row.get('Content', '').strip()
            content.append(text)
    return ' '.join(content)

def split_long_substitutions(hyp_segment, ref_segment, tolerance, recursion_depth=0, max_recursion_depth=5):
    """
    Recursively splits long substitutions into smaller substitutions and insertions/deletions.
    """
    # Base case to prevent infinite recursion
    if recursion_depth > max_recursion_depth:
        # Handle the hyp_segment length as substitution, rest of ref_segment as insertion
        result_html = ''
        total_D, total_I, total_S = 0, 0, 0
        size_hyp = len(hyp_segment)

        # Handle substitution
        hyp_text = html.escape(' '.join(hyp_segment))
        ref_text = html.escape(' '.join(ref_segment[:size_hyp]))
        result_html += ' <span style="background-color: yellow;"><s>{}</s> → {}</span>'.format(hyp_text, ref_text)
        total_S += size_hyp

        # Remaining tokens in ref_segment are treated as insertions
        ref_segment = ref_segment[size_hyp:]
        if ref_segment:
            insert_text = html.escape(' '.join(ref_segment))
            result_html += ' <span style="background-color: lightgreen;">{}</span>'.format(insert_text)
            total_I += len(ref_segment)

        return result_html, total_D, total_I, total_S

    hyp_words = hyp_segment
    ref_words = ref_segment
    # Create lowercased versions for comparison
    hyp_words_lower = [word.lower() for word in hyp_words]
    ref_words_lower = [word.lower() for word in ref_words]

    sm = difflib.SequenceMatcher(None, hyp_words_lower, ref_words_lower)
    opcodes = sm.get_opcodes()
    result_html = ''
    total_D, total_I, total_S = 0, 0, 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            result_html += ' ' + ' '.join(hyp_words[i1:i2])
        elif tag == 'replace':
            hyp_sub_seg = hyp_words[i1:i2]
            ref_sub_seg = ref_words[j1:j2]
            allowed_length = (i2 - i1) + tolerance
            if (j2 - j1) > allowed_length:
                # Further split the substitution
                sub_html, D, I, S = split_long_substitutions(hyp_sub_seg, ref_sub_seg, tolerance, recursion_depth + 1, max_recursion_depth)
                result_html += ' ' + sub_html
                total_D += D
                total_I += I
                total_S += S
            else:
                hyp_text = html.escape(' '.join(hyp_sub_seg))
                ref_text = html.escape(' '.join(ref_sub_seg))
                result_html += ' <span style="background-color: yellow;"><s>{}</s> → {}</span>'.format(hyp_text, ref_text)
                total_S += max(len(hyp_sub_seg), len(ref_sub_seg))
        elif tag == 'delete':
            delete_text = html.escape(' '.join(hyp_words[i1:i2]))
            result_html += ' <span style="background-color: lightcoral;"><s>{}</s></span>'.format(delete_text)
            total_D += i2 - i1
        elif tag == 'insert':
            insert_text = html.escape(' '.join(ref_words[j1:j2]))
            result_html += ' <span style="background-color: lightgreen;">{}</span>'.format(insert_text)
            total_I += j2 - j1
    return result_html.strip(), total_D, total_I, total_S

def format_diff_WER(ref_text, hyp_text, max_insert_length=None, tolerance_replace=2):
    """
    Compare reference and hypothesis texts and format differences for HTML display.
    Provides options to exclude long consecutive insertions and handles substitutions with tolerance.
    """
    ref_words = word_tokenize(ref_text)
    hyp_words = word_tokenize(hyp_text)
    # Create lowercased versions for comparison
    ref_words_lower = [word.lower() for word in ref_words]
    hyp_words_lower = [word.lower() for word in hyp_words]

    sm = difflib.SequenceMatcher(None, hyp_words_lower, ref_words_lower)
    opcodes = sm.get_opcodes()
    diff_html = ''
    total_D, total_I, total_S = 0, 0, 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            diff_html += ' ' + ' '.join(hyp_words[i1:i2])

        elif tag == 'replace':
            hyp_segment = hyp_words[i1:i2]
            ref_segment = ref_words[j1:j2]
            allowed_length = (i2 - i1) + tolerance_replace
            if (j2 - j1) > allowed_length:
                # Split the long substitution
                sub_html, D, I, S = split_long_substitutions(hyp_segment, ref_segment, tolerance_replace, recursion_depth=1)
                diff_html += ' ' + sub_html
                total_D += D
                total_I += I
                total_S += S
            else:
                hyp_text = html.escape(' '.join(hyp_segment))
                ref_text = html.escape(' '.join(ref_segment))
                diff_html += ' <span style="background-color: yellow;"><s>{}</s> → {}</span>'.format(hyp_text, ref_text)
                total_S += max(len(hyp_segment), len(ref_segment))

        elif tag == 'insert':
            insert_len = j2 - j1
            insert_text = html.escape(' '.join(ref_words[j1:j2]))
            if max_insert_length is not None and insert_len > max_insert_length:
                # Exclude long insertions from WER calculation
                diff_html += ' <span style="background-color: lightgray;">{}</span>'.format(insert_text)
            else:
                diff_html += ' <span style="background-color: lightgreen;">{}</span>'.format(insert_text)
                total_I += insert_len

        elif tag == 'delete':
            delete_len = i2 - i1
            delete_text = html.escape(' '.join(hyp_words[i1:i2]))
            diff_html += ' <span style="background-color: lightcoral;"><s>{}</s></span>'.format(delete_text)
            total_D += delete_len

    return diff_html.strip(), total_D, total_I, total_S


def calculate_wer_and_generate_html(prediction_file, reference_file, output_file, max_insert_length=None, tolerance_replace=2):
    """
    Flatten prediction and reference CSVs, calculate WER, and generate an HTML output highlighting the changes needed
    to match the prediction to the reference. Optionally excludes long consecutive insertions and handles substitutions with tolerance.
    """
    # Step 1: Flatten both prediction and reference CSVs into plain text
    reference_text = flatten_csv_content(reference_file)
    prediction_text = flatten_csv_content(prediction_file)

    # Step 2: Compare the plain texts and highlight what changes are needed
    highlighted_diff, total_D, total_I, total_S = format_diff_WER(reference_text, prediction_text, max_insert_length, tolerance_replace)

    # Step 3: Calculate WER
    ref_words = word_tokenize(reference_text)
    total_words = len(ref_words)
    wer = (total_D + total_I + total_S) / total_words if total_words > 0 else 0

    # Step 4: Save results to an HTML file
    html_output = "<html><head><style>"
    html_output += "body {font-family: Arial, sans-serif;}"
    html_output += "s {text-decoration: line-through;}"
    html_output += "</style></head><body>"

    # Title
    html_output += f"<h1>Transcription (ASR) comparison for {os.path.basename(reference_file)}</h1>"

    # WER Metrics
    html_output += f"<p><strong>WER:</strong> {wer:.2%} &nbsp;&nbsp; "
    html_output += f"Total Words (Reference): {total_words} &nbsp;&nbsp; "
    html_output += f"Deletions: {total_D} &nbsp;&nbsp; "
    html_output += f"Insertions: {total_I} &nbsp;&nbsp; "
    html_output += f"Substitutions: {total_S}</p>"

    # Legend
    html_output += "<p>"
    html_output += "<span style='background-color: yellow;'>Substitution</span> &nbsp; "
    html_output += "<span style='background-color: lightgreen;'>Insertion</span> &nbsp; "
    html_output += "<span style='background-color: lightcoral;'>Deletion</span> &nbsp; "
    html_output += "<span style='background-color: lightgray;'>Excluded from WER</span>"
    html_output += "</p>"

    html_output += "<hr>"
    html_output += f"<div>{highlighted_diff}</div>"
    html_output += "</body></html>"

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(html_output)

    # Step 5: Return metrics
    metrics = {
        'WER': round(wer, 4),
        'Total Words': total_words,
        'Deletions': total_D,
        'Insertions': total_I,
        'Substitutions': total_S,
    }

    return metrics

def process_folder(prediction_folder, reference_folder, max_insert_length=None, tolerance_replace=2, dir_visual = 'visual_comparison',info =None):
    """
    Process all CSV files in the prediction folder, compare with matching files in the reference folder,
    calculate WER and Diarization metrics, generate HTML visual files, and save metrics to a CSV file.

    Parameters:
        prediction_folder (str): Path to the folder containing prediction CSV files.
        reference_folder (str): Path to the folder containing reference CSV files.
        max_insert_length (int, optional): Maximum length for insertions (used in WER calculation).
        tolerance_replace (int, optional): Tolerance for replacements (used in WER calculation).
        dir_visual (str, optional): Name of the subdirectory to save visual comparison HTML files.

    Returns:
        pd.DataFrame: DataFrame containing WER and Diarization metrics for all processed files.
    """
    metrics_list = []

    # Create the 'visual comparison' folder inside the reference folder if it doesn't exist
    visual_comparison_folder = os.path.join(prediction_folder, dir_visual)
    os.makedirs(visual_comparison_folder, exist_ok=True)
    
    # Loop over all CSV files in the prediction folder
    for filename in os.listdir(reference_folder):
        if filename.endswith('.csv'):
            prediction_file = os.path.join(prediction_folder, filename)
            reference_file = os.path.join(reference_folder, filename)
            if os.path.exists(prediction_file):
                print(f"Processing file: {filename}")
                # Define the output HTML file path
                base_name = os.path.splitext(filename)[0]
                wer_output_file = os.path.join(visual_comparison_folder, f'{base_name}_WER.html')
                dia_output_file = os.path.join(visual_comparison_folder, f'{base_name}_Diarization.html')

                # Call the function to calculate WER and generate HTML
                wer_metrics = calculate_wer_and_generate_html(
                    prediction_file,
                    reference_file,
                    wer_output_file,
                    max_insert_length=max_insert_length,
                    tolerance_replace=tolerance_replace
                )

                if info is not None:
                    ref_duration = info[info["File_name"] == base_name]["Duration_sec"].values[0]
                else:
                    ref_duration = None
                    
                # Call the function to generate Diarization HTML and get Diarization metrics
                dia_metrics = diarisation_html(
                    reference_file,
                    prediction_file,
                    dia_output_file,
                    info_ref = ref_duration
                )

                # Combine WER and Diarization metrics
                combined_metrics = {
                    'Filename': filename,
                    'WER': wer_metrics.get('WER'),
                    'DER': dia_metrics.get('DER'),
                    'Total Words': wer_metrics.get('Total Words'),
                    'Deletions': wer_metrics.get('Deletions'),
                    'Insertions': wer_metrics.get('Insertions'),
                    'Substitutions': wer_metrics.get('Substitutions'),
                    'Reference Speech Duration': dia_metrics.get('Reference Speech Duration'),
                    'Missed Duration': dia_metrics.get('Missed Duration'),
                    'False Alarm Duration': dia_metrics.get('False Alarm Duration'),
                    'Confusion Duration': dia_metrics.get('Confusion Duration')
                }

                # Append combined metrics to the list
                metrics_list.append(combined_metrics)

                print(f"Processed file: {filename}")
            else:
                print(f"No matching reference file for {filename} in {prediction_folder}")

    # Save metrics to a CSV file
    metrics_df = pd.DataFrame(metrics_list)

    if not metrics_df.empty:
        return metrics_df
    else:
        print("No metrics to save.")


# Diarisation metrics performance

def time_to_seconds(t):
    """
    Convert time format 'HH:MM:SS', 'HH:MM:SS.mmm', or 'HH:MM:SS,mmm' to seconds.
    """
    t = t.strip()
    t = t.replace(',', '.')  # Replace comma with dot for decimal separator
    parts = t.split(':')
    if len(parts) != 3:
        raise ValueError(f"Time format incorrect: {t}")
    h, m, s = parts
    h = int(h)
    m = int(m)
    s = float(s)
    return h * 3600 + m * 60 + s

def format_time(seconds):
    """
    Format seconds into 'HH:MM:SS.mmm'.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def load_data_time(reference_file, prediction_file):
    """
    Load reference and prediction data from CSV files and convert times to seconds.
    """
    df_ref = pd.read_csv(reference_file)
    df_pred = pd.read_csv(prediction_file)

    # Convert times to seconds
    df_ref['Start'] = df_ref['Start Time'].apply(time_to_seconds)
    df_ref['End'] = df_ref['End Time'].apply(time_to_seconds)
    df_pred['Start'] = df_pred['Start Time'].apply(time_to_seconds)
    df_pred['End'] = df_pred['End Time'].apply(time_to_seconds)

    return df_ref, df_pred

def create_time_segments(df_ref, df_pred):
    """
    Create non-overlapping time segments based on unique time points from both DataFrames.
    """
    time_points = set(df_ref['Start']).union(df_ref['End']).union(df_pred['Start']).union(df_pred['End'])
    sorted_time_points = sorted(time_points)
    segments = [(sorted_time_points[i], sorted_time_points[i+1]) for i in range(len(sorted_time_points)-1)]
    return segments

def get_active_speakers(df, start_time, end_time):
    """
    Get active speakers during a time segment.
    """
    mask = (df['Start'] < end_time) & (df['End'] > start_time)
    active_speakers = df.loc[mask, 'Speaker'].unique()
    return set(map(str, active_speakers))

def compute_der(df_ref, df_pred, total_ref_duration=None, tolerance=1.0):
    """
    Compute the Diarization Error Rate (DER) and error durations, excluding silence periods
    where neither reference nor prediction has active speakers.
    Incorporates a tolerance window for missed detections and false alarms, due to manual annotation delays & overlaps.
    """
    segments = create_time_segments(df_ref, df_pred)
    total_ref_speech_duration = 0.0
    total_missed_duration = 0.0
    total_false_alarm_duration = 0.0
    total_confusion_duration = 0.0

    dialogue_data = []

    for start_time, end_time in segments:
        duration = end_time - start_time
        ref_speakers = get_active_speakers(df_ref, start_time, end_time)
        pred_speakers = get_active_speakers(df_pred, start_time, end_time)

        if ref_speakers:
            total_ref_speech_duration += duration

        # Skip segments where there are **no speakers** in both reference and prediction
        if not ref_speakers and not pred_speakers:
            continue  # Ignore this segment

        if ref_speakers == pred_speakers:
            # Correct detection
            error_type = ''
        elif ref_speakers and pred_speakers:
            total_confusion_duration += duration
            error_type = 'Confusion'
        elif ref_speakers and not pred_speakers:
            # Check for nearby prediction within tolerance
            nearby_pred_speakers = has_nearby_speaker(df_pred, start_time, end_time, tolerance)
            if nearby_pred_speakers.size > 0:
                # Do not count as missed detection
                error_type = ''  # No error
            else:
                total_missed_duration += duration
                error_type = 'Missed Detection'
        elif not ref_speakers and pred_speakers:
            # Check for nearby reference within tolerance
            nearby_ref_speakers = has_nearby_speaker(df_ref, start_time, end_time, tolerance)
            if nearby_ref_speakers.size > 0:
                # Do not count as false alarm
                error_type = ''  # No error
            else:
                total_false_alarm_duration += duration
                error_type = 'False Alarm'
        else:
            # This case should not occur due to the earlier check, but added for completeness
            error_type = 'Silence'

        dialogue_data.append({
            'Start Time': format_time(start_time),
            'End Time': format_time(end_time),
            'Reference Speaker': ', '.join(ref_speakers) if ref_speakers else '',
            'Prediction Speaker': ', '.join(pred_speakers) if pred_speakers else '',
            'Error Type': error_type
        })

    if total_ref_duration is not None: # As we only have the min and max time detected by the model, therefore worst case scenario
        #print(f"Previous calculated Reference Duration: {total_ref_speech_duration}")
        total_ref_speech_duration = total_ref_duration
        #print(f"New Reference Duration: {total_ref_speech_duration}")

    total_error_duration = total_missed_duration + total_false_alarm_duration + total_confusion_duration
    DER = (total_error_duration / total_ref_speech_duration) if total_ref_speech_duration > 0 else 0.0

    error_durations = {
        'DER': round(DER, 4),
        'Reference Speech Duration': round(total_ref_speech_duration,4),
        'Missed Duration': round(total_missed_duration,4),
        'False Alarm Duration': round(total_false_alarm_duration,4),
        'Confusion Duration': round(total_confusion_duration,4)
    }

    dialogue_df = pd.DataFrame(dialogue_data)
    return dialogue_df, error_durations

def has_nearby_speaker(df, start_time, end_time, tolerance):
    """
    Check if there is any speaker in df within start_time - tolerance to end_time + tolerance
    """
    mask = (df['Start'] < end_time + tolerance) & (df['End'] > start_time - tolerance)
    return df.loc[mask, 'Speaker'].unique()


# For Visualization

def align_rows_by_time(df_ref, df_pred):
    """
    Align rows from reference and prediction DataFrames based on overlapping time intervals.
    """
    aligned_pairs = []
    ref_intervals = list(zip(df_ref['Start'], df_ref['End'], df_ref.index))
    pred_intervals = list(zip(df_pred['Start'], df_pred['End'], df_pred.index))

    ref_idx = 0
    pred_idx = 0

    while ref_idx < len(ref_intervals) and pred_idx < len(pred_intervals):
        ref_start, ref_end, ref_i = ref_intervals[ref_idx]
        pred_start, pred_end, pred_i = pred_intervals[pred_idx]

        # Skip if start or end times are None
        if ref_start is None or ref_end is None:
            ref_idx += 1
            continue
        if pred_start is None or pred_end is None:
            pred_idx += 1
            continue

        # Ensure intervals have a minimum duration
        if ref_end <= ref_start:
            ref_end = ref_start + 0.001  # Add small epsilon
        if pred_end <= pred_start:
            pred_end = pred_start + 0.001  # Add small epsilon

        # Check for overlap with a small tolerance
        overlap_start = max(ref_start, pred_start)
        overlap_end = min(ref_end, pred_end)

        if overlap_start < overlap_end + 0.001:
            # Intervals overlap
            aligned_pairs.append((df_ref.loc[ref_i], df_pred.loc[pred_i]))
            ref_idx += 1
            pred_idx += 1
        elif ref_end <= pred_start:
            # Reference interval ends before prediction interval starts
            aligned_pairs.append((df_ref.loc[ref_i], None))  # Missing in prediction
            ref_idx += 1
        else:
            # Prediction interval ends before reference interval starts
            aligned_pairs.append((None, df_pred.loc[pred_i]))  # Extra in prediction
            pred_idx += 1

    # Handle any remaining intervals

    # As missed detections (remaining reference intervals not aligned with prediction)
    while ref_idx < len(ref_intervals):
        ref_start, ref_end, ref_i = ref_intervals[ref_idx]
        aligned_pairs.append((df_ref.loc[ref_i], None))
        ref_idx += 1

    # As false alarms (remaining prediction intervals not aligned with reference)
    while pred_idx < len(pred_intervals):
        pred_start, pred_end, pred_i = pred_intervals[pred_idx]
        aligned_pairs.append((None, df_pred.loc[pred_i]))
        pred_idx += 1

    return aligned_pairs

def format_diff(ref_text, hyp_text):
    """
    Format differences between reference and hypothesis texts for HTML display.
    """
    # Replace non-string inputs with empty strings
    ref_text = ref_text if isinstance(ref_text, str) else ""
    hyp_text = hyp_text if isinstance(hyp_text, str) else ""

    ref_words = ref_text.split()
    hyp_words = hyp_text.split()
    sm = difflib.SequenceMatcher(None, hyp_words, ref_words) # Highlight the changes needed in the prediction to match the reference
    opcodes = sm.get_opcodes()
    diff_html = ''
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            diff_html += ' ' + ' '.join(hyp_words[i1:i2])
        elif tag == 'replace':
            diff_html += ' <span style="background-color: yellow;">[<s>{}</s> → {}]</span>'.format(
                ' '.join(hyp_words[i1:i2]), ' '.join(ref_words[j1:j2]))
        elif tag == 'insert':
            diff_html += ' <span style="background-color: lightgreen;">[{}]</span>'.format(
                ' '.join(ref_words[j1:j2]))
        elif tag == 'delete':
            diff_html += ' <span style="background-color: lightcoral;">[<s>{}</s>]</span>'.format(
                ' '.join(hyp_words[i1:i2]))
    return diff_html.strip()


def format_cell_diff(ref_value, pred_value):
    """
    Format cell differences for columns other than 'Content' (text).
    """
    if str(ref_value) != str(pred_value):
        if str(pred_value) != '' and str(ref_value) != '':
            # Speaker misattribution or bad timing
            return f"<td style='background-color: mediumpurple;'>{pred_value} → {ref_value}</td>"
        elif str(ref_value) != '':  
            # Added row
            return f"<td style='background-color: lightgray;'> → {ref_value}</td>"
        else:
            # Removed row
            return f"<td style='background-color: lightgray;'>{pred_value} → </td>"
    else:
        # No change
        return f"<td>{ref_value}</td>"

def diarisation_html(reference_file, prediction_file, output_file, info_ref=None, tolerance=1.0):
    """
    Compare two CSV files and generate an HTML output highlighting the differences,
    including the dialogue DataFrame and diarization metrics.

    Returns:
        dict: A dictionary containing diarization metrics (error_durations).
    """
    # Read CSV files
    df_ref = pd.read_csv(reference_file, encoding='utf-8')
    df_pred = pd.read_csv(prediction_file, encoding='utf-8')

    # Convert times to seconds
    df_ref['Start'] = df_ref['Start Time'].apply(time_to_seconds)
    df_ref['End'] = df_ref['End Time'].apply(time_to_seconds)
    # Adjust zero-length intervals
    df_ref.loc[df_ref['End'] <= df_ref['Start'], 'End'] = df_ref['Start'] + 0.001

    df_pred['Start'] = df_pred['Start Time'].apply(time_to_seconds)
    df_pred['End'] = df_pred['End Time'].apply(time_to_seconds)
    # Adjust zero-length intervals
    df_pred.loc[df_pred['End'] <= df_pred['Start'], 'End'] = df_pred['Start'] + 0.001

    # Align rows
    aligned_pairs = align_rows_by_time(df_ref, df_pred)

    # Compute DER and get dialogue DataFrame
    dialogue_df, error_durations = compute_der(df_ref, df_pred, total_ref_duration=info_ref, tolerance=tolerance)

    # Prepare HTML output
    html_output = "<html><head><style>"
    # Basic table styling
    html_output += """
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 20px;
    }
    th, td {
        border: 1px solid black;
        padding: 5px;
        text-align: left;
    }
    s {
        text-decoration: line-through;
    }
    /* Flex container for side-by-side tables */
    .flex-container {
        display: flex;
        gap: 20px;
    }
    .flex-item {
        flex: 1;
        overflow-x: auto;
    }
    /* Styling for the metrics section */
    .metrics {
        margin-bottom: 20px;
    }
    .metrics span {
        display: inline-block;
        margin-right: 15px;
        font-weight: bold;
    }
    """
    html_output += "</style></head><body>"

    # Title
    html_output += f"<h1>Diarization comparison for {os.path.basename(reference_file)}</h1>"

    # Diarization Metrics as a single row of text
    html_output += "<div class='metrics'>"
    metrics_text = []
    for key, value in error_durations.items():
        if key == 'DER':
            metrics_text.append(f"<strong>{key}:</strong> {value:.2%}")
        else:
            metrics_text.append(f"<strong>{key}:</strong> {value:.3f} seconds")
    html_output += " | ".join(metrics_text)
    html_output += "</div>"

    # Legend
    html_output += "<p>"
    html_output += "<span style='background-color: mediumpurple; padding: 2px 5px;'>Confusion or bad timing</span> &nbsp; "
    html_output += "<span style='background-color: lightgray; padding: 2px 5px;'>Added or removed row</span> &nbsp; "
    html_output += "<span style='background-color: yellow; padding: 2px 5px;'>Substitution</span> &nbsp; "
    html_output += "<span style='background-color: lightgreen; padding: 2px 5px;'>Insertion</span> &nbsp; "
    html_output += "<span style='background-color: lightcoral; padding: 2px 5px;'>Deletion</span>"
    html_output += "</p>"

    # Container for side-by-side tables
    html_output += "<div class='flex-container'>"

    # Aligned Rows Visualization Table
    html_output += "<div class='flex-item'>"
    html_output += "<h3>Aligned Rows Visualization</h3>"
    html_output += "<table>"
    html_output += "<tr><th>Index</th><th>Start Time</th><th>End Time</th><th>Speaker</th><th>Content</th></tr>"

    for idx, (ref_row, pred_row) in enumerate(aligned_pairs):
        html_output += "<tr>"
        # Index
        html_output += f"<td>{idx + 1}</td>"

        # Start Time
        ref_start_time = ref_row['Start Time'] if ref_row is not None else ''
        pred_start_time = pred_row['Start Time'] if pred_row is not None else ''
        html_output += format_cell_diff(ref_start_time, pred_start_time)

        # End Time
        ref_end_time = ref_row['End Time'] if ref_row is not None else ''
        pred_end_time = pred_row['End Time'] if pred_row is not None else ''
        html_output += format_cell_diff(ref_end_time, pred_end_time)

        # Speaker
        ref_speaker = ref_row['Speaker'] if ref_row is not None else ''
        pred_speaker = pred_row['Speaker'] if pred_row is not None else ''
        html_output += format_cell_diff(ref_speaker, pred_speaker)

        # Content
        ref_content = ref_row['Content'] if ref_row is not None else ''
        pred_content = pred_row['Content'] if pred_row is not None else ''
        content_diff_html = format_diff(ref_content, pred_content)
        html_output += f"<td>{content_diff_html}</td>"

        html_output += "</tr>"

    html_output += "</table>"
    html_output += "</div>"  # Close Aligned Rows Visualization div

    # Dialogue Error Analysis Table
    html_output += "<div class='flex-item'>"
    html_output += "<h3>Dialogue Error Analysis</h3>"
    html_output += "<table>"
    html_output += "<tr><th>Start Time</th><th>End Time</th><th>Ref</th><th>Pred</th><th>Error Type</th></tr>"

    # Define styles for error types
    error_styles = {
        'Confusion': "style='background-color: mediumpurple;'",
        'Missed Detection': "style='background-color: lightgray;'",
        'False Alarm': "style='background-color: lightblue;'",
        'Silence': "",
        '': ""  # Correct detection
    }

    for idx, row in dialogue_df.iterrows():
        error_type = row['Error Type']
        style = error_styles.get(error_type, "")
        html_output += f"<tr {style}>"
        html_output += f"<td>{row['Start Time']}</td>"
        html_output += f"<td>{row['End Time']}</td>"
        html_output += f"<td>{row['Reference Speaker']}</td>"
        html_output += f"<td>{row['Prediction Speaker']}</td>"
        html_output += f"<td>{error_type}</td>"
        html_output += "</tr>"

    html_output += "</table>"
    html_output += "</div>"  # Close Dialogue Error Analysis div

    html_output += "</div>"  # Close flex-container div

    html_output += "</body></html>"

    # Save HTML output
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"Combined HTML file saved as {output_file}")
    

    return error_durations
