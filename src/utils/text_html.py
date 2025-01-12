import re
from bs4 import BeautifulSoup

def chunk_div_content(transcript_div):
    """
    Splits the content of a <div> into chunks (text vs. span).
    Returns a list of tuples:
      - (content, 'text', speaker_label, extra_info)
      - (content, 'span', speaker_label, extra_info)

    Where:
      - extra_info: 'text_ends_with_bracket' for text or 'sub', 'del', or 'ins' for spans.
      - speaker_label: 'speaker_label' if [ ... ] is detected, otherwise None.
    """
    chunks = []
    text_buffer = []  # accumulate text from sibling NavigableStrings

    for node in transcript_div.children:
        # Check if it's a <span> node
        if node.name == 'span':
            # First, if we have any text waiting in the buffer, push it as a text chunk
            buffered_text = "".join(text_buffer).strip()
            if buffered_text:
                # Check if text ends with "["
                extra_info = "text_ends_with_bracket" if buffered_text.endswith(' [') else None
                speaker_label = True if has_speaker_label(buffered_text) else None

                chunks.append((buffered_text, 'text', speaker_label, extra_info))
                text_buffer = []

            # Now handle the <span> as a separate chunk
            span_html = str(node)
            style_attr = node.get('style', '')
            color_type = detect_span_color_type(style_attr)
            speaker_label = True if has_speaker_label(span_html) else None

            chunks.append((span_html, 'span', speaker_label, color_type))

        else:
            # node could be NavigableString (plain text), or other inline tags (e.g., <br>)
            # Convert it to string and accumulate
            text_buffer.append(str(node))

    # After the loop, if there's leftover text in the buffer, add it
    leftover_text = "".join(text_buffer).strip()
    if leftover_text:
        extra_info = "text_ends_with_bracket" if buffered_text.endswith(' [') else "no_label"
        speaker_label = True if has_speaker_label(buffered_text) else "no_label"
        chunks.append((buffered_text, 'text', speaker_label, extra_info))
        
    return chunks

def count_misassigned_words(transcript_div):
    """
    Count how many words are misassigned based on modified speaker labels.
    """
    last_correct_speaker_label = None
    misassigned_count = 0   
    conflict_speaker_label = None  # Store the speaker label that triggered the conflict (if any)
    continue_again = False

    chunked_result = chunk_div_content(transcript_div)

    # Iterate over chunks and look for the first span with a speaker label or the first text with ending bracket
    for idx, (content, chunk_type, speaker_label, extra_info) in enumerate(chunked_result):
        
        if continue_again:
            continue_again = False
            continue
        
        # Check if we have a conflict
        if conflict_speaker_label:
            if chunk_type == 'text':
                # Count words until the conflict is resolved
                count, label = count_words_until_first_speaker(content)
                print(f"    Index:{idx} Added words:{count} Unmodified label:{label}")
                misassigned_count += count
                if label:
                    last_correct_speaker_label = get_last_speaker(content)
                    conflict_speaker_label = None
                    print(f"Index:{idx} Conflict solved found Unmodified 'speaker_label'\n")
                continue
            if chunk_type == 'span' and speaker_label:
                speaker = get_speaker_from_span(content)
                print(f"Index:{idx} Speaker:{speaker} | conflict_speaker_label:{conflict_speaker_label} | last_correct_speaker_label:{last_correct_speaker_label}")
                if speaker == conflict_speaker_label:
                    print(f"Index:{idx} Conflict solved same speaker \n")
                    conflict_speaker_label = None
                    continue
                elif speaker == last_correct_speaker_label:
                    print(f"Index:{idx} Conflict solved it was an comment inside a bigger segment of 'correct_speaker_label'\n")
                    conflict_speaker_label = None
                    continue
                else:
                    conflict_speaker_label = speaker
                continue

        # Look for the next correct_speaker_label or conflict
        else:
            # Look for ``last_correct_speaker_label``
            if chunk_type == 'text':
                if speaker_label:
                    last_correct_speaker_label = get_last_speaker(content)
                    #print(f"Index:{idx} Last correct speaker label:{last_correct_speaker_label}")
                # Conflict due probably to a subtitution of the speaker label
                if extra_info == 'text_ends_with_bracket':
                    print(f"Index:{idx} Conflict detected: 'text_ends_with_bracket")
                    speaker_pred , speaker_ref = extract_substitution_contents(chunked_result[idx+1][0])
                    print(f"    Speaker pred: {speaker_pred} | Speaker ref: {speaker_ref}")
                    conflict_speaker_label = speaker_pred
                    last_correct_speaker_label = speaker_ref
                    continue_again = True
                else:
                    continue
                
            # Look for modified speaker label conflict
            if chunk_type == 'span' and speaker_label:
                #print(f"Index:{idx} last_correct_speaker_label:{last_correct_speaker_label}")
                if extra_info in ("ins", "del"):
                    conflict_speaker_label = get_speaker_from_span(content)
                    print(f"Index:{idx} Conflict Ins/Del 'speaker_label': {conflict_speaker_label}")
                if extra_info == 'sub':
                    print(f"Index:{idx} Conflict Sustitution 'speaker_label': {conflict_speaker_label}")
                    pred , _ = extract_substitution_contents(content)
                    count, label = count_words_until_first_speaker(pred)
                    print(f"    Index:{idx} Added words:{count} Unmodified label:{label}")
                    misassigned_count += count
                    if label:
                        last_correct_speaker_label = get_last_speaker(content)
                        conflict_speaker_label = None
                        print(f"Index:{idx} Conflict solved just move some words with Sustitution'\n")

    print(f"Misassigned words: {misassigned_count}")
    return misassigned_count, chunked_result

def process_html(html_file, total_words_reference):
    """
    Process the HTML file to calculate the DER (Diarization Error Rate).
    
    :param html_file: Path to the HTML file containing the diarization transcript.
    :param total_words_reference: Reference count of total words (int).
    :return: DER (float) - Diarization Error Rate.
    """
    # Validate inputs
    if total_words_reference <= 0:
        raise ValueError("Total words in reference must be greater than zero.")

    
    # Open and read the HTML file
    with open(html_file, 'r', encoding='utf-8') as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    transcript_div = soup.find('div')

    # Check if transcript div was found
    if not transcript_div:
        raise ValueError("Unable to locate the transcript div in the HTML file.")

    # Count misassigned words
    misassigned_count, chunked_result = count_misassigned_words(transcript_div)

    # Calculate DER
    DER = misassigned_count / total_words_reference

    #print(f"Misassigned words: {misassigned_count}")
    print(f"Total words reference: {total_words_reference}")

    return DER, chunked_result

## Helper functions

def has_speaker_label(text):
    # True if there's at least one "[ ... ]" pattern
    return bool(re.search(r'\[[^]]*\]', text))

def detect_span_color_type(style_attr):
    """
    Determine the type ('sub', 'del', 'ins') of a span based on its background color.
    """
    style_attr = style_attr.lower() if style_attr else ""
    if 'yellow' in style_attr:
        return 'sub'
    elif 'lightcoral' in style_attr:
        return 'del'
    elif 'lightgreen' in style_attr:
        return 'ins'
    else:
        return 'ins'  # Default type for spans without a recognized color

def get_last_speaker(content):
    """
    Extracts the speaker inside the last set of square brackets [ ... ] in the content,
    removing spaces from the beginning and end of the content inside the brackets.

    :param content: A string containing text with speaker labels in [ ... ].
    :return: The last speaker label (string) inside the brackets, or None if no speaker label is found.
    """
    # Regular expression to match [ ... ]
    matches = re.findall(r'\[([^\]]+)\]', content)
    # If matches are found, strip spaces from the last match and return
    if matches:
        return matches[-1].strip()
    return None

def extract_substitution_contents(span_html):
    """
    Extracts the first content inside <s> and the second content after the arrow (→) in a span.

    :param span_html: A string containing the span HTML, e.g.,
                      '<span style="..."><s>Participant</s> → Interviewer 2</span>'
    :return: A tuple (first_content, second_content), or (None, None) if not found.
    """
    # Regular expression to capture <s>...</s> and the content after → up to the next tag or end of string
    match = re.search(r'<s>(.*?)</s>\s*→\s*([^<]+)', span_html)
    if match:
        first_content = match.group(1).strip()  # Content inside <s>
        second_content = match.group(2).strip()  # Content after →, stopping before the next tag
        return first_content, second_content
    return None, None

def count_words_until_first_speaker(content):
    """
    Counts the number of words in the content until the first speaker label ([ ... ]) is encountered.
    
    :param content: A string containing text with optional speaker labels in [ ... ].
    :return: The count of words before the first speaker label, or the total word count if no speaker label exists.
    :return: A boolean indicating whether a speaker label was found.
    """
    # Regex to match the first occurrence of [ ... ]
    speaker_match = re.search(r'\[[^]]*\]', content)
    speaker_label = False
    
    if speaker_match:
        # Extract the portion before the first speaker label
        before_speaker = content[:speaker_match.start()]
        speaker_label = True
    else:
        # If no speaker label exists, use the entire content
        before_speaker = content

    # Split the text into words and count them
    word_count = len(re.findall(r'\b\w+\b', before_speaker))
    return word_count, speaker_label

def get_speaker_from_span(span_html):
    """
    Extracts the speaker label content inside [ ... ] from the given span HTML.

    :param span_html: A string containing the span HTML, e.g.,
                      '<span style="background-color: lightgreen;">[ Participant ] :</span>'
    :return: The content inside [ ... ], or None if no speaker label is found.
    """
    # Regex to extract content inside [ ... ]
    match = re.search(r'\[([^\]]+)\]', span_html)
    if match:
        return match.group(1).strip()  # Return the captured group without leading/trailing spaces
    return None