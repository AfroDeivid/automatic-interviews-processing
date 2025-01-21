import re
import os
import csv

from sacrebleu.metrics import BLEU, CHRF

# Function to split text into sentences
def split_into_sentences(text):
    # Use regex to split by sentence-ending punctuation
    return re.split(r'(?<=[.!?])\s+', text.strip())

# Function to process files row by row and handle mismatches
def process_rows(reference_file, hypothesis_file):
    # Read rows from reference and hypothesis files
    with open(reference_file, 'r', encoding='utf-8') as ref_file:
        ref_reader = csv.DictReader(ref_file)
        ref_rows = [row["Text"] for row in ref_reader]
    
    with open(hypothesis_file, 'r', encoding='utf-8') as hyp_file:
        hyp_reader = csv.DictReader(hyp_file)
        hyp_rows = [row["Text"] for row in hyp_reader]

    # Initialize containers for aligned sentences
    aligned_references = []
    aligned_hypotheses = []

    # Process each row
    for ref, hyp in zip(ref_rows, hyp_rows):
        # Split rows into sentences
        ref_sentences = split_into_sentences(ref)
        hyp_sentences = split_into_sentences(hyp)

        # Check if sentence counts match
        if len(ref_sentences) == len(hyp_sentences):
            # Append sentences individually if counts match
            aligned_references.extend(ref_sentences)
            aligned_hypotheses.extend(hyp_sentences)
        else:
            # Concatenate all sentences in the row into a single "big sentence"
            aligned_references.append(" ".join(ref_sentences))
            aligned_hypotheses.append(" ".join(hyp_sentences))

    return aligned_references, aligned_hypotheses

def evaluate_metric_by_filename(reference_folder, hypothesis_folder, scorer_bleurt=None):
    """
    Evaluate translations by processing reference and hypothesis files directly from folders.

    Parameters:
    - reference_folder (str): Path to the folder containing reference files.
    - hypothesis_folder (str): Path to the folder containing hypothesis files.

    Returns:
    - A dictionary with per-file scores for METEOR and BLEU.
    """
    # Find all files in both folders
    reference_files = {
        os.path.basename(f).replace(" English", ""): os.path.join(reference_folder, f)
        for f in os.listdir(reference_folder)
    }
    hypothesis_files = {
        os.path.basename(f): os.path.join(hypothesis_folder, f)
        for f in os.listdir(hypothesis_folder)
    }

    # Ensure matching filenames exist in both folders
    common_files = set(reference_files.keys()) & set(hypothesis_files.keys())
    if not common_files:
        raise ValueError("No matching files found in the reference and hypothesis folders.")

    # Initialize per-file scores
    file_scores = {}

    # Process each matching file
    for filename in common_files:
        #print(f"Processing: {filename}")

        ref_sentences, hyp_sentences = process_rows(reference_files[filename], hypothesis_files[filename])
        
        # Compute metrics
        bleu = BLEU()
        bleu_info = bleu.corpus_score(hyp_sentences, [ref_sentences])

        chrf = CHRF()
        chrf_info = chrf.corpus_score(hyp_sentences, [ref_sentences])

        if scorer_bleurt:
            bleurt_info = compute_bleurt_corpus(ref_sentences, hyp_sentences, scorer_bleurt)
        else:
            bleurt_info = None

        #meteor = compute_meteor(ref_sentences, hyp_sentences)

        # Save per-file scores
        file_scores[filename] = {
            "BLEU": bleu_info,
            "CHRF": chrf_info,
            "BLEURT": bleurt_info
        }

    return file_scores

def evaluate_metrics_aggregated(reference_folder, hypothesis_folder, scorer_bleurt=None):
    """
    Evaluate translations by processing reference and hypothesis files directly from folders.
    Computes aggregate metrics across all sentences instead of per-file scores.

    Parameters:
    - reference_folder (str): Path to the folder containing reference files.
    - hypothesis_folder (str): Path to the folder containing hypothesis files.
    - scorer_bleurt (BleurtScorer, optional): BLEURT scorer instance for BLEURT evaluation.

    Returns:
    - A dictionary with aggregated scores for BLEU, CHRF, and BLEURT.
    """

    # Find all files in both folders
    reference_files = {
        os.path.basename(f).replace(" English", ""): os.path.join(reference_folder, f)
        for f in os.listdir(reference_folder)
    }
    hypothesis_files = {
        os.path.basename(f): os.path.join(hypothesis_folder, f)
        for f in os.listdir(hypothesis_folder)
    }

    # Ensure matching filenames exist in both folders
    common_files = set(reference_files.keys()) & set(hypothesis_files.keys())
    if not common_files:
        raise ValueError("No matching files found in the reference and hypothesis folders.")

    # Initialize lists to store all sentences
    all_ref_sentences = []
    all_hyp_sentences = []

    # Process each matching file
    for filename in common_files:
        #print(f"Processing: {filename}")

        ref_sentences, hyp_sentences = process_rows(reference_files[filename], hypothesis_files[filename])
        
        all_ref_sentences.extend(ref_sentences)
        all_hyp_sentences.extend(hyp_sentences)

    print("Number of sentences:", len(all_ref_sentences))
    # Compute metrics on the aggregated sentences
    bleu = BLEU()
    bleu_info = bleu.corpus_score(all_hyp_sentences, [all_ref_sentences])

    chrf = CHRF()
    chrf_info = chrf.corpus_score(all_hyp_sentences, [all_ref_sentences])

    if scorer_bleurt:
        bleurt_info = compute_bleurt_corpus(all_ref_sentences, all_hyp_sentences, scorer_bleurt)
    else:
        bleurt_info = None

    # Return aggregate scores
    aggregated_scores = {
        "BLEU": bleu_info,
        "CHRF": chrf_info,
        "BLEURT": bleurt_info
    }

    return aggregated_scores


def compute_bleurt_corpus(references, hypotheses, scorer):
    """
    Computes BLEURT scores for a list of reference-hypothesis pairs and averages the scores for the whole corpus.
    
    Args:
        references (list of str): A list of reference sentences.
        hypotheses (list of str): A list of hypothesis sentences.
        scorer (BleurtScorer): A BLEURT scorer object initialized with a checkpoint.
    
    Returns:
        dict: A dictionary with sentence-level scores and the average score for the corpus.
    """
    if len(references) != len(hypotheses):
        raise ValueError("The number of references and hypotheses must be the same.")
    
    # Compute BLEURT scores for all sentence pairs
    scores = scorer.score(references=references, candidates=hypotheses)
    
    # Calculate the average score for the corpus
    average_score = sum(scores) / len(scores)
    
    return {
        "sentence_scores": scores,
        "average_score": average_score
    }