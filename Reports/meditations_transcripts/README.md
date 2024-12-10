# Dataset of verbal reports from cognitive neuroscience research project on meditation (LNCO)

This dataset includes transcriptions of semi-structured interviews from three distinct experiments: **OBE1**, **OBE2**, and **Compassion**. Each experiment utilized a mixed-reality immersive environment, combining virtual reality (VR) and meditation experiences under varying conditions. Details of the experiments can be found [here](#overview-experiments).

# Folders & Files

`audio/` 
Contains the original audio files of the interviews.

`raw/` 
Contains the raw transcripts output by the transcription model.
*(Link to repository to come)*

`transcripts_per_interviews/` 
Contains the transcripts after preprocessing & manual verification.
- Preprocessing Steps:
    - Removal of filler words and repetitions.
    - Visual cleaning of the text.
    - Prediction of speaker roles.
- Manual verification:
    - Correct errors of transcription & diariazation.
    - Verify role assignments.
    - Cut unrelated sections.
- Available in both text and CSV formats.

`overview_interviews.csv` 
Provides an overview of all the interviews conducted across experiments.

`transcripts_merged.csv` 
Contains all interview transcripts merged into a single CSV file for analysis.

# CSV Structure

Each row in the dataset corresponds to a speech segment, as captured by the transcription model. This structure help to accounts for the speaker's pacing.

## Headers Explanation:

- **Experiment:** Name of the experiment to which the interview belongs (e.g., OBE1, OBE2, Compassion).
- **File Name:** The unique identifier for the file containing the interview data.
- **Id:** Participant ID associated with the interview.
- **Content Type:** Type of the orignal content (for this dataset, it is always "Audio").
- **Start Time:** Start timestamp of the speech segment (in HH:MM:SS,SSS format).
- **End Time:** End timestamp of the speech segment (in HH:MM:SS,SSS format).
- **Speaker:** The role for the speaker, such as "Participant" or "Interviewer."  
    - If multiple interviewers are present, they are labeled with a numerical suffix (e.g., Interviewer 1, Interviewer 2).
    - *Note:* Interviewer labels are arbitrary within each interview and do not necessarily represent the same individual across different interviews.
- **Content:** Transcription of the speech segment.

In this dataset I also manually add:
- **Condition:** The experimental condition for the interview.
    - *C*: Control condition.
    - *I*: Intervention condition.
    - *1*: Only one interview conducted for the participant.
    - *0*: No "real" interview (e.g., setup phase, small talk).
- **Order Condition:** Represents the sequence in which the participant performed the conditions.
    - IC: Intervention (I) first, followed by Control (C).s
    - Unknown: The order is unavailable.

### Note for conditions

For most participants, the sequence of conditions is available, enabling the determination of the specific condition corresponding to each interview.

However, for participants with only one interview, it is not possible to determine whether the interview was conducted after the first or second condition based solely on the *"Order Condition"* column.  
While manual review of the interview content can sometimes provide context, participants and interviewers often compare both conditions during their exchanges. This overlap makes it difficult to attribute content to a single condition and raises doubts about the reliability of automated text analysis for comparing conditions across interviews.

As a result, cases with only one interview are labeled as **1**, indicating that the condition remains ambiguous. Furthermore, even interviews labeled as **C** (Control) or **I** (Intervention) may contain references to both conditions, complicating direct comparisons between them. *(Nonetheless, it may be worth exploring this further.)*


# Overview Experiments

All experiments were conducted using virtual reality (VR) environments and involved two conditions for each participant:
- Intervention (*I*)
- Control (*C*)

More visual details and experimental design are available here: [Linked presentation](https://docs.google.com/presentation/d/1ODSBcryrDgOaYXnXrpVGVIpXwxi28-_G/edit?usp=sharing&ouid=102524386561627991544&rtpof=true&sd=true) & *"paper to come maybe"*

## Out-of-Body Experience (*OBE1* & *OBE2*)
Participants meditated in a VR environment guided by a "virtual assistant/teacher" and surrounded by a forest setting.

- Intervention Condition (I): Participants experienced an out-of-body experience (OBE).
- Control Condition (C): Participants engaged in a "normal" meditation without an OBE.

### OBE2 Experiment

An enhanced version of OBE1 that includes **breathing biofeedback** to improve the meditation experience:

- Participants observed their virtual body "blinking" in synchrony (sync) or out of synchrony (async) with their breathing.
- The sync/async breathing feedback was not tied to the intervention or control conditions but alternated during the session.

## Self-Compassion (*Compassion*)

- Control Condition (C): Participants remained in the VR environment with their eyes closed for most of the session.
- Intervention Condition (I): Participants experienced a *virtual autoscopy* (Self view as another) with *cardio-visual synchrony*.

# Data overviews

## Distribution of interviews across experiments

![experiments](./plots/interviews_by_participant.png)

## Conditions 
C : Control , I : Intervention, 1 : Only one interview conducted for the participant, 0 : No interview (eg. Set-up)

![conditions](./plots/stripplot_word_count_id.png)
Connected points represent the same participant (ID).
