# File Structure

**Raw:** Contains individual transcripts for each file without any further preprocessing.

**Text** Provides merged transcripts in text (.txt) format for easy manual revision.
- Combining multiple recordings of the same interview into a single file (e.g., if there are two videos of the same interview, both transcripts are combined into one CSV).
- Without any preprocessing except for the assigned speaker roles (labeled as Interviewer and Participant) & visual cleaning.

**Note:** Since these files include separate recordings, the model may assign different speaker labels across files (e.g., identifying the interviewer as "0" in one recording and "1" in another). However, the model generally maintains accurate speaker distinctions. --- This issue may be resolved in the **Text** files by the speaker role predictions.

### Additional Notes
If there are any issues or suggestions regarding the format, please let me know, as I can adapted ;)

Best,
David Friou