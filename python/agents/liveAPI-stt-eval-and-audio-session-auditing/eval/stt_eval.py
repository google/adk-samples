import glob
import os
import re
from pathlib import Path

def parse_transcript(file_path):
    """Parses a transcript file, combining sequential USER tags into individual sentences."""
    sentences = []
    current_sentence_parts = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Match tags (USER: or MODEL:)
            match_user = re.match(r"^USER:\s*(.*)$", line, re.IGNORECASE)
            match_model = re.match(r"^MODEL:\s*(.*)$", line, re.IGNORECASE)
            
            if match_user:
                text = match_user.group(1).strip()
                current_sentence_parts.append(text)
            elif match_model:
                if current_sentence_parts:
                    # Combine sequential USER parts and save
                    sentences.append(" ".join(current_sentence_parts))
                    current_sentence_parts = []
                    
        # Append final USER sentence if file ends with USER input
        if current_sentence_parts:
            sentences.append(" ".join(current_sentence_parts))
            
    return sentences

def parse_ground_truth(file_path):
    """Parses STT ground truth file, treating each non-empty line as a sentence."""
    sentences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)
    return sentences

def calculate_wer(ref, hyp):
    """Calculates Word Error Rate (WER) using Levenshtein distance at the word level."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    
    n = len(ref_words)
    m = len(hyp_words)
    
    if n == 0:
        return float(m)  # All words inserted
        
    # Create Levenshtein matrix
    d = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                substitution = d[i - 1][j - 1] + 1
                deletion = d[i - 1][j] + 1
                insertion = d[i][j - 1] + 1
                d[i][j] = min(substitution, deletion, insertion)
                
    return d[n][m] / n

def main():
    eval_dir = Path(__file__).parent.resolve()
    base_dir = eval_dir.parent
    
    ground_truth_path = eval_dir / "STT_ground_truth.txt"
    if not ground_truth_path.exists():
        print(f"Error: Ground truth file not found at {ground_truth_path}")
        return

    # Find the most recent transcript in the transcripts folder
    transcripts_dir = base_dir / "transcripts"
    transcript_files = glob.glob(str(transcripts_dir / "*.txt"))
    
    if not transcript_files:
        print(f"Error: No transcripts found in {transcripts_dir}")
        return
        
    latest_transcript = max(transcript_files, key=os.path.getmtime)
    print(f"Evaluating Transcript: {Path(latest_transcript).name}")
    print(f"Against Ground Truth: {ground_truth_path.name}")
    print("-" * 50)

    gt_sentences = parse_ground_truth(ground_truth_path)
    detected_sentences = parse_transcript(latest_transcript)
    
    # Calculate Per-Sentence Ratios
    min_length = min(len(gt_sentences), len(detected_sentences))
    sentence_wers = []
    
    print("\n--- Per-Sentence Comparison (Word Error Rate) ---")
    for i in range(min_length):
        # Normalize inputs: lowercasing and removing punctuation
        gt = re.sub(r'[.,;?!:]', ' ', gt_sentences[i]).replace("'", "").lower()
        gt = re.sub(r'\s+', ' ', gt).strip()
        
        dt = re.sub(r'[.,;?!:]', ' ', detected_sentences[i]).replace("'", "").lower()
        dt = re.sub(r'\s+', ' ', dt).strip()
        
        wer = calculate_wer(gt, dt)
        sentence_wers.append(wer)
        
        print(f"\nSentence {i+1}:")
        print(f"  Ground Truth: {gt}")
        print(f"  Detected:     {dt}")
        print(f"  Word Error Rate (WER): {wer:.4f} ({wer * 100:.2f}%)")

    if len(gt_sentences) != len(detected_sentences):
        print(f"\n[Warning] Length Mismatch: Ground Truth has {len(gt_sentences)} sentences, Detected has {len(detected_sentences)}.")

    avg_wer = sum(sentence_wers) / len(sentence_wers) if sentence_wers else 0
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Average Sentence WER: {avg_wer:.4f} ({avg_wer * 100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
