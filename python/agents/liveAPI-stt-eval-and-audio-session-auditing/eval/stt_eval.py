import difflib
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
    sentence_ratios = []
    
    print("\n--- Per-Sentence Comparison ---")
    for i in range(min_length):
        gt = re.sub(r'[.,]', ' ', gt_sentences[i]).replace("'", "").lower()
        gt = re.sub(r'\s+', ' ', gt).strip()
        
        dt = re.sub(r'[.,]', ' ', detected_sentences[i]).replace("'", "").lower()
        dt = re.sub(r'\s+', ' ', dt).strip()
        
        sm = difflib.SequenceMatcher(None, gt.lower(), dt.lower())
        ratio = sm.ratio()
        sentence_ratios.append(ratio)
        
        print(f"\nSentence {i+1}:")
        print(f"  Ground Truth: {gt}")
        print(f"  Detected:     {dt}")
        print(f"  LCS Ratio:    {ratio:.4f}")

    if len(gt_sentences) != len(detected_sentences):
        print(f"\n[Warning] Length Mismatch: Ground Truth has {len(gt_sentences)} sentences, Detected has {len(detected_sentences)}.")

    avg_sentence_ratio = sum(sentence_ratios) / len(sentence_ratios) if sentence_ratios else 0
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS")
    print("=" * 50)
    print(f"Average Sentence LCS Ratio: {avg_sentence_ratio:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    main()
