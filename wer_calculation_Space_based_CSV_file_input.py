# Updated script with the tokenizer changed to match manual (space-based) word counting:
import numpy as np
import regex as re
import csv

# Updated tokenizer: Split by spaces (manual-like word count)
def tokenize_sinhala(text):
    return text.strip().split()

def wer(reference, hypothesis):
    r = tokenize_sinhala(reference)  # Tokenized reference
    h = tokenize_sinhala(hypothesis)  # Tokenized hypothesis
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)

    for i in range(len(r)+1):
        d[i][0] = i
    for j in range(len(h)+1):
        d[0][j] = j

    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            if r[i-1] == h[j-1]:
                cost = 0
            else:
                cost = 1
            d[i][j] = min(
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + cost
            )

    i, j = len(r), len(h)
    substitutions = deletions = insertions = 0

    while i > 0 and j > 0:
        if r[i-1] == h[j-1]:
            i -= 1
            j -= 1
        elif d[i][j] == d[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif d[i][j] == d[i-1][j] + 1:
            deletions += 1
            i -= 1
        elif d[i][j] == d[i][j-1] + 1:
            insertions += 1
            j -= 1

    while i > 0:
        deletions += 1
        i -= 1
    while j > 0:
        insertions += 1
        j -= 1

    errors = substitutions + deletions + insertions
    return errors, len(r), substitutions, deletions, insertions

def overall_wer(reference_list, hypothesis_list):
    total_errors = total_words = total_subs = total_dels = total_ins = 0

    for ref, hyp in zip(reference_list, hypothesis_list):
        errors, words, subs, dels, ins = wer(ref, hyp)
        total_errors += errors
        total_words += words
        total_subs += subs
        total_dels += dels
        total_ins += ins

    wer_value = total_errors / total_words if total_words > 0 else 0.0
    return wer_value, total_words, total_subs, total_dels, total_ins

def read_csv_input(file_path):
    references = []
    hypotheses = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if 'reference' in row and 'hypothesis' in row:
                references.append(row['reference'])
                hypotheses.append(row['hypothesis'])
    return references, hypotheses

def write_csv_output(file_path, wer_score, total_words, subs, dels, ins):
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['WER', f"{wer_score:.2%}"])
        writer.writerow(['Total Words', total_words])
        writer.writerow(['Substitutions', subs])
        writer.writerow(['Deletions', dels])
        writer.writerow(['Insertions', ins])

if __name__ == "__main__":
    input_file = 'input.csv'
    output_file = 'output_metrics.csv'

    refs, hyps = read_csv_input(input_file)
    wer_score, total_words, subs, dels, ins = overall_wer(refs, hyps)

    print("\n--- Sinhala Word Error Rate Summary ---")
    print(f"Overall WER: {wer_score:.2%}")
    print(f"Total reference words: {total_words}")
    print(f"Substitutions: {subs}")
    print(f"Deletions: {dels}")
    print(f"Insertions: {ins}")

    write_csv_output(output_file, wer_score, total_words, subs, dels, ins)
