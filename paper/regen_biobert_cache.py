# -*- coding: utf-8 -*-
"""
regen_biobert_cache.py — regenerate HINT's empty criteria-embedding cache (defect D5).

The shipped hint/data/sentence2embedding.pkl is an empty 6-byte dict, so HINT's
eligibility-criteria branch returns a zero 768-vector for every trial (defect D5). This script
repopulates it using the paper's stated encoder (dmis-lab/biobert-v1.1 via transformers), reusing
HINT's own sentence-collection function so the keys match exactly what protocol2feature looks up.

Run from the HINT checkout root (needs hint/HINT/protocol_encode.py on the path):
    cd ~/ctp/hint && python regen_biobert_cache.py

Result: hint/data/sentence2embedding.pkl becomes {sentence: torch.FloatTensor(768)} (~1.5 GB).
Do NOT commit that file; the repo keeps the 6-byte empty original as the D5 evidence. Restore it
after the D5 experiment with:  cp data/sentence2embedding.EMPTY.pkl data/sentence2embedding.pkl

Measured effect (Section "Defects", D5): re-scoring the released HINT checkpoint with the live
cache changes phase-III ROC-AUC by -0.0001 (p=0.93) and PR-AUC by +0.0002 (p=0.78) -- a genuine
defect with no measurable effect on HINT's numbers.
"""
import sys, pickle, torch
sys.path.insert(0, 'HINT')
from protocol_encode import collect_cleaned_sentence_set          # HINT's own sentence set
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

tok = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1').eval()

@torch.no_grad()
def embed(text):
    enc = tok(text, return_tensors='pt', truncation=True, max_length=64)
    return model(**enc).last_hidden_state[0].mean(0).float()       # (768,) mean-pool

def main():
    sents = collect_cleaned_sentence_set()
    print(f"embedding {len(sents)} unique sentences...")
    d = {}
    for s in tqdm(sents):
        try: d[s] = embed(s)
        except Exception: d[s] = torch.zeros(768)
    pickle.dump(d, open('data/sentence2embedding.pkl', 'wb'))
    print("cache len:", len(d))

if __name__ == "__main__":
    main()
