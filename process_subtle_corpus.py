from pathlib import Path 
from tqdm import tqdm 
import json 
import math 

corpus_path = 'data/corpus0sDialogues.txt' 

with Path(corpus_path).open('r') as f: 
    corpus = f.readlines() 

json_dict = []  
idx = 0 
for l in tqdm(corpus): 
    if l[:4] == "I - ": 
        p = l[4:]
    if l[:4] == "R - ": 
        r = l[4:]
        json_dict.append({
            'idx': idx,
            'p': p.replace('\n', ''), 
            'r': r.replace('\n', '')
        })
        idx += 1

Path('data/subtle_corpus').mkdir(parents=True, exist_ok=True)

for idx in range(20): 
    interval = math.ceil(len(json_dict)/20)
    json_path = f'data/subtle_corpus/subtle_corpus{idx}.json'
    with Path(json_path).open('w') as f: 
        json.dump(json_dict[interval*idx:interval*(idx+1)], f, indent=4, sort_keys=True)

