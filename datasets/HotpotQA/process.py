import json

with open("/mnt/panruotong2021/Code/CAG/datasets/HotpotQA/HotpotQA_cred_qstart.json") as f:
    data = json.load(f)

for i in range(len(data)):
    prompt = data[i]["conversations"][0]['value']
    q, docs_prompt = prompt.split('\n', 1)
    data[i]["conversations"][0]['value'] = docs_prompt+"\n\n"+q+"\n"+"Answer:"
with open("/mnt/panruotong2021/Code/CAG/datasets/HotpotQA/HotpotQA_cred.json", "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)