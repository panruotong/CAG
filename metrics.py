import json
import string
import re
import numpy as np
import os

def load_jsonl(data_path):
    data = []
    with open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    outputs = [x['output'] for x in data]
    goldens = [x['golden'] for x in data]
    return outputs, goldens

def load_json(data_path):
    data = []
    with open(data_path, "r") as f:
        data = json.load(f)
    outputs = [x['output'] for x in data]
    goldens = [x['golden'] for x in data]
    return outputs, goldens

def eval(data_path):
    outputs, goldens = load_jsonl(data_path)
    
    if "llama-2" in data_path or "Llama-2" in data_path:
        
        for i in range(len(outputs)):
            outputs[i] = outputs[i].split('\n\n')[0]
    
    rets = {}
    count = 0
    for output, golden in zip(outputs, goldens):
        if golden in output:
            count += 1
    print({"exact_str_match": count / len(outputs)})
    #return 

def eval_trivia(data_path):
    start = 0
    end = 500
    outputs, goldens = load_jsonl(data_path)
    
    if "llama-2" in data_path or "Llama-2" in data_path:
        
        for i in range(len(outputs)):
            outputs[i] = outputs[i].split('\n')[0]
    
    count = 0
    for output, golden in zip(outputs, goldens):
        for golden_ in golden:
            if golden_ in output:
                count += 1
                break
    rets = {"exact_str_match": count/len(outputs)}
    print(rets)

def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def eval_evotemp(data_path):
    outputs, goldens = load_jsonl(data_path)

    if "llama-2" in data_path or "Llama-2" in data_path:
        for i in range(len(outputs)):
            outputs[i] = outputs[i].split('\n')[0]

    count = 0
    for output, golden in zip(outputs, goldens):
        golden_ = golden[0]
        found = True
        for g in golden_:
            if isinstance(g, list):
                for g_ in g:
                    if g_ not in output:
                        found = False
            else:
                if g not in output:
                    found = False
            if found:
                count += 1
                break

    rets = {"exact_str_match": count/len(outputs)}
    score_path = data_path[:-len(".json")] + "_score.json"
    with open(score_path, "w") as f:
        json.dump(rets, f, indent=4, ensure_ascii=False)
    print(rets)

def eval_PopQA(data_path):
    outputs, goldens = load_jsonl(data_path)
    count = 0
    for output, golden in zip(outputs, goldens):
        for g in golden:
            if g in output:
                count += 1
                break
    print({"exact_str_match": count / len(outputs)})

def checkanswer(prediction, ground_truth):
    prediction = prediction.lower()
    if type(ground_truth) is not list:
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if type(instance)  == list:
            flag = False 
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))
    return labels

def eval_rgb(data_path):
    outputs, goldens = load_jsonl(data_path)
    
    if "llama-2" in data_path or "Llama-2" in data_path:
        for i in range(len(outputs)):
            outputs[i] = outputs[i].split('\n')[0]
    
    count = 0
    for output, golden in zip(outputs, goldens):
        labels = checkanswer(output, golden)
        #print(labels)
        if 0 not in labels and 1 in labels:
            count += 1
    rets = {"exact_str_match": count/len(outputs)}
    print(rets)

def compute_exact_match(data_path, dataset_name):
    if dataset_name == "Trivia":
        return eval_trivia(data_path)
    elif dataset_name == "EvoTemp" or dataset_name == "EvoTempQBefore":
        return eval_evotemp(data_path)
    elif dataset_name == "PopQA":
        return eval_PopQA(data_path)
    elif dataset_name == "RGB":
        return eval_rgb(data_path)
    else:
        return eval(data_path)


def EM(model_type, type_):
    #/mnt/panruotong2021/Code/CAG/result-eval/HotpotQA/self_rag_concat_tmp0.01.json
    base_dir = "/mnt/panruotong2021/Code/CAG/result-eval/"
    #, "EvoTemp", "TriviaQA" "RGB"
    #for data_name in ["EvoTemp"]: NewsPolluted
    #EvoTempQBefore "HotpotQA", "2wikiMultiHopQA", "Musique", 
    #"EvoTemp", "NewsPolluted" 2wikiMultiHopQA
    for data_name in ["Musique"]:
        print(data_name)
        for file in os.listdir(f"{base_dir}{data_name}"):
            if type_ in file:
                print(file)
                compute_exact_match(os.path.join(f"{base_dir}{data_name}", file), data_name)
                print("\n")
        '''
        data_path = f"{base_dir}{data_name}/{model_type}_{type_}_tmp0.01.json"
        compute_exact_match(data_path, data_name)
        
        #0.2, 0.4, 0.6, 0.8
        #0.5, 0.67, 0.75
        for noise_ratio in [0.2, 0.4, 0.6, 0.8]:
            data_path = f"{base_dir}{data_name}/{model_type}_{type_}_tmp0.01_noise_ratio{noise_ratio}.json"
            compute_exact_match(data_path, data_name)
        '''
        
if __name__ == "__main__":
    for type_ in [model_type]:
        print(type_)
        EM(model_type, type_)