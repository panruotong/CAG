import json
import torch
from transformers import (
    GenerationConfig, 
    LlamaForCausalLM, 
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse
import requests
from tqdm import tqdm
from fastchat.model import load_model
from fastchat.conversation import SeparatorStyle, get_conv_template
import os
from torch.utils.data import Dataset
from transformers.trainer_pt_utils import LabelSmoother
from typing import List
from transformers import PreTrainedTokenizer
from typing import Dict, Optional, Sequence
from datasets.base_dataset import BaseDataset
from metrics import compute_exact_match
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
from torch.utils.data import DataLoader
from eval_chatgpt import evalChatgpt
IGNORE_TOKEN_ID = LabelSmoother.ignore_index

def load_data(eval_data_path, dataset, type_):
    #if dataset == "EvoTemp":
    #    dataset = "EvoTempQBefore"
    print(dataset)
    print(type_)
    dataset_path = os.path.join(eval_data_path, dataset)
    for file in os.listdir(dataset_path):
        if type_ in file:
            print(file)
            with open(os.path.join(dataset_path, file), "r") as f:
                eval_data = json.load(f)
            break
    return eval_data

def load_model_tokenizer(args):
    max_new_tokens = 512
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto"
            )
    return model, tokenizer

def get_eval_dataset_list(args):
    data_list = []
    if args.wikimulti:
        data_list.append("2wikiMultiHopQA")
    if args.hotpot:
        data_list.append("HotpotQA")
    if args.musique:
        data_list.append("Musique")
    if args.rgb:
        data_list.append("RGB")
    if args.evotemp:
        data_list.append("EvoTemp")
    if args.misinfo:
        data_list.append("NewsPolluted")
    return data_list

def get_system_prompt(setting_type):
    if "cred" in setting_type:
        return "You are an assistant who can answer questions based on the given passages. Each passage has a credibility score that indicates the relevance and accuracy of the passage to the question. Your answer need to combine multiple passages and their credibility."
    else:
        return "You're a helpful AI assistant. The assistant answers questions based on given passages.\n"

    
def infer_vllm(model, model_type, eval_data, batch_size, fw, system):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.01, top_p=1, max_tokens=100)
    rets = []
    for i in tqdm(range(0, len(eval_data), batch_size)):
        batched_inp = []
        responses = []
        for sample in eval_data[i: i + batch_size]:
            if "vicuna" in model_type or "vanilla" in model_type:
                conv = get_conv_template("vicuna_v1.1")
                conv.append_message(conv.roles[0], sample["conversations"][0]["value"])
            elif "mistral" in model_type and "instruct" not in model_type:
                conv = get_conv_template("mistral")
                conv.append_message(conv.roles[0], system + sample["conversations"][0]["value"])
            elif "mistral" in model_type and "instruct" in model_type:
                conv = get_conv_template("mistral")
                conv.append_message(conv.roles[0], system + sample["conversations"][0]["value"])
            conv.append_message(conv.roles[1], None)
            responses.append({"golden": sample["conversations"][1]["value"]})
            prompt = conv.get_prompt()
            batched_inp.append(prompt)
        try:
            outputs = model.generate(
                batched_inp, 
                sampling_params,
                use_tqdm=False
            )
        except ValueError:
            continue
        for output, response in zip(outputs, responses):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            response.update({"output": generated_text.strip()})
            fw.write(json.dumps(response, ensure_ascii=False)+"\n")

def infer_lm_vllm(model, tokenizer, model_type, eval_data, shots, batch_size, f, system):
    from vllm import SamplingParams
    sampling_params = SamplingParams(temperature=0.01, top_p=1, max_tokens=512)
    rets = []
    demo = shots
    
    for i in tqdm(range(0, len(eval_data), batch_size)):
        batched_inp = []
        responses = []
        for sample in eval_data[i: i + batch_size]:
            prompt = system + demo + "\n\n" + sample["conversations"][0]["value"]
            responses.append({"golden": sample["conversations"][1]["value"]})
            if len(tokenizer.tokenize(prompt)) > 4096:
                prompt_length = 4096 - len(tokenizer.tokenize(system+demo+"\n\n"))-2
                input_ids = tokenizer.encode(sample["conversations"][0]["value"], max_length=prompt_length, truncation=True, truncation_strategy='only_first')
                truncated_conversation = tokenizer.decode(input_ids)
                prompt = system + demo + "\n\n" + truncated_conversation
                
            batched_inp.append(prompt)
        try:
            outputs = model.generate(
                batched_inp, 
                sampling_params,
                use_tqdm=False
            )
        except ValueError:
            continue
        for output, response in zip(outputs, responses):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            response.update({"output": generated_text.strip()})
            f.write(json.dumps(response, ensure_ascii=False)+"\n")

def eval_chatgpt(args):
    data_list = get_eval_dataset_list(args)
    for data_name in data_list:
        if data_name == "EvoTemp":
            for noise_ratio in [0.4, 0.6, 0.8]:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    evalChatgpt(eval_data, args.model_type, args.setting_type, args.temperature, f)
        elif data_name == "NewsPolluted":
            for noise_ratio in [0.5, 0.67, 0.75]:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    evalChatgpt(eval_data, args.model_type, args.setting_type, args.temperature, f)
        elif data_name == "RGB":
            for noise_ratio in [0.2, 0.4, 0.6, 0.8]:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    evalChatgpt(eval_data, args.model_type, args.setting_type, args.temperature, f)
        else:
            eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart")
            with open(os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}.json"), "w") as f:
                evalChatgpt(eval_data, args.model_type, args.setting_type, args.temperature, f)

def eval_vllm(args):
    from vllm import LLM
    data_list = get_eval_dataset_list(args)
    if any(model_type in args.model_type for model_type in ["llama-2", "vicuna", "CAG", "vanilla"]):
        max_model_length = 4096
        model = LLM(model=args.model_path, max_num_batched_tokens=max_model_length, tensor_parallel_size=args.parallel_size)
    elif "mistral" in args.model_type:
        model = LLM(model=args.model_path, tensor_parallel_size=args.parallel_size)

    is_lm = args.is_lm
    batch_size = args.batch_size
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        padding_side="left"
    )
    system_prompt = get_system_prompt(args.setting_type)
    for data_name in data_list:
        if is_lm:
            with open(os.path.join('./prompt', f'{data_name}.txt')) as fshots:
                shots = fshots.read()
                
        if data_name == "EvoTemp":
            for noise_ratio in [0.4, 0.6, 0.8]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        responses = infer_lm_vllm(model, tokenizer, args.model_type, eval_data, shots, batch_size, f, system_prompt)
                    else:
                        responses = infer_vllm(model, args.model_type, eval_data, batch_size, f, system_prompt)
                compute_exact_match(output_path, data_name)
        elif data_name == "RGB":
            #for noise_ratio in [0.2, 0.4, 0.6, 0.8]:
            for noise_ratio in [0.4]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        responses = infer_lm_vllm(model, tokenizer, args.model_type, eval_data, shots, batch_size, f, system_prompt)
                    else:
                        responses = infer_vllm(model, args.model_type, eval_data, batch_size, f, system_prompt)
                compute_exact_match(output_path, data_name)
        elif data_name == "NewsPolluted":
            for noise_ratio in [0.5, 0.67, 0.75]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        responses = infer_lm_vllm(model, tokenizer, args.model_type, eval_data, shots, batch_size, f, system_prompt)
                    else:
                        responses = infer_vllm(model, args.model_type, eval_data, batch_size, f, system_prompt)
                compute_exact_match(output_path, data_name)
        else:
            if is_lm:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}.json")
            else:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart")
            
            output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}.json")
            with open(output_path, "w") as f:
                if is_lm:
                    responses = infer_lm_vllm(model, tokenizer, args.model_type, eval_data, shots, batch_size, f, system_prompt)
                else:
                    responses = infer_vllm(model, args.model_type, eval_data, batch_size, f, system_prompt)
        compute_exact_match(output_path, data_name)

def infer_lm(temperature, max_new_tokens, eval_data, shots, tokenizer, model, model_type, f, system):
    rets = []
    for item in tqdm(eval_data):
        if "CAG" in model_type:
            conv = get_conv_template("CAG")
            conv.append_message(conv.roles[0], item["conversations"][0]["value"])
        elif "llama-2" in model_type:
            demo = shots
            prompt = system + demo + "\n\n" + item["conversations"][0]["value"]
        else:
            conv = get_conv_template("vicuna_v1.1")
            conv.append_message(conv.roles[0], system+"\n"+item["conversations"][0]["value"])
        golden = item["conversations"][1]["value"]  
        
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        if len(input_ids)>4096:
            input_ids = input_ids[:4096]
        input_ids = input_ids.to(device)
        try:
            output_ids = model.generate(input_ids, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            continue
        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(output_ids)
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        f.write(json.dumps({"output": output, "golden": golden}, ensure_ascii=False)+"\n")

def infer(temperature, max_new_tokens, eval_data, tokenizer, model, model_type, f, system):
    rets = []
    for item in tqdm(eval_data):
        if "CAG" in model_type:
            conv = get_conv_template("CAG")
            conv.append_message(conv.roles[0], item["conversations"][0]["value"])
        elif "vicuna" in model_type or "vanilla" in model_type:
            conv = get_conv_template("vicuna_v1.1")
            conv.append_message(conv.roles[0], system+"\n"+item["conversations"][0]["value"])
        elif "mistral" in model_type and "instruct" not in model_type:
            conv = get_conv_template("mistral")
            conv.append_message(conv.roles[0], system+"\n"+item["conversations"][0]["value"])
        elif "mistral" in model_type and "instruct" in model_type:
            conv = get_conv_template("mistral")
            conv.append_message(conv.roles[0], system + sample["conversations"][0]["value"])
        golden = item["conversations"][1]["value"]
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()
        input_ids = tokenizer([prompt], return_tensors="pt").input_ids
        if len(input_ids)>4096:
            input_ids = input_ids[:4096]
        input_ids = input_ids.to(device)
        try:
            output_ids = model.generate(input_ids, do_sample=True, temperature=temperature, max_new_tokens=max_new_tokens)
        except torch.cuda.OutOfMemoryError:
            continue
        output_ids = output_ids[0][len(input_ids[0]):]
        output = tokenizer.decode(output_ids)
        for special_token in tokenizer.special_tokens_map.values():
            if isinstance(special_token, list):
                for special_tok in special_token:
                    output = output.replace(special_tok, "")
            else:
                output = output.replace(special_token, "")
        output = output.strip()
        f.write(json.dumps({"output": output, "golden": golden}, ensure_ascii=False)+"\n")

def eval(args):
    model, tokenizer = load_model_tokenizer(args)
    is_lm = args.is_lm
    data_list = get_eval_dataset_list(args)
    system_prompt = get_system_prompt(args.setting_type)
    for data_name in data_list:
        if data_name == "EvoTemp":
            for noise_ratio in [0.4, 0.6, 0.8]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                            shots = f_shot.read()
                        infer_lm(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model, args.model_type, f, system_prompt)
                    else:
                        infer(args.temperature, args.max_new_tokens, eval_data, tokenizer, model, args.model_type, f, system_prompt)
                compute_exact_match(output_path, data_name)
        elif data_name == "NewsPolluted":
            for noise_ratio in [0.5, 0.67, 0.75]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                            shots = f_shot.read()
                        infer_lm(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model, args.model_type, f, system_prompt)
                    else:
                        infer(args.temperature, args.max_new_tokens, eval_data, tokenizer, model, args.model_type, f, system_prompt)
                compute_exact_match(output_path, data_name)
        elif data_name == "RGB":
            for noise_ratio in [0.2, 0.4, 0.6, 0.8]:
                if is_lm:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_noise_ratio{noise_ratio}")
                else:
                    eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart_noise_ratio{noise_ratio}")
                output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}_noise_ratio{noise_ratio}.json")
                with open(output_path, "w") as f:
                    if is_lm:
                        with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                            shots = f_shot.read()
                        infer_lm(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model, args.model_type, f, system_prompt)
                    else:
                        infer(args.temperature, args.max_new_tokens, eval_data, tokenizer, model, args.model_type, f, system_prompt)
                compute_exact_match(output_path, data_name)
        else:
            if is_lm:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}.json")
            else:
                eval_data = load_data(args.data_path, data_name, f"{args.setting_type}_qstart")
            output_path = os.path.join("./result", data_name, f"{args.model_type}_{args.save_suffix}_tmp{args.temperature}.json")
            with open(output_path, "w") as f:
                if is_lm:
                    with open(f'./prompt/{data_name}.txt', 'r') as f_shot:
                        shots = f_shot.read()
                    infer_lm(args.temperature, args.max_new_tokens, eval_data, shots, tokenizer, model, args.model_type, f, system_prompt)
                else:
                    infer(args.temperature, args.max_new_tokens, eval_data, tokenizer, model, args.model_type, f, system_prompt)
            compute_exact_match(output_path, data_name)
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-type", type=str, required=True)
    parser.add_argument("--save-suffix", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--setting-type", type=str, required=True)
    parser.add_argument("--wikimulti", action='store_true')
    parser.add_argument("--hotpot", action='store_true')
    parser.add_argument("--musique", action='store_true')
    parser.add_argument("--wikiqa", action='store_true')
    parser.add_argument("--rgb", action='store_true')
    parser.add_argument("--evotemp", action='store_true')
    parser.add_argument("--misinfo", action='store_true')
    parser.add_argument("--is_lm", action='store_true')
    parser.add_argument("--vllm", action='store_true')
    parser.add_argument("--parallel_size", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parser()
    if "gpt" in args.model_type:
        eval_chatgpt(args)
    else:
        if args.vllm:
            eval_vllm(args)
        else:
            eval(args)