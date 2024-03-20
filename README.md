# CAG

This repository contains the code and data for the paper *Not All Contexts Are Equal: Teaching LLMs Credibility-aware Generation*. 
![](first.png)
## QuickStart
You can download our Credibility-aware Generation model and our Credibility-aware Generation Benchmark from HuggingFace Hub.

[CAG 7B model](https://huggingface.co/ruotong-pan/CAG-7b) | [CAG 13B model](https://huggingface.co/ruotong-pan/CAG-13b) | [CAG Mistral 7B model](https://huggingface.co/ruotong-pan/CAG-mistral-7b)

To evaluate the model you need to install the environment.
```
pip install -r requirements.txt
```
## CAGB
## Evaluate
setting-type indicates the scenario to be evaluated, options include: concat, rerank, cred.

```
#Eval the LM
bash scripts/llama7b.sh

#Eval Chatgpt
bash scripts/chatgpt.sh

#Eval CAG
bash scripts/cag-7b.sh
```
## Training
Our training data is available [here](https://drive.google.com/file/d/1gQgdLaQON1tqflHNbJmjS5jGZU_m9mjg/view?usp=sharing).

We use [FastChat](https://github.com/lm-sys/FastChat)  to fine-tune the LLaMA 7B, 13B model and use [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) to fine-tune the Mistral 7B model.
## Citation