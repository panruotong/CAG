modeltype=llama-2-7b
modelpath=~/llama-2-7b
temperature=0.01
suffix=cred
python eval.py \
    --model-path ${modelpath} \
    --model-type ${modeltype} \
    --data-path ~/datasets \
    --save-suffix ${suffix} \
    --temperature ${temperature} \
    --setting-type ${suffix} \
    --wikimulti \
    --hotpot \
    --musique \
    --rgb \
    --misinfo \
    --evotemp \
    --is_lm \
    --vllm