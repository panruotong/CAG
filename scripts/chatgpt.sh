suffix=concat
python eval.py \
    --data-path /mnt/panruotong2021/Code/CAG/datasets \
    --model-type gpt-3.5-turbo-16k \
    --save-suffix ${suffix} \
    --temperature 0.01 \
    --setting-type ${suffix} \
    --wikimulti \
    --hotpot \
    --musique \
    --rgb \
    --misinfo \
    --evotemp