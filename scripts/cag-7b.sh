modeltype=CAG-7b
modelpath=~/CAG-7b
suffix=cred
temperature=0.01
python eval.py \
    --model-path ${modelpath} \
    --model-type ${modeltype} \
    --data-path /mnt/panruotong2021/Code/CAG/datasets \
    --save-suffix ${suffix} \
    --temperature ${temperature} \
    --setting-type ${suffix} \
    --wikimulti \
    --hotpot \
    --musique \
    --rgb \
    --misinfo \
    --evotemp