#!/bin/bash
echo "hello!"
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Please provide a benchmark and an experiment number as arguments."
    echo "Usage: bash script_name.sh <benchmark> <experiment>"
    exit 1
fi

benchmark=$1
experiment=$2

export HF_ENDPOINT=https://hf-mirror.com
# The directory to copy from
dir=./output/results/${benchmark}/${benchmark}_${experiment}
# The directory to copy to
dest=./input/results/${benchmark}/${benchmark}_${experiment}

echo "output dir: ${dir}, input dir: ${dest}"

mkdir -p $dir
mkdir -p $dest
touch $dir/${benchmark}_${experiment}.log #create log file
nohup python3 -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/instructblip/train/${benchmark}/finetune_instructblip_${benchmark}_${experiment}.yaml 2>&1 | tee $dir/${benchmark}_$experiment.log
rsync -av --no-o --no-g --chmod=777 --exclude='*.pth' $dir/ $dest/ 