# cd data_loading
# python create_splits.py
# cd ..

launcher="local"
while getopts "l:" opt; do
    case ${opt} in
    l)
        launcher="$OPTARG"
        ;;
    esac
done

# for each_split in $(ls ./data_loading/splits/); do

#     if [ $launcher = "slurm" ]; then
#         LAUNCHER="launcher=slurm ngpus=1 timeout=01:00:00 mem_per_cpu=20000 cpus_per_task=2"
#     fi

#     cmd="python main.py create_coco=true split_path=data_loading/splits/${each_split} \
#         coco_json_prefix=${each_split} name=data_${each_split} ${LAUNCHER}"
#     echo $cmd
#     eval $cmd

# done

if [ $launcher = "slurm" ]; then
    LAUNCHER="launcher=slurm ngpus=1 timeout=01:20:00 mem_per_cpu=20000 cpus_per_task=2"
fi

cmd="python main.py create_coco=true  name=data ${LAUNCHER}"
echo $cmd
eval $cmd
