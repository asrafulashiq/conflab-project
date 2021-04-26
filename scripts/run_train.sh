launcher="slurm"
while getopts "l:" opt; do
    case ${opt} in
    l)
        launcher="$OPTARG"
        ;;
    esac
done

for each_split in $(ls ./data_loading/splits/); do

    if [ $launcher = "slurm" ]; then
        LAUNCHER="launcher=slurm ngpus=4 timeout=06:00:00 mem_per_cpu=10000 cpus_per_task=6"
    fi

    base_name=$(cut -d'.' -f1 <<<"${each_split}")
    cmd="python main.py mode=train split_path=data_loading/splits/${each_split} \
        coco_json_prefix=${each_split} name=kp_${base_name} ${LAUNCHER}"
    echo $cmd
    eval $cmd

done
