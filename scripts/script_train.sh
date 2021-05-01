launcher="slurm"
task=keypoint
backbone=R50_FPN

mode="train"
while getopts "l:t:b:m:" opt; do
    case ${opt} in
    l)
        launcher="$OPTARG"
        ;;
    t)
        task=$OPTARG
        ;;
    b)
        backbone=$OPTARG
        ;;
    m)
        mode="$OPTARG"
        ;;
    esac
done

if [ $launcher = "slurm" ]; then
    LAUNCHER="launcher=slurm ngpus=4 timeout=06:00:00 mem_per_cpu=10000 cpus_per_task=6"
fi

for rank in "0" "1" "2" "3"; do

    zoo=${task}_${backbone} ${LAUNCHER}
    cmd="python main.py mode=${mode} create_coco=true name=${zoo}_kr_${rank} \
        task=${task} zoo=${zoo} ${LAUNCHER} kp_rank=${rank}"
    echo $cmd
    eval $cmd
done
