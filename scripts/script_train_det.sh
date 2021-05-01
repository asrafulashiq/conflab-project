launcher="slurm"
task=detection
backbone=R50_FPN
mode="train"
while getopts "l:t:b:m:r:" opt; do
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
    r)
        ranks=($OPTARG)
        ;;
    esac
done

if [ $launcher = "slurm" ]; then
    LAUNCHER="launcher=slurm ngpus=4 timeout=06:00:00 mem_per_cpu=10000 cpus_per_task=6"
fi

zoo=${task}_${backbone}
cmd="python main.py mode=${mode} create_coco=false name=${zoo} \
        task=${task} zoo=${zoo} ${LAUNCHER}"
echo $cmd
eval $cmd
