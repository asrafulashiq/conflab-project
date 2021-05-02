launcher="slurm"
task=detection
backbones=("R50_FPN")
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
        backbones=($OPTARG)
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

for backbone in "${backbones[@]}"; do
    zoo=${task}_${backbone}
    name=${zoo}

    extra=""
    suff=""
    if [ ${mode} == "test" ]; then
        extra="extra checkpoint=ckpt/${name}/model_final_pth.ckpt"
        suff="_test"
    fi

    cmd="python main.py mode=${mode} create_coco=false name=${zoo}${suff} \
        task=${task} zoo=${zoo} ${LAUNCHER}"
    echo $cmd
    eval $cmd
done
