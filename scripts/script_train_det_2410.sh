launcher="slurm"
task=detection
backbones=("R50_FPN")
mode="train"
half=false

while getopts "l:t:b:m:r:h:" opt; do
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
    h)
        half=$OPTARG
        ;;
    esac
done

if [ $launcher = "slurm" ]; then
    LAUNCHER="launcher=slurm  timeout=06:00:00 mem_per_cpu=10000 cpus_per_task=6"
fi

for backbone in "${backbones[@]}"; do
    zoo=${task}_${backbone}
    name=${zoo}

    extra=""
    suff=""
    if [ ${mode} = "test" ]; then
        checkpoint=ckpt/${name}/model_final.pth
        if [ ! -e ${checkpoint} ]; then
            echo ${checkpoint} does not exist
            exit
        fi
        extra="${extra} checkpoint=${checkpoint}  ngpus=1"
        suff="_test"
    else
        extra="ngpus=4"
    fi

    if [ $half = "true" ]; then
        suff="${suff}_half"
    fi

    coco_json_prefix=_2410
    cmd="python main.py mode=${mode} create_coco=false name=${zoo}${coco_json_prefix}${suff} \
        task=${task} 'train_cam=[cam2,cam4,cam10]' coco_json_prefix=${coco_json_prefix}  \
        zoo=${zoo} ${LAUNCHER} ${extra} half_crop=${half}"
    echo $cmd
    eval $cmd
done
