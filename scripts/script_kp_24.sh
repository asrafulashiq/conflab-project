launcher="slurm"
task=keypoint
backbones=(R50_FPN)
# ranks=("0" "1" "2" "3")
ranks=("0")
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
    LAUNCHER="launcher=slurm ngpus=4 timeout=06:00:00 mem_per_cpu=10000 cpus_per_task=6"
fi

for rank in "${ranks[@]}"; do
    for backbone in "${backbones[@]}"; do
        zoo=${task}_${backbone}
        coco_json_prefix=_24
        name=kp${coco_json_prefix}_${zoo}_kr_${rank}
        extra=""
        suff=""
        if [ ${mode} = "test" ]; then
            checkpoint=ckpt/${name}/model_final.pth
            if [ ! -e ${checkpoint} ]; then
                echo ${checkpoint} does not exist
                exit
            fi
            extra="${extra} checkpoint=${checkpoint} ngpus=1"
            suff="_test"
        else
            extra="ngpus=4"
        fi

        if [ $half = "true" ]; then
            suff="${suff}_half"
        fi

        cmd="python main.py mode=${mode} create_coco=true \
        name=kp${coco_json_prefix}_${zoo}_kr_${rank}${suff} \
        task=${task} 'train_cam=[cam2,cam4]' zoo=${zoo} \
        coco_json_prefix=${coco_json_prefix} create_coco=true force_register=false \
        ${LAUNCHER} kp_rank=${rank} ${extra} half_crop=${half}"
        echo $cmd
        eval $cmd
    done
done
