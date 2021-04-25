cd data_loading
python create_splits.py
cd ..

launcher="local"
while getopts "l:" opt; do
    case ${opt} in
    l)
        launcher="$OPTARG"
        ;;
    esac
done

for each_split in $(ls ./data_loading/splits/); do

    if [ $launcher = "slurm" ]; then
        LAUNCHER="ngpus=0 timeout=00:30:00"
    fi

    cmd="python main.py create_coco=true split_path=data_loading/splits/${each_split} \
        coco_json_prefix=${each_split} name=data_each_split ${LAUNCHER}"
    echo $cmd
    eval $cmd

done
