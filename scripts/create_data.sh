cd data_loading
python create_splits.py
cd ..

for each_split in $(ls ./data_loading/splits/); do

    cmd="python main.py create_coco=true split_path=data_loading/splits/${each_split} \
        coco_json_prefix=${each_split}"
    echo $cmd
    eval $cmd

done
