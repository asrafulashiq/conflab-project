mode="d"
while getopts "m:" opt; do
    case ${opt} in
    m)
        mode="$OPTARG"
        ;;
    esac
done

cams1=("cam2" "cam4" "cam6" "cam8")
vid_names1=(
    "vid2-seg8-scaled-denoised.mp4"
    "vid2-seg9-scaled-denoised.mp4"
    "vid3-seg1-scaled-denoised.mp4"
    "vid3-seg2-scaled-denoised.mp4"
    "vid3-seg3-scaled-denoised.mp4"
    "vid3-seg4-scaled-denoised.mp4"
    "vid3-seg5-scaled-denoised.mp4"
    "vid3-seg6-scaled-denoised.mp4"
    "vid3-seg7-scaled-denoised.mp4"
    "vid3-seg8-scaled-denoised.mp4"
    "vid3-seg9-scaled-denoised.mp4"
)

cams2=("cam10")
vid_names2=(
    "vid3-seg3-scaled-denoised.mp4"
    "vid3-seg4-scaled-denoised.mp4"
    "vid3-seg5-scaled-denoised.mp4"
    "vid3-seg6-scaled-denoised.mp4"
    "vid3-seg7-scaled-denoised.mp4"
    "vid3-seg8-scaled-denoised.mp4"
    "vid3-seg9-scaled-denoised.mp4"
)

if [[ "$mode" = "d"* ]]; then
    username="e000768"
    password="TUDelft@asrafulashiq1125"

    root_url="https://webdata.tudelft.nl/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/processed/annotation/videoSegments/"

    save_dir="${HOME}/datasets/conflab-mm/processed/videoSegments"
    mkdir -p "$save_dir"

    for cam in "${cams1[@]}"; do
        for name in "${vid_names1[@]}"; do
            out_dir="${save_dir}/${cam}"
            mkdir -p $out_dir
            url="${root_url}/${cam}/${name}"
            wget --user "${username}" --password "${password}" -P "${out_dir}" "${url}"
        done
    done

    for cam in "${cams2[@]}"; do
        for name in "${vid_names2[@]}"; do
            out_dir="${save_dir}/${cam}"
            mkdir -p $out_dir
            url="${root_url}/${cam}/${name}"
            wget --user "${username}" --password "${password}" -P "${out_dir}" "${url}"
        done
    done

elif [[ "$mode" = "e"* ]]; then
    # extract frames
    in_dir="${HOME}/datasets/conflab-mm/processed/videoSegments"
    save_dir="${HOME}/datasets/conflab-mm/frames/videoSegments"
    mkdir -p "$save_dir"

    for cam in "${cams1[@]}"; do
        for name in "${vid_names1[@]}"; do
            basename=$(cut -d'.' -f1 <<<"$name")
            out_path="${save_dir}/${cam}/${basename}"
            mkdir -p "${out_path}"
            in_path="${in_dir}/${cam}/${name}"
            ffmpeg -i "${in_path}" "${out_path}/%06d.jpg"
        done
    done

    for cam in "${cams2[@]}"; do
        for name in "${vid_names2[@]}"; do
            basename=$(cut -d'.' -f1 <<<"$name")
            out_path="${save_dir}/${cam}/${basename}"
            mkdir -p "${out_path}"
            in_path="${in_dir}/${cam}/${name}"
            ffmpeg -i "${in_path}" "${out_path}/%06d.jpg"
        done
    done

fi
