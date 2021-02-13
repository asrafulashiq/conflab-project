### Date: Jan 12

The set of video data that we're annotating can be found in the  

**/staff-bulk/ewi/insy/SPCDataSets/conflab-mm/processed/annotation/videoSegments/**

Concretely, for `cam2`, `cam4`, `cam6` and `cam8`, we are annotating the following segments in their respective subdirectory: 



vid2-seg8-scaled-denoised.mp4

vid2-seg9-scaled-denoised.mp4

vid3-seg1-scaled-denoised.mp4

vid3-seg2-scaled-denoised.mp4

vid3-seg3-scaled-denoised.mp4

vid3-seg4-scaled-denoised.mp4

vid3-seg5-scaled-denoised.mp4

vid3-seg6-scaled-denoised.mp4

vid3-seg7-scaled-denoised.mp4

vid3-seg8-scaled-denoised.mp4

vid3-seg9-scaled-denoised.mp4



for `cam10`, we are annotating the following segments (this is a subset of the list above; the rest is not included because there's no subject in view):



vid3-seg3-scaled-denoised.mp4
vid3-seg4-scaled-denoised.mp4
vid3-seg5-scaled-denoised.mp4
vid3-seg6-scaled-denoised.mp4
vid3-seg7-scaled-denoised.mp4
vid3-seg8-scaled-denoised.mp4
vid3-seg9-scaled-denoised.mp4

From cam2 to cam10, the video covers the whole *interaction floor*. So these segments with the same (X,Y) vidX-segY should be concurrent in time, though perfectly matching them requires the timecode information. For each camera, we will have **00:21:16** of data, and between the 5 cameras, we are expecting **01:46:20** of data. Of course on a scene level for the whole interaction floor, it would still be 00:21:16 worth of data. The annotations are done on a frame level. 

The videos were recorded with **59.94** FPS, 1920x1080. These segments listed above were scaled from the original video file to 960x540. However, the annotations themselves are recorded as fractions of the original dimensions so they can be easily converted by multiply the x values by 1920 and y values by 1080. (I suspect Jose would've done this already for the files that he planned for Friday). These segments are then denoised for better performance with extracting the optical flow, as the annotation tool uses this information of a local patch near the cursor to dynamically adjust the video speed. If you need any intermediate segments before/after some treatment, you can run the bash script in each of the "cam" subdirectories. 

You can find a lot more data in /staff-bulk/ewi/insy/SPCDataSets/conflab-mm/raw/video/. We have 4 cameras from elevated-side view though we cannot publish these videos according to a prior ethical agreement. We have 9 cameras from overhead view which we can publish. For came04, as you already noticed, the videos were recorded in a portrait view so the rotated videos can be found in rotatedCam4 for the landscape view for consistency. Even though we're only annotating for 00:21:16 of data from 5 out of the 9 cameras. In reality, the cameras were recording throughout the whole event, which was probably around 1.25 hours or so. In each of these camera subdirectories, you can find files of the form GH010003, GH020003, GH030003, etc. The exact suffix changes based on the camera but there's a consistent pattern of GHX1XXXX, GHX2XXXX, GHX3XXX, GHX4XXXX, and GHX5XXXX, a total of 5 MP4 files with a few rare exceptions if the camera died in the middle of the event. The portions that we are annotating are the last 00:03:38 of GHX2XXXX and the entire duration (00:17:38) of GHX3XXXX, for each of the 5 cameras that I mentioned above. 