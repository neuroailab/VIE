# Kinetics-400 related scripts

## Required packages
The codes are tested under python3.7, you need to install these two python packages: `opencv-python, pytube3`. 
You also need to install `ffmpeg`. 

## Download the videos

```
python download_videos.py 
  --csv_path path_to_the_downloaded_kinetics_train.csv
  --save_dir directory_to_host_downloaded_videos
  --save_raw_dir temporary_directory_for_downloading
  --len_idx 246535
```

You may want to run multiple downloading processes on different computers with different `sta_idx` (start index of the downloading) 
and `len_idx` (number of videos that will be downloaded) to be able to download them faster. 
Otherwise, the downloading can take a very long time.

## Extract the frames

```
python extract_frames.py
  --csv_path path_to_the_downloaded_kinetics_train.csv
  --video_dir directory_to_host_downloaded_videos
  --jpg_dir directory_to_host_the_jpgs
  --len_idx 246535
```

Similarly, you may want to run it on different computers and then finally merge all results.
