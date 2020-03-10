# UCF101 related scripts

## Required packages
The codes are tested under python3.7, you need to install the package `opencv-python`. 
You also need to install `unrar, ffmpeg`. 

## Download the videos

```
cd /path/to/UCF101/
wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
unrar e UCF101.rar videos/
```

## Extract the frames

```
python extract_frames.py
  --raw_video_dir /path/to/UCF101/videos
  --jpg_dir /path/to/UCF101/extracted_frames
```

