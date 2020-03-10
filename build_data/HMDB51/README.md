# HMDB51 related scripts

## Required packages
The codes are tested under python3.7, you need to install the package `opencv-python`. 
You also need to install `unrar, ffmpeg`. 

## Download the videos

```
cd /path/to/HMDB51/
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar
unrar e hmdb51_org.rar videos/
find videos/ -name '*.rar' -execdir unrar x -o- {} \;
```

## Clean filenames 
Convert the original filenames of HMDB51 videos to numbers, because the original video names contain many special characters. Meanwhile, the mapping of old and new filenames is saved in a text file.

```
python convert_filename_to_number.py 
  --video_root_dir /path/to/HMDB51/videos 
  --mapping_file_path /path/to/HMDB51/file_num_mapping.txt
```

## Extract the frames

```
python extract_frames.py
  --raw_video_dir /path/to/HMDB51/videos
  --jpg_dir /path/to/HMDB51/extracted_frames
```

