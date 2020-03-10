import os
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--video_root_dir',
            type=str, action='store',
            help='Directory to hold the downloaded videos')
    parser.add_argument(
            '--mapping_file_path',
            type=str, action='store',
            help='Path to the mapping file')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
 
    os.chdir(args.video_root_dir)
    
    with open(args.mapping_file_path, 'w+') as mapping_file:
        count = 1
        categories = [d for d in os.listdir('.') if os.path.isdir(d)]
        categories = sorted(categories)
        for cat in tqdm(categories):
            videos = sorted(os.listdir(cat))
            for video in videos:
                # Rename
                video_old_path = os.path.join(cat, video)
                video_new_path = os.path.join(cat, str(count)+'.avi')
                os.rename(video_old_path, video_new_path)
                # Write to mapping file
                mapping_file.write("{} {}\n".format(video_old_path, video_new_path))

                count += 1

    mapping_file.close() 


if __name__ == '__main__':
    main()