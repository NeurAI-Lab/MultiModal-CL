import glob
import multiprocessing
import subprocess
import os
import argparse

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_input',
        default='/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/video/',
        type=str,
        help='Input directory path of videos or audios')
    parser.add_argument(
        '--audio_output',
        default='/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/audio/',
        type=str,
        help='Output directory path of videos')
    return parser.parse_args() 

def convert(v):
    subprocess.check_call([
    'ffmpeg',
    '-n',
    '-i', v,
    '-acodec', 'pcm_s16le',
    '-ac','1',
    '-ar','16000',
    args.audio_output + '%s.wav' % v.split('/')[-1][:-4]])

def obtain_list():
    files = []
    txt = glob.glob(args.video_input + '/*.mp4')  # '/*.flac'
    for item in txt:
        files.append(item)
    return files

args = get_arguments()
# p = multiprocessing.Pool(32)
# p.map(convert, obtain_list())

lst_videos = obtain_list()

count = 0
total = len(lst_videos)
for video_path in lst_videos:
    try:
        convert(video_path)
        count += 1
        if count % 100 == 0:
            print('!' * 30)
            print('!' * 30)
            print('!' * 30)
            print('!' * 30)
            print('!' * 30)
            print(f'{count} of {total} Converted')
            print('!' * 30)

    except:
        print(f'{video_path} failed')
