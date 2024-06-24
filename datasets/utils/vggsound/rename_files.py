from glob import glob
import shutil
import pandas as pd
import os

src_folder = r'/data/input-ai/datasets/VGGSound/data/vggsound/video/'
dest_folder = r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/video'


csv_file = r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/datasets/utils/vggsound/vggsound.csv'
df = pd.read_csv(csv_file, header=None)

filecount = 0

for row in df.iterrows():

    src_filename = '{}_{}_{}.mp4'.format(row[1][0], int(row[1][1] * 1000), int((row[1][1] + 10) * 1000))
    dest_filename = '{}_{:0>6}.mp4'.format(row[1][0], int(row[1][1]))
    src_path = os.path.join(src_folder, src_filename)
    dest_path = os.path.join(dest_folder, dest_filename)

    if os.path.exists(src_path):
        if not os.path.exists(dest_path):
            shutil.copy(src_path, dest_path)

    filecount += 1

    if filecount % 10 == 0:
        print(f'{filecount} of {len(df)} files moved')

