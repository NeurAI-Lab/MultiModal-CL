import pandas as pd
import numpy as np
import os

df_sel = pd.read_csv(r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/datasets/utils/vggsound/VGGSound_selected_classes.csv')

df_sel = df_sel[df_sel.Selected == 1]
sel_classes = df_sel.label.tolist()

df_vgg = pd.read_csv(r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/datasets/utils/vggsound/vggsound.csv')
df_vgg.columns = ['sample', 'start_time', 'label', 'split']

sel_samples = [label in sel_classes for label in df_vgg.label]

df_vgg_subset = df_vgg[sel_samples]

df_vgg_subset.to_csv('vgg_subset_all_samples.csv', index=False)

src_folder = r'/data/input-ai/datasets/VGGSound/data/vggsound/video/'


lst_downloaded = []
filecount = 0
for row in df_vgg_subset.iterrows():
    src_filename = '{}_{}_{}.mp4'.format(row[1][0], int(row[1][1] * 1000), int((row[1][1] + 10) * 1000))
    dest_filename = '{}_{:0>6}.mp4'.format(row[1][0], int(row[1][1]))
    src_path = os.path.join(src_folder, src_filename)

    if os.path.exists(src_path):
        lst_downloaded.append(1)
    else:
        lst_downloaded.append(0)

    filecount += 1

    if filecount % 10 == 0:
        print(f'{filecount} of {len(df_vgg_subset)} files checked')

df_vgg_subset['downloaded'] = lst_downloaded
# df_vgg_subset = df_vgg_subset[df_vgg_subset.downloaded==0]
df_vgg_subset.to_csv('vgg_subset_status.csv', index=False)


# Increase Test Set
df_vgg_subset['orig_split'] = df_vgg_subset['split']
print(df_vgg_subset.groupby(by=['label', 'split']).count())
df_vgg_subset = df_vgg_subset.sort_values(by=['label', 'split'])
print(df_vgg_subset.groupby(by=['label', 'split']).count())


df_vgg_subset = df_vgg_subset[df_vgg_subset.downloaded == 1]
df_vgg_subset = df_vgg_subset.sort_values(by=['label', 'split'])
print(df_vgg_subset.groupby(by=['label', 'split']).count())

split = []
for sel in sorted(sel_classes):
    count = 0
    subset = df_vgg_subset[df_vgg_subset.label == sel]
    print(sel, len(subset))
    for row in subset.iterrows():
        if row[1].label == sel:
            if count < 50:
                count += 1
                split.append('test')
                print('setting to test')
            else:
                split.append('train')

df_vgg_subset['split'] = split
# print(df_vgg_subset.groupby(by=['label', 'split']).count())[]

df_vgg_subset.to_csv('vgg_seq_dataset.csv', index=False)

df_vgg_subset = df[df.valid == 1]
df_vgg_subset = df_vgg_subset.sort_values(by=['label', 'split'])
split = []
for sel in sorted(sel_classes):
    test_count = 0
    train_count = 0

    subset = df_vgg_subset[df_vgg_subset.label == sel]
    print(sel, len(subset))
    for row in subset.iterrows():
        if row[1].label == sel:
            if test_count < 50:
                test_count += 1
                split.append('test')
                print('setting to test')
            elif train_count < 500:
                train_count += 1
                split.append('train')
            else:
                split.append('reserve')

df_vgg_subset['split'] = split
# print(df_vgg_subset.groupby(by=['label', 'split']).count())[]

df_vgg_subset.to_csv('vgg_seq_dataset_capped.csv', index=False)


import os

lst_files = os.listdir(r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/video')
print(len(lst_files))

lst_files = os.listdir(r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/images/Image-01-FPS')
print(len(lst_files))


import os
import csv
import pandas as pd

csv_path = r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/vgg_seq_dataset.csv'
dataset_dir = r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed'
fps = 1

lst_path = []
lst_video = []
lst_audio = []
lst_len = []
lst_valid = []

df = pd.read_csv(csv_path)

count = 0
with open(csv_path) as f:
    csv_reader = csv.reader(f)

    for item in csv_reader:
        count += 1


lst_path = []
lst_video = []
lst_audio = []
lst_len = []
lst_valid = []

count = 0
with open(csv_path) as f:
    csv_reader = csv.reader(f)

    for item in csv_reader:
        print(item)
        # if count == 0:
        #     continue
        video_dir = os.path.join(dataset_dir, 'images', 'Image-{:02d}-FPS'.format(fps), '{}_{:0>6}.mp4'.format(item[0], item[1]))
        audio_dir = os.path.join(dataset_dir, 'audio', '{}_{:0>6}.wav'.format(item[0], item[1]))

        if os.path.exists(video_dir):
            lst_video.append(1)
        else:
            lst_video.append(0)

        if os.path.exists(audio_dir):
            lst_audio.append(1)
        else:
            lst_audio.append(0)

        # if len(os.listdir(video_dir)) > 3:
        #     lst_len.append(1)
        # else:
        #     lst_len.append(0)

        if os.path.exists(video_dir) and os.path.exists(audio_dir):
            lst_valid.append(1)
        else:
            lst_valid.append(0)

        count += 1

        if count % 100 == 0:
            print(count)

df['video_exists'] = lst_video
df['audio_exists'] = lst_audio
df['valid'] = lst_valid

df.to_csv('vgg_seq_dataset_stats.csv', index=False)
