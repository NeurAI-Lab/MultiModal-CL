import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.nn.functional as F
from PIL import Image
import os
from datasets.utils.av_continual_dataset import AVContinualDataset, get_previous_train_loader
from datasets.transforms.denormalization import DeNormalize
import csv
import torch
import librosa
import random
from backbone.AudioVideoNet import AVClassifier
from argparse import Namespace
import pandas as pd


class VGGSound(Dataset):

    def __init__(self, csv_path, dataset_dir, mode='train', fps=1, num_video_frames=4, transform=None, target_transform=None, task=0, dict_super_cat=None, dict_task_classes=None):

        self.csv_path = csv_path
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.fps = fps
        self.num_video_frames = num_video_frames
        self.transform = transform
        self.not_aug_transform = transforms.Compose([transforms.Resize(size=(360, 360)), transforms.ToTensor()])
        self.target_transform = target_transform

        train_video_data = []
        train_audio_data = []
        test_video_data = []
        test_audio_data = []
        train_label = []
        test_label = []
        train_class = []
        test_class = []

        train_label_fine = []
        print(f'Task {task}')
        print(f'Classes in Categories {dict_task_classes[task]}')

        with open(self.csv_path) as f:
            csv_reader = csv.reader(f)

            for item in csv_reader:
                if item[3] == 'train':
                    video_dir = os.path.join(self.dataset_dir, 'images', 'Image-{:02d}-FPS'.format(self.fps), '{}_{:0>6}.mp4'.format(item[0], item[1]))
                    audio_dir = os.path.join(self.dataset_dir, 'audio', '{}_{:0>6}.wav'.format(item[0], item[1]))
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        label = item[2]
                        if label in dict_task_classes[task]:
                            train_video_data.append(video_dir)
                            train_audio_data.append(audio_dir)

                            if dict_super_cat[label] not in train_class:
                                train_class.append(dict_super_cat[label])
                            train_label.append(dict_super_cat[label])
                            train_label_fine.append(label)

                if item[3] == 'test':
                    video_dir = os.path.join(self.dataset_dir, 'images', 'Image-{:02d}-FPS'.format(self.fps), '{}_{:0>6}.mp4'.format(item[0], item[1]))
                    audio_dir = os.path.join(self.dataset_dir, 'audio', '{}_{:0>6}.wav'.format(item[0], item[1]))
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        label = item[2]
                        if label in dict_task_classes[task]:
                            test_video_data.append(video_dir)
                            test_audio_data.append(audio_dir)
                            if dict_super_cat[label] not in test_class:
                                test_class.append(dict_super_cat[label])
                            test_label.append(dict_super_cat[label])

        # assert len(train_class) == len(test_class)
        self.classes = train_class

        class_dict = dict(zip(self.classes, range(len(self.classes))))
        self.class_dict = class_dict

        if mode == 'train':
            self.video = np.array(train_video_data)
            self.audio = np.array(train_audio_data)
            self.targets = np.array([class_dict[train_label[idx]] for idx in range(len(train_label))])
            print('Training Samples:', len(self.video))

        if mode == 'test':
            self.video = np.array(test_video_data)
            self.audio = np.array(test_audio_data)
            self.targets = np.array([class_dict[test_label[idx]] for idx in range(len(test_label))])
            print('Test Samples:', len(self.video))

    def __len__(self):
        return len(self.video)

    def __getitem__(self, idx):

        # audio
        sample, rate = librosa.load(self.audio[idx], sr=16000, mono=True)
        while len(sample)/rate < 10.:
            sample = np.tile(sample, 2)

        start_point = random.randint(a=0, b=rate*5)
        new_sample = sample[start_point:start_point+rate*5]
        new_sample[new_sample > 1.] = 1.
        new_sample[new_sample < -1.] = -1.

        spectrogram = librosa.stft(new_sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)

        # Visual
        image_samples = os.listdir(self.video[idx])
        select_index = np.random.choice(len(image_samples), size=self.num_video_frames, replace=False)
        select_index.sort()
        images = torch.zeros((self.num_video_frames, 3, 224, 224))
        org_images = torch.zeros((self.num_video_frames, 3, 360, 360))

        for i in range(self.num_video_frames):
            filename = image_samples[i].decode('utf-8') if isinstance(image_samples[i], bytes) else image_samples[i]
            img = Image.open(os.path.join(self.video[idx], filename)).convert('RGB')

            org_images[i] = self.not_aug_transform(img)
            img = self.transform(img)
            images[i] = img

        images = torch.permute(images, (1, 0, 2, 3))
        org_images = torch.permute(org_images, (1, 0, 2, 3))
        label = self.targets[idx]

        return (spectrogram, images), label, (spectrogram, org_images)


class DomainVGGSound(AVContinualDataset):
    NAME = 'domain_vggsound'
    SETTING = 'multimodal-domain-il'

    N_CATEGORIES = 5
    CLASSES_PER_CATEGORY = 20
    N_TASKS = 5

    IMG_TRANSFORM = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __init__(self, args):
        super(DomainVGGSound, self).__init__(args)
        np.random.seed(args.domain_vgg_seed)
        DomainVGGSound.N_TASKS = args.domain_vgg_ntasks

        cat_df = pd.read_csv(self.args.vggsound_categories_path)  # Give the sorted path
        # cat_df = cat_df.sample(frac=1).reset_index(drop=True)
        # cat_df = cat_df.sort_values(by='category')
        classes_per_task = int(DomainVGGSound.CLASSES_PER_CATEGORY / DomainVGGSound.N_TASKS)

        np_tasks = list(range(0, DomainVGGSound.N_TASKS))
        tasks_assign = list(np.repeat(np_tasks, classes_per_task)) * DomainVGGSound.N_CATEGORIES

        cat_df['task'] = tasks_assign

        # Create a dictionary of tasks
        self.dict_task_classes = {task_id: [] for task_id in range(DomainVGGSound.N_TASKS)}
        self.dict_super_cat = {}

        for i, row in cat_df.iterrows():
            self.dict_task_classes[row.task].append(row.label)
            self.dict_super_cat[row.label] = row.category

    def get_data_loaders(self):
        transform = DomainVGGSound.IMG_TRANSFORM

        test_transform = transforms.Compose([
            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            self.get_normalization_transform()
        ])

        self.train_dataset = VGGSound(
            csv_path=self.args.vggsound_csv_path,
            dataset_dir=self.args.dataset_dir,
            mode='train',
            fps=self.args.fps,
            num_video_frames=self.args.num_video_frames,
            transform=transform,
            task=self.task,
            dict_super_cat=self.dict_super_cat,
            dict_task_classes=self.dict_task_classes,
        )

        self.test_dataset = VGGSound(
            csv_path=self.args.vggsound_csv_path,
            dataset_dir=self.args.dataset_dir,
            mode='test',
            fps=self.args.fps,
            num_video_frames=self.args.num_video_frames,
            transform=test_transform,
            task=self.task,
            dict_super_cat=self.dict_super_cat,
            dict_task_classes=self.dict_task_classes,
        )

        train, test = self.store_loaders(self.train_dataset, self.test_dataset)
        return train, test

    def get_backbone(self):
        return AVClassifier(n_classes=DomainVGGSound.N_CATEGORIES, fusion=self.args.fusion_method, modalities_used=self.args.modalities_used)

    @staticmethod
    def get_loss():
        return F.cross_entropy

    def get_transform(self):
        transform = transforms.Compose(
            [transforms.ToPILImage(), self.IMG_TRANSFORM])
        return transform

    @staticmethod
    def get_normalization_transform():
        transform = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_denormalization_transform():
        transform = DeNormalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        return transform

    @staticmethod
    def get_scheduler(model, args):
        return None

    @staticmethod
    def get_epochs():
        return 50

    @staticmethod
    def get_batch_size():
        return 32

    @staticmethod
    def get_minibatch_size():
        return DomainVGGSound.get_batch_size()
