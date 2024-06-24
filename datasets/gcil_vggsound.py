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
import numpy as np
from numpy.random import choice
from datasets.transforms.denormalization import DeNormalize
from collections import Counter, OrderedDict
from copy import deepcopy
import math


class sampler(object):
    def __init__(self, train_y, test_y, epoch_size, weight_dist, pretrain_class_nb, pretrain, args=None):
        self.class_numb = 0
        self.epoch_size = epoch_size  # total count of samples in this epoch
        self.pretrain = pretrain
        self.pretrain_class_nb = pretrain_class_nb  # the number of classes for pre-training
        self.new_class_num = max(train_y) - pretrain_class_nb + 1
        self.counter = Counter(np.array(train_y))
        self.chosen_class_sizes = None
        self.end_training = False  # end the incremental training if data of any class run out
        self.index_class_map_train = {class_: np.where(train_y == class_)[0] for class_ in self.counter.keys()}  # key: class, value: all indices in the dataset
        self.index_class_map_train_fixed = deepcopy(self.index_class_map_train)  # for later reference
        self.index_class_map_test = {class_: np.where(test_y == class_)[0] for class_ in self.counter.keys()}
        self.experienced_classes = OrderedDict()
        self.current_batch_class_indices = None
        self.experienced_counts = []
        self.class_not_in_this_batch = []  # used for distillation loss
        self.args = args
        if 'unif' in weight_dist or 'noise' in weight_dist:
            self.class_weight_dist = {class_: 1 for class_ in self.counter.keys()}  # to be modified
        elif weight_dist == 'longtail':  # long tail weight dist
            # this is similar to the long-tailed cifar. with eponential decay, with 0.97, the imbalance factor around 20 (class 0 appears 20 times more often than class 99)
            self.class_weight_dist = {class_: math.pow(0.984, class_) for class_ in self.counter.keys()}

            print(self.class_weight_dist)

    def sample_class_sizes(self, chosen_class_sizes):
        if chosen_class_sizes:
            self.chosen_class_sizes = chosen_class_sizes
        # pretrain
        elif self.pretrain and len(self.experienced_classes.keys()) < self.pretrain_class_nb:
            sampled_classes = list(choice(list(self.counter.keys()), size=self.pretrain_class_nb, replace=False))
            self.chosen_class_sizes = {sampled_class: self.counter[sampled_class] for sampled_class in sampled_classes}

        else:
            non_empty_classes = np.array([class_ for class_ in self.counter.keys() if self.counter[class_] != 0])
            self.class_numb = choice(min(len(non_empty_classes), self.args.phase_class_upper), size=1)[0] + 1  # this number should be greater than 0

            # we sample remaining class uniformly
            sampled_classes = list(choice(non_empty_classes, size=self.class_numb, replace=False))

            if 'noise' in self.args.weight_dist:
                weight_for_sampled_classes = [self.class_weight_dist[sampled_class] + max(np.random.normal(0, 0.2), -0.99) for sampled_class in sampled_classes]
            else:
                weight_for_sampled_classes = [self.class_weight_dist[sampled_class] for sampled_class in sampled_classes]
            normalized_weight_for_sampled_classes = [weight / sum(weight_for_sampled_classes) for weight in weight_for_sampled_classes]

            samples = []
            while not len(set(samples)) == self.class_numb:
                samples = list(choice(sampled_classes, size=self.epoch_size, replace=True, p=normalized_weight_for_sampled_classes))

            # count of data in chosen classes, however, we can not have it more than # of samples left for this class in counter
            # total_sampled_classes_weight = sum(list(map(self.class_weight_dist.get, sampled_classes)))
            self.chosen_class_sizes = {sampled_class: min(samples.count(sampled_class), self.counter[sampled_class]) for sampled_class in sampled_classes}

        # update records
        self.counter.subtract(Counter(self.chosen_class_sizes))

        return self.chosen_class_sizes

    # output a list of sample indices
    def sample_train_data_indices(self, current_batch_class_indices=None):
        if current_batch_class_indices:
            self.current_batch_class_indices = current_batch_class_indices
            chosen_class_sizes = {_class: len(_class_indices) for _class, _class_indices in current_batch_class_indices.items()}
            self.class_numb = len(chosen_class_sizes)
        else:
            # sample and remove indices
            self.current_batch_class_indices = {}
            chosen_class_sizes = None

        # get the class sizes
        _ = self.sample_class_sizes(chosen_class_sizes=chosen_class_sizes)

        for class_, size_ in self.chosen_class_sizes.items():
            if current_batch_class_indices:
                class_indices = self.current_batch_class_indices[class_]
            else:
                class_indices = list(choice(self.index_class_map_train[class_], size_, replace=False))
                # store data indices for this class
                self.current_batch_class_indices[class_] = class_indices
            # remove sampled indices
            self.index_class_map_train[class_] = list(set(self.index_class_map_train[class_]) - set(class_indices))
            # update record
            if class_ in self.experienced_classes:
                self.experienced_classes[class_] += class_indices
                self.experienced_counts[list(self.experienced_classes.keys()).index(class_)] += size_
            else:
                self.experienced_classes[class_] = class_indices
                self.experienced_counts.append(size_)

        self.class_not_in_this_batch = list(set(range(len(self.experienced_classes))) - set([self.map_labels(i) for i in self.chosen_class_sizes.keys()]))

        return np.concatenate([indices for indices in self.current_batch_class_indices.values()]).astype(int), len(self.chosen_class_sizes)

    # output a list of current epoch class test.sh indices and a list of cumulative class test.sh indices
    def sample_test_data_indices(self):
        # the indices of the current epoch classes
        current_test_ind = np.concatenate([self.index_class_map_test[class_] for class_ in self.chosen_class_sizes.keys()])
        # the indices of all past classes
        cumul_test_ind = np.concatenate([self.index_class_map_test[class_] for class_ in self.experienced_classes])

        return current_test_ind, cumul_test_ind

    # map original labels to the order labels
    def map_labels(self, original_label):
        return list(self.experienced_classes.keys()).index(original_label)

    # convert the sample index in the whole dataset to its index in the class
    def map_index_in_class(self, class_, indices_in_dataset):
        return [np.where(self.index_class_map_train_fixed[class_] == index_in_dataset)[0][0] for index_in_dataset in indices_in_dataset]  # index in its class


class VGGSound(Dataset):

    def __init__(self, csv_path, dataset_dir, mode='train', fps=1, num_video_frames=4, transform=None, target_transform=None):

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

        with open(self.csv_path) as f:
            csv_reader = csv.reader(f)

            for item in csv_reader:
                if item[3] == 'train':
                    video_dir = os.path.join(self.dataset_dir, 'images', 'Image-{:02d}-FPS'.format(self.fps), '{}_{:0>6}.mp4'.format(item[0], item[1]))
                    audio_dir = os.path.join(self.dataset_dir, 'audio', '{}_{:0>6}.wav'.format(item[0], item[1]))

                    # print(video_dir, audio_dir)

                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        # print(video_dir)
                        train_video_data.append(video_dir)
                        train_audio_data.append(audio_dir)
                        if item[2] not in train_class:
                            train_class.append(item[2])
                        train_label.append(item[2])

                if item[3] == 'test':
                    video_dir = os.path.join(self.dataset_dir, 'images', 'Image-{:02d}-FPS'.format(self.fps), '{}_{:0>6}.mp4'.format(item[0], item[1]))
                    audio_dir = os.path.join(self.dataset_dir, 'audio', '{}_{:0>6}.wav'.format(item[0], item[1]))
                    if os.path.exists(video_dir) and os.path.exists(audio_dir) and len(os.listdir(video_dir)) > 3:
                        # print(video_dir)
                        test_video_data.append(video_dir)
                        test_audio_data.append(audio_dir)
                        if item[2] not in test_class:
                            test_class.append(item[2])
                        test_label.append(item[2])

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


class GCILVGGSound(AVContinualDataset):
    NAME = 'gcil_vggsound'
    SETTING = 'multimodal-class-il'

    N_CLASSES_PER_TASK = 5
    N_TASKS = 20
    N_CLASSES = 100

    IMG_TRANSFORM = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test.sh lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """

        super(GCILVGGSound, self).__init__(args)

        transform = GCILVGGSound.IMG_TRANSFORM

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
            transform=transform
        )

        self.test_dataset = VGGSound(
            csv_path=self.args.vggsound_csv_path,
            dataset_dir=self.args.dataset_dir,
            mode='test',
            fps=self.args.fps,
            num_video_frames=self.args.num_video_frames,
            transform=test_transform
        )

        self.V_train_total = np.array(self.train_dataset.video)
        self.A_train_total = np.array(self.train_dataset.audio)
        self.Y_train_total = np.array(self.train_dataset.targets)

        self.V_valid_total = np.array(self.test_dataset.video)
        self.A_valid_total = np.array(self.test_dataset.audio)
        self.Y_valid_total = np.array(self.test_dataset.targets)

        self.current_batch_class_indices = None
        self.current_training_indices = None

        self.sampling_count = 0
        np.random.seed(self.args.gcil_seed)
        self.sampling_seeds = [np.random.randint(0, 10000) for _ in range(GCILVGGSound.N_TASKS)]
        print(self.sampling_seeds)

        self.ind_sampler = sampler(
            self.Y_train_total,
            self.Y_valid_total,
            epoch_size=self.args.epoch_size,
            pretrain=self.args.pretrain,
            pretrain_class_nb=self.args.pretrain_class_nb,
            weight_dist=self.args.weight_dist,
            args=self.args,
        )

    def get_data_loaders(self):
        transform = GCILVGGSound.IMG_TRANSFORM

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
            transform=transform
        )

        self.test_dataset = VGGSound(
            csv_path=self.args.vggsound_csv_path,
            dataset_dir=self.args.dataset_dir,
            mode='test',
            fps=self.args.fps,
            num_video_frames=self.args.num_video_frames,
            transform=test_transform
        )

        np.random.seed(self.sampling_seeds[self.sampling_count])
        indice_train, num_classes = self.ind_sampler.sample_train_data_indices(current_batch_class_indices=self.current_batch_class_indices)
        indice_test, indice_test_cumul = self.ind_sampler.sample_test_data_indices()

        self.current_training_indices = indice_train

        # access data for this phase
        V_train = self.V_train_total[indice_train]
        A_train = self.A_train_total[indice_train]
        Y_train = self.Y_train_total[indice_train]

        V_test_cumul = self.V_valid_total[indice_test_cumul]
        A_test_cumul = self.A_valid_total[indice_test_cumul]
        Y_test_cumul = self.Y_valid_total[indice_test_cumul]

        print('=' * 30)
        print('samples for current Task')
        (unique, counts) = np.unique(Y_train, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        print(frequencies)

        # label mapping
        map_Y_train = np.array([self.ind_sampler.map_labels(i) for i in Y_train])  # train labels: the order of classes
        map_Y_test_cumul = np.array([self.ind_sampler.map_labels(i) for i in Y_test_cumul])

        print('X_train size: ', len(Y_train))
        print('number of classes: ', self.ind_sampler.class_numb)

        self.train_dataset.targets = Y_train
        self.train_dataset.audio = A_train
        self.train_dataset.video = V_train

        self.test_dataset.targets = Y_test_cumul
        self.test_dataset.audio = A_test_cumul
        self.test_dataset.video = V_test_cumul

        train_loader, test_loader = self.store_loaders(self.train_dataset, self.test_dataset)
        self.sampling_count += 1

        return train_loader, test_loader

    def get_backbone(self):
        return AVClassifier(n_classes=GCILVGGSound.N_CLASSES, fusion=self.args.fusion_method, modalities_used=self.args.modalities_used)

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
        return GCILVGGSound.get_batch_size()

# args = Namespace(
#     vggsound_csv_path=r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed/vgg_seq_dataset_capped_small.csv',
#     dataset_dir=r'/data/input-ai/datasets/VGGSound/data/vggsound/data_processed',
#     fps=1,
#     num_video_frames=3,
#     batch_size=8,
#     fusion_method='gated',
#     modalities_used='audio_video'
# )
#
# dataset = SequentialVGGSound(args)
#
# train_loader, test_loader = dataset.get_data_loaders()
#
# total = len(train_loader)
# lst_labels = []
# for i, ((spec, video), label, _) in enumerate(train_loader):
#     lst_labels.append(label)
#
# labels = torch.cat(lst_labels)
#
# from backbone.AudioVideoNet import AVClassifier
# model = AVClassifier(309, 'gated', modalities_used='audio_video')
#
# a, v, out = model(spec.unsqueeze(1), video)

# import pandas as pd
# df = pd.read_csv(r'/volumes2/workspace/nie_continual_learning/mammoth-multimodal/vgg_seq_dataset_capped.csv')
# df = df.sample(frac=1)
