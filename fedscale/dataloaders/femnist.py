from __future__ import print_function

import csv
import os
import os.path
import warnings

from PIL import Image

import redis
import heapdict
import PIL
from rediscluster import RedisCluster
from collections import OrderedDict



class FEMNIST():
    """
    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    classes = []

    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, dataset='train', transform=None, target_transform=None, imgview=False):

        self.data_file = dataset  # 'train', 'test', 'validation'
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.path = os.path.join(self.processed_folder, self.data_file) #this path formation is of no use.

        # load data and targets
        #self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
        #load clients, corresponding data and targets
        self.clients, self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
        #self.data, self.targets = self.load_file(self.path) #self.path is not used. check self.load_file.
        #self.mapping = {idx:file for idx, file in enumerate(raw_data)}

        self.imgview = imgview

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
            #this method is accessed whenever a client has a to fetch a data sample for training.
        """

        imgName, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.root, imgName))
        client = self.clients[index]
        #print(f'client {self.clients[index]} is accessing image {index}: {imgName}') 
        #

        #print(f'type of client before: {type(client)}')
        #print(f'type of index: {type(index)}')
        client = int(client)
        #print(f'type of client after: {type(client)}')

        ######## CACHING LOGIC GOES HERE #######
        fetched_from_cache = 0

        if index % 2 == 0:
            fetched_from_cache = 1

        ######################################

        # avoid channel error
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        #return img, target, index
        #return client, img, target, index
        return img, target, index, client, fetched_from_cache
        #return img, target

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return self.root

    @property
    def processed_folder(self):
        return self.root

    def _check_exists(self):
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.data_file)))

    def load_meta_data(self, path):
        #datas, labels = [], []
        clients, datas, labels = [], [], [] #adding clients list

        with open(path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0: #skip the first row as it has headers client_id,sample_path,label_name,label_id
                    clients.append(row[0]) #appending clients in clients list
                    datas.append(row[1]) # take the location as data ex. data/raw_data/by_write/hsf_0/f0388_26/u0388_26/u0388_26_00004.png
                    labels.append(int(row[-1])) # take the target as label ex. 6
                line_count += 1

        #return datas, labels # return a list of datas and a list of labels.
        return clients, datas, labels# return a list of clients, datas and a list of labels.

    def load_file(self, path):

        # load meta file to get labels
        # datas, labels = self.load_meta_data(os.path.join(
        #     self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))
        clients, datas, labels = self.load_meta_data(os.path.join(
            self.processed_folder, 'client_data_mapping', self.data_file+'.csv'))

        #return datas, labels
        return clients, datas, labels
