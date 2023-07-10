# Copied from https://github.com/microsoft/CameraTraps/blob/main/research/active_learning/deep_learning/data_loader.py
# except with a few modifications as noted in the comments and docstrings below

"""
Changelog

BalancedBatchSampler
- Changed assignment of self.labels in __init__ function
    so that it uses the Subset class's get_item method
    rather than using the dataset's indices attribute within
    the subset class. This is so that ExtendedImageFolder's
    index abstraction, self.indices, could be removed and
    to remove the dependecy of BalancedBatchSampler on this
    attribute
    
    
ExtendedImageFolder
- Added method get_labelled_indices which returns
    the indicies of all labelled data
- Added method add_data() which adds data to the data set
    from a folder of images.
"""


from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import (RandomCrop, RandomErasing, 
CenterCrop, ColorJitter, RandomRotation, RandomHorizontalFlip, RandomOrder,
Normalize, Resize, Compose, ToTensor, RandomGrayscale)
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

import numpy as np
import pandas as pd
import os
import random
import logging
from PIL import ImageStat
from .engine import Engine

class ExtendedImageFolder(ImageFolder):
    """
    Like the Image Folder data loader but with some added features.
    Features include setting up image normalisation and other transformations
    for train and validation data sets.
    
    Defines custom samplers that load data with two different modes: train
    and val. Train mode performs random transformations on the images
    to artificially generate new images whereas Val mode does not.
    Both modes will crop images to be of size 224 by 224 pixels.
    
    Changelog
    
    - Added "update_labels" function which updates
        the labels of images in the dataset
    - Added condition to balanced loader to set n_classes
        to be at most the number of classes that actually
        exist as to stop it from crashing when that was
        the case
    - Removed part of the function calc_mean_std that
        saves the mean and std as this is done by the
        checkpoint manager
    - Added function called remove_label that removes a 
        label from the dataset
    - Added fit parameter to __init__ that when set to True
        will calculate the mean and std of the dataset
    - Removed self.base_folder attribute as it's a duplicate
        of Image Folder's default self.root attribute
    """

    def __init__(self, base_folder, fit = True):
        """
        fit : If True, mean and std of dataset
            will be fitted on all images in base_folder
        
        """
    
        super().__init__(base_folder)
        
        # Class number for unlabelled images
        self.unlab_class_num = float('-inf')
        
        
        # Fit means and std of images if it should be done
        if fit:
            self.mean, self.std = self.calc_mean_std()
        else:
            self.mean = None
            self.std = None
        
        self.trnsfm = {}
        self.trnsfm['train'] = self.get_transform('train')
        self.trnsfm['val'] = self.get_transform('val')

    def setTransform(self, transform):
        assert transform in ["train", "val"]
        self.transform = self.trnsfm[transform]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
          Args:
            index (int): Index
          Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path
    
    def remove_label(self, label):
        """
        Method has been deprecated. Please use 
        add_data(path, ignore_labels = True)
        method instead
        
        Removes a label from the dataset by relabelling
        images with the "label" for unlabelled images
        
        label (string): Real name of label that is to be
            removed
        set_num : numerical encoding of unlabelled images
        """
        
        # Raise warning that method has been deprecated
        logging.warning(
            """
                Method has been deprecated. Please use 
                add_data(path, ignore_labels = True)
                method instead
            """
        )
        
        # Raise error if unlabelled folder cannot be found
        if not (label in self.classes):
            raise ValueError("""
                Folder containing unlabelled data could not be found.
                Please ensure that folder name passed into
                "unlabelled_folder_name" is exactly the same as the
                name of the folder containing unlabelled data.
            """)
        
        # Find numerical encoding of unlabelled imags
        unlabelled_data_class_number = self.class_to_idx[label]
        
        # Remove the unlabelled class
        del self.classes[unlabelled_data_class_number]
        
        # Find unlabelled data
        numpy_targets = np.array(self.targets, dtype = np.object_)
        numpy_targets[numpy_targets == unlabelled_data_class_number] = self.unlab_class_num
        numpy_targets[numpy_targets > unlabelled_data_class_number] = numpy_targets[numpy_targets > unlabelled_data_class_number] - 1
        
        # Remove labels for unlabelled data
        self.targets = numpy_targets.tolist()
        self.class_to_idx =  dict([(name, i) for i, name in enumerate(self.classes)])
        self.samples = [(self.samples[i][0], self.targets[i]) for i in range(len(self.samples))]
        self.imgs = self.samples
    
    
    def update_labels(self, new_labels):
        """
        Assigns new labels to members of dataset
        
        new_labels (dict) : Dictionary of new labels.
            Key must be the index of the images in the
            dataset that will be assigned new labels.
            Values should be the names of the new labels
            as strings
        """
        
        # For each new label
        for img_index, label in new_labels.items():
            
            # Validate label
            if not (label in self.classes):
                raise ValueError("""
                    A label of {} for image number {} is not a valid
                    label. Please assign a label from the list of available 
                    classes.
                """.format(label, img_index))
            
            # Find numeric code for label
            class_num = self.class_to_idx[label]
            
            # Assign new label
            self.targets[img_index] = class_num
            self.samples[img_index] = (self.samples[img_index][0], class_num)
            self.imgs[img_index] = (self.imgs[img_index][0], class_num)
    
    def get_snapshot(self):
        """
        Compiles dataset into a pandas dataframe 
        of filenames and labels
        
        Returns Pandas dataframe with two columns:
            - path : path to image file relative to
                root of data directory
            - labels : label of image
        """
        
        paths = [] # File paths of images
        labels = [] # Labels of images
        
        
        for path, label in self.samples:
            
            # Remove root dir from path and compile
            paths.append(os.path.relpath(path, self.root))
            
            # Decode label
            if label == self.unlab_class_num:
                
                decoded_label = 'unlabelled'
            else:
                decoded_label = self.classes[label]
            
            # Compile decoded label
            labels.append(decoded_label)
        
        
        # Compile data as Pandas Dataframe
        return pd.DataFrame({
            'path' : paths,
            'label' : labels
        })
        
    def get_path_csv(self, inds):
        """
        Returns file paths for selected
        images as a pandas dataframe
        
        inds (list of int): indices
            of images to be included in
            the pandas dataframe
            
        Returns dataframe with an index
        that is the indices that were passed
        to the function and one column called
        'path' which is the absolute file
        path of those images
        """
        paths = []
        
        for ind in inds:
            _, _, path = self[ind]
            paths.append(path)
            
        return pd.DataFrame({
            'index' : inds,
            'path' : paths
        }).set_index('index')
    
    def get_transform(self, trns_mode):
        transform_list = []
        transform_list.append(Resize((256, 256)))
        if trns_mode == 'train':
            transform_list.append(RandomCrop((224, 224)))
            transform_list.append(RandomGrayscale())
            transform_list.append(RandomOrder(
                [RandomHorizontalFlip(), ColorJitter(), RandomRotation(20)]))
        else:
            transform_list.append(CenterCrop((224, 224)))
            
        transform_list.append(ToTensor())
        transform_list.append(Normalize(self.mean, self.std))
        if trns_mode == 'train':
            transform_list.append(RandomErasing(value='random'))

        return Compose(transform_list)

    def calc_mean_std(self):
        
        means = np.zeros((3))
        stds = np.zeros((3))
        sample_size = min(len(self.samples), 10000)
        for i in range(sample_size):
            img = self.loader(random.choice(self.samples)[0])
            stat = ImageStat.Stat(img)
            means += np.array(stat.mean)/255.0
            stds += np.array(stat.stddev)/255.0
        means = means/sample_size
        stds = stds/sample_size
        
        return means, stds

    def getClassesInfo(self):
        return self.classes, self.class_to_idx

    def getBalancedLoader(self, P= 10, K= 10, num_workers = 4, sub_indices= None, transfm = 'train'):
        self.setTransform(transfm)
        if sub_indices is not None:
            subset = Subset(self, sub_indices)
            train_batch_sampler = BalancedBatchSampler(subset, n_classes = P, n_samples = K)
            return DataLoader(subset, batch_sampler = train_batch_sampler, num_workers = num_workers)
        train_batch_sampler = BalancedBatchSampler(self, n_classes = P, n_samples = K)
        return DataLoader(self, batch_sampler = train_batch_sampler, num_workers = num_workers)

    def getSingleLoader(self, batch_size = 128, shuffle = True, num_workers = 4, sub_indices= None, transfm = 'train'):
        self.setTransform(transfm)
        if sub_indices is not None:
            return DataLoader(Subset(self, sub_indices), batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)   
        return DataLoader(self, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers)
    
    
    def get_labelled_indices(self):
        """
        Get indices of all labelled data
        
        Returns list of indices of all labelled data.
        These indices can be passed to __getitem__ to
        get the labelled images
        """
        
        # Note, the ">= 0" comparison only works because self.unlab_class_num
        # is a negative number
        return np.argwhere(np.array(self.targets) >= 0).flatten().tolist()
    
    def add_data(self, base_folder, ignore_labels = False):
        """
        Adds data to the dataset. As it uses the ImageFolder
        PyTorch class, all image data must exist in subfolders
        within the directory, even if the data is unlabelled.
        Also note, if labels aren't ignored then the labels
        in the new folder must all be found in the original
        dataset.
        
        base_folder (path or string) : Absolute file path
            to folder that contains the new images
        ignore_labels (Bool) : If True, labels provided by
            the first layer of subfolders will be ignored
            and the data will be considered to be unlabelled
                
        Return indices of new data after it has been added to
        the data set (for reference, the new data is always
        added to the end of the indices
        """
        
        # Read new images
        new_dataset = ImageFolder(base_folder)
        
        # If labels should be ignored
        if ignore_labels:
    
            # Unlabel classes
            new_dataset.targets = [self.unlab_class_num] * len(new_dataset.targets)
            new_dataset.samples = [(sample[0], self.unlab_class_num) for sample in new_dataset.samples]
            
            
        # If labels should be included
        else: 
            
            # Validate new labels
            for label in new_dataset.classes:
                if not (label in self.classes):
                    raise ValueError('{} is not a valid species. Please only use the following species {}'.format(label, self.classes))
            
            # Change class numbers to that of original dataset
            # (Has no effect if all species are present in new data)
            new_targets = [self.class_to_idx[new_dataset.classes[target]] for target in new_dataset.targets]
            new_dataset.targets = new_targets
            new_dataset.samples = [(new_dataset.samples[i][0], new_targets[i]) for i in range(len(new_dataset.samples))]
        
        # Get future indices of new dataset
        idx = list(range(len(self), len(self) + len(new_dataset)))
        
        # Add data to dataset
        self.targets = self.targets + new_dataset.targets
        self.samples = self.samples + new_dataset.samples
        self.imgs = self.samples
        
        # Return indices of new data
        return idx
        
        
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, underlying_dataset, n_classes, n_samples):
        
        self.labels = [data[1] for data in underlying_dataset]
        self.labels_set = set(self.labels)
        
        # Cap n_classes at number of actual classes
        if n_classes > len(self.labels_set): n_classes = len(self.labels_set) 
        
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = underlying_dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            #print(self.labels_set, self.n_classes)
            classes = np.random.choice(
                list(self.labels_set), self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                   class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // (self.n_samples*self.n_classes)