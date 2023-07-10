"""
Changelog

    - Added class called "PooledALSampler" which performs
        active learning on active and default pools of
        indices of an embedding rather than the whole
        embedding. This was to allow for a validation
        pool and other future pools to be added to the
        environment
"""

from .engine import Engine
import numpy as np
import torch
import random
import sys
import logging
from camera_trap_al.utils import glob_vars # Import global variables
from camera_trap_al.active_learning_methods.constants import get_AL_sampler, get_wrapper_AL_mapping

class PooledALSampler():
    """
    Helper class for ActiveLearningEnvironment object
    that wraps around the active learning sampler.
    Main purpose is to translate indices from data
    pools into indices used by the sampler and vice
    versa
    """
    
    def __init__(self, al_strat, embedding, al_pool, classes):
        """
        Sets up sampler and key between AL environment
        indices and sampler indices
        
        embedding (2D NumPy array) : Full embedding
            of dataset
        al_pool (list) : Row indices of embedding that
            are to be seen by the sampler. Should be
            the union of the default and active pools
        classes (List) : Classes of the dataset.
            Only used in clustering AL algorithms to determine
            the number of clusters which is taken as the number
            of classes
        """
        
        # Get key for changing sampler indices to indices
        # from active learning pools
        self.sampler_ind_to_pool_ind = np.unique(al_pool)
        
        # Enforce uniqueness of indices
        if len(self.sampler_ind_to_pool_ind) != len(al_pool):
            raise AssertionError('AL pool indices are not unique. Cannot construct sampler')
        
        # Construct sampler
        self.sampler = get_AL_sampler(al_strat)(
            X = embedding[self.sampler_ind_to_pool_ind], # Subset embedding
            y = classes, 
            seed = random.randint(0, sys.maxsize * 2 + 1)
        )
    
    def select_batch(self, N, active_pool, model):
        """
        Selects a batch of data to label from the
        default pool
        
        N (int) : Number of data points to sample
        active_pool (list of int) : Indices in
            full embedding that already have
            labels
        model (scikit-learn classifier) : The model
            that is to be trained
        """
        
        # Get active pool indices in sampler
        sampler_active_ind = np.searchsorted(self.sampler_ind_to_pool_ind, active_pool)
        
        # Find optimum images to label using AL
        sampler_indices = self.sampler.select_batch(
            N = N, 
            already_selected = sampler_active_ind, 
            model = model
        )
        
        # Get data indices as they appear in AL pools
        pool_indices = self.sampler_ind_to_pool_ind[sampler_indices]
        
        # Return pool indices as base Python list of integers
        return list(map(int, pool_indices))
    
class ActiveLearningEnvironment(object):
    """
    Handles data including moving data between unlabelled and
    labelled pools. Also utilises the "engine" which is the
    object that feeds data into and trains the embedding model
    
    Changelog
    
    
    - Changed starting values of active_pool and default_pool
        in __init__ function
    - Modified updateEmbedding function to only find the embeddings for data
        that lies in either the active or default pools
    - Added options to change hyperparameters in training and finetuning
        parameters for embedding model
    - Added option in finetune_embedding to use either the balanced data
        loader or the shuffled sequential loader
    - Renamed finetune_embedding to train_embedding_model
    - Added function called update_engine that re-initialises the engine
        that's used to train the embedding with different hyperparameters
    - Removed setPool function and self.current_pool variable as they aren't
        used by any other function in package
    - Removed self.criterion variable from environment to avoid confusion
        when that attribute is updated but criterion used by the engine
        isn't updated as it might have been passed by value
    - Similarly, deleted self.device and self.embedding_model
    - Changed __init__ function to require the training engine to be
        initialised outside of the object to make it consistent with
        the rest of the parameters
    - Renamed ActiveLearningManager class to ActiveLearningEnvironment
    - Moved code that sets up the AL sampler from run_active_learning.py
        function to UpdateEmbedding function
    """

    #constructor
    def __init__(self, dataset, engine,
        active_learning_strategy, data_pools = None, num_new_labels = 0,
        embedding = None):
        """
        
        active_learning_strategy : Strategy for the ActiveLearningSampler
        num_new_labels (int) : The number of labelled images that have
            been added to the active pool since the embedding
            model was last finetuned
        """
    
        # Set up attributes
        self.dataset = dataset
        self.engine = engine
        self.embedding = embedding
        self.sampler = None
        self.num_new_labels = num_new_labels
        self.al_strat = active_learning_strategy
        
        # Set up active learning global variables
        get_wrapper_AL_mapping()
        
        # Default value of pools
        self.val_pool = []
        self.default_pool = []
        
        # If no data_pools were passed
        if data_pools is None:
        
            # Take all labeled data as active pool
            self.active_pool = dataset.get_labelled_indices()
            
            # Take all unlabeled data as default pool
            self.default_pool = list(set(list(range(len(dataset)))).difference(self.active_pool))
            
        # Use custom data pools if its of the right type    
        elif type(data_pools) is dict :
            
            self.active_pool = data_pools['active']
            self.default_pool = data_pools['default']
            
            # If a test pool exists
            if 'val' in data_pools.keys():
                
                # Assign validation pool
                self.val_pool = data_pools['val']
            
        # Else, return type error
        else:
            
            raise TypeError(
                """
                data_pools parameter was of the wrong type.
                Only None types and Dictionaries of lists of indices
                are acceptable
                """
            )
        
        
    #update embedding values after a finetuning
    def updateEmbedding(self, normalize = True, batch_size = 256, num_workers = 4):
        
        logging.info('Extracting embedding...')
        
        # Update embedding
        self.embedding = self.engine.embedding(
            self.dataset.getSingleLoader(
                batch_size = batch_size, 
                shuffle = False, 
                num_workers = num_workers, 
                sub_indices= list(range(len(self.dataset))), 
                transfm ="val"
            ), 
            normalize = normalize
        )
        
        logging.info('Finished extracting embedding')
        
        # Update the sampler on new embedding
        self.update_sampler()
        
    def update_sampler(self, al_strat = None):
        """
        Update sampler on embedding with
        new active learning strategy if provided
        
        al_strat (string): New AL strategy.
            Defaults to the one saved by the object
            if none is passed
        """
        
        # Use saved AL strategy if none is provided
        if al_strat is None: al_strat = self.al_strat
        
        # Update active learning sampler
        self.sampler = PooledALSampler(
            al_strat = al_strat,
            embedding = self.embedding, 
            al_pool = self.active_pool + self.default_pool,
            classes = self.dataset.classes
        )
    
    def get_test_set(self):
        """
        Gets test set for classifier. Data is provided as a 2D NumPy
        array and labels are provided as a list of numerical encodings
        """
        return self.embedding[self.val_pool], np.asarray([self.dataset.samples[x][1] for x in self.val_pool])
    
    def get_train_set(self):
        """
        Gets train set for classifier. Data is provided as a 2D NumPy
        array and labels are provided as a list of numerical encodings
        """
        return self.embedding[self.active_pool], np.asarray([self.dataset.samples[x][1] for x in self.active_pool])

    def train_embedding_model(self, epochs, loader_type, batch_size = 128,
        num_classes = 20, num_samples = 10, num_workers=10):
        """
        Trains/finetunes the embedding model over the labelled images.
        
        epochs (int) : Number of epochs to train model.
        loader_type : Method of loading data into model. Can
            either be 'single' or 'balanced'.
        batch_size (int) : Batch size of data loader
        num_classes (int) : Number of classes to sample from 
            at each batch of a balanced loader
        num_samples (int) : Number of samples to take from 
            each class in each batch of a balanced
        num_workers : Number of parallel processes to use
            to train the model.
        """
        
        # If single loader should be used
        if loader_type == 'single':
            
            # Build Single Loader
            train_loader = self.dataset.getSingleLoader(
                batch_size = batch_size,
                shuffle = True, 
                num_workers = num_workers,
                sub_indices= self.active_pool,
                transfm = 'train'
            )
        
        # If balanced loader should be used
        elif loader_type == 'balanced':
            
            
            # Define balanced data loader
            train_loader = self.dataset.getBalancedLoader(
                P = num_classes, 
                K = num_samples, 
                num_workers = num_workers, 
                sub_indices = self.active_pool,
                transfm = 'train'
            )
            
        # Raise error if data loader is not recognised
        else:
        
            raise ValueError("""
                '{}' is an invalid data loader for updating
                the embedding model. Please choose a valid 
                loader type.
            """.format(loader_type))
        
        logging.info('Training embedding model...')
        
        for epoch in range(epochs):
            self.engine.train_one_epoch(train_loader, epoch, False)
        
        logging.info('Finished training embedding model...')
        
    # a utility function for saving the snapshot
    def get_pools(self):
        return {"embedding":self.embedding, "active_indices": self.active_pool, "default_indices":self.default_pool,
               "active_pool":[self.dataset.samples[self.dataset.indices[x]] for x in self.active_pool], 
               "default_pool":[self.dataset.samples[self.dataset.indices[x]] for x in self.default_pool]}
    
    
    def update_optimizer(self, lr, weight_decay):
        """
        Creates a new engine with the provided
        parameters
        """
        
        self.engine.optimizer = torch.optim.Adam(
            self.engine.model.parameters(), 
            lr = lr, 
            weight_decay = weight_decay
        )
        
    def add_new_labels(self, new_labels):
        """
        Add new labels to environment and update
        active and default pools
        
        new_labels (dict) : Dictionary of new labels.
            Key must be the index of the images in the
            dataset that will be assigned new labels.
            Values should be the names of the new labels
            as strings
        """
        
        self.dataset.update_labels(new_labels)
        
        # Move newly labelled data to labelled dataset
        self.active_pool.extend(new_labels.keys())
        self.default_pool = list(set(self.default_pool).difference(new_labels.keys()))
        
        self.num_new_labels += len(new_labels)
        
    def sample_labels(self, N, model):
        """
        Sample images from the unlabelled pool using
        active learning
        
        Returns list of indices
        """
        return self.sampler.select_batch(
            N = N, 
            active_pool = self.active_pool, 
            model = model
        )
        """
        # Find optimum images to label using AL
        indices = self.sampler.select_batch(
            N = N, 
            already_selected = self.active_pool, 
            model = model
        )
        
        # Change indices to Python integers
        indices = list(map(int, indices))
        
        # Get absolute filepaths of images
        #paths = [self.dataset[x][1] for x in indices]
        
        # Return indices and paths
        #return list(zip(indices, paths))
        
        return indices
        """
    
    def get_path_csv(self, data_pool):
        """
        Returns file paths for chosen data
        pools
        
        data_pool (string or list of strings) : 
            Names of datas
        """
        
        # If datapool is not a list, make it a singleton list
        if not isinstance(data_pool, list): data_pool = [data_pool]
        
        # Validate chosen data pools
        assert set(data_pool).issubset(glob_vars.AL_POOL_TYPES)
        
        # Set up pool for chosen data
        joined_pool = []
        
        # Add chosen data to joined pool
        if 'active' in data_pool: joined_pool.extend(self.active_pool)
        if 'default' in data_pool: joined_pool.extend(self.default_pool)
        if 'val' in data_pool: joined_pool.extend(self.val_pool)
        
        # Return filepath dataframe for chosen pools
        return self.dataset.get_path_csv(joined_pool)
    
    def add_data(self, root_dir, data_type):
        """
        Adds data to the dataset and the active learning
        pools. Does not update the embedding which will
        need to be updated outside of this method before
        active learning is performed.
        
        root_dir (string or path) : Path to new data
        data_type (string) : Type of data. Determines
        which pool data is added to. Can either be
        'train', 'val' or 'unlabelled'
        """
        
        # Validate data type
        assert data_type in glob_vars.DATASET_TYPES
        
        # Check if new data has labels
        ignore_labels = True if data_type == 'unlabelled' else False
        
        # Add data to dataset
        new_idx = self.dataset.add_data(root_dir, ignore_labels = ignore_labels)
        
        # Add new data to correct pool
        if data_type == 'unlabelled':
            self.default_pool.extend(new_idx)
        elif data_type == 'val':
            self.val_pool.extend(new_idx)
        elif data_type == 'train':
            self.active_pool.extend(new_idx)
        else:
            raise NotImplementedError('Adding data for dataset type {} has not been implemented'.format(data_type))
    
    def val_pool_exists(self):
        """
        Returns True if there are images in the
        validation pool. Returns False otherwise
        """
        
        return len(self.val_pool) > 0
    
    def get_pool_size(self, pool_type):
        """
        Returns size of active pool
        """
        assert pool_type in glob_vars.AL_POOL_TYPES
        
        # Get appropriate pool
        if pool_type == 'active':
            pool = self.active_pool
        elif pool_type == 'default':
            pool = self.default_pool
        elif pool_type == 'val':
            pool = self.val_pool
        else:
            raise NotImplementedError('Getting length of pool with type "{}" has not been implemented'.format(pool_type))   
            
        # Return length of pool
        return len(pool)