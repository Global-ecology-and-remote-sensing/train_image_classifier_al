import argparse
import os
import time
import pickle
import logging
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from camera_trap_al.deep_learning.data_loader import ExtendedImageFolder
from camera_trap_al.deep_learning.engine import Engine
from camera_trap_al.deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet
from camera_trap_al.deep_learning.utils import getCriterion
from camera_trap_al.deep_learning.active_learning_manager import ActiveLearningEnvironment
from camera_trap_al.utils import glob_vars # Import global variables
from camera_trap_al.utils.objects import CheckpointManager, LabelRetriever

def validate_choice(variable_name, choice, valid_choices):
    """
    Raises an error if the option chosen by
    the user isn't one of the options that they
    could choose from
    
    variable_name : Name of the variable that the user made the choice for
    choice : Choice that the user made
    valid_choices : List of all possible choices
    """
    
    if not (choice in valid_choices):
        raise ValueError('{} is an invalid {}. Please choose one from the following {}'.format(choice, variable_name, valid_choices))

def run_active_learning(
    train_data,
    unlabelled_data,
    validation_data = None,
    validate_model : bool = True,
    use_checkpoints : bool = True,
    num_workers : int = 0,
    active_batch : int = 100, 
    active_learning_strategy : str = 'margin',
    output_dir = 'default',
    use_pretrained : bool = True,
    embedding_arch : str = 'resnet18',
    embedding_loss_type : str =  'triplet',
    embedding_loss_margin : float = 1.0,
    embedding_loss_data_strategy : str = 'random',
    normalize_embedding : bool = True,
    feat_dim : int = 256,
    extract_embedding_batch_size : int = 256,
    embedding_finetuning_period = 2000,
    embedding_finetuning_lr : float = 0.0001,
    embedding_finetuning_weight_decay = 0,
    embedding_finetuning_num_epochs : int = 20,
    embedding_finetuning_loader_type : str = 'balanced',
    embedding_train_lr : float = 0.00001,
    embedding_train_weight_decay = 0.0005,
    embedding_train_num_epochs : int = 5,
    embedding_train_loader_type : str = 'single',
    balanced_loader_num_classes : int = 20,
    balanced_loader_num_samples : int = 10,
): 
    """
    PyTorch ImageNet Training
    
    train_data (path or string) : Path to labelled train dataset. (Required)
    unlabelled_data (path or string) : Path to dataset of unlabelled 
        images. All data must be contained in subfolders of this directory;
        images that aren't in subfolders won't be read. Images that are labelled
        by this program should not be moved from the unlabelled_data folder
        even if they appear in labels.csv CSV. (Required)
    validation_data (path or string) : Path to validation dataset.
        Data must be organised
        in the same way as the training data but all training data
        species need not be present. (Default : None)
    validate_model (bool) : If True and validation data exists, the model will be 
        tested on the validation set as it is being trained. (Default : True)
    use_checkpoints (bool) : If True, Function will load model and data mappings from
        a checkpoint, if it exists that is. Otherwise, it will train a model
        from scratch. Checkpoints are always overwritten by this program and so
        if this is set to False then any progress made by checkpoints will be lost.
        (Default : True)
    num_workers (int) :  Number of workers that will
        train the models and extract features in parallel.
        WARNING num_workers must be set to 0 if running on a Windows machine
        or else the program will freeze indefinitely. This is because Windows
        OS blocks multi-processing requests from PyTorch. (Default: 0)
    active_batch (int) : Number of queries per batch (Default : 100)
    active_learning_strategy (str) : Strategy for choosing which images to label
        (Default : 'margin')
    output_dir (str or PATH object) : Absolute filepath where checkpoints, labels
        and final trained model will be stored. By default, the program will create
        a folder in the working directory. (Default : 'default')
    use_pretrained (bool) : If True, the program will implement transfer learning using
        the pre-trained weights of the base model. This parameter is ignored if a
        model is loaded from a checkpoint. (Default : True)
    embedding_arch (str) : Architecture of embedding model. 
        Ignored if model is loaded from checkpoint. (Default : 'resnet18')
    embedding_loss_type (str) : Loss function of classifier. (Default: triplet loss)
    embedding_loss_margin (float): Margin for siamese or triplet loss (Default : 1.0)
    embedding_loss_data_strategy (str) : Data selection strategy for embedding model's
        loss function (Default : 'random')
    normalize_embedding (bool) : If True, embedding values will be 
        normalised as to avoid bias caused by dominating features when active 
        learning is performed. It is highly recommended that this parameter be True
        unless you are certain that the features extracted by the embedding model
        will be of the same scale (Default : True)
    feat_dim (int): Number of features that the embedding model should extract
        from the images. Ignored if model is loaded from checkpoint. (Default : 256)
    extract_embedding_batch_size (int): Batch Size when features are extracted
        from images. (Default: 256)
    embedding_finetuning_period (int) : Number of images between when the embedding
        is finetuned. (Default : 2000)
    embedding_finetuning_lr (float) : Learning rate when embedding model is
        being finetuned. (Default : 0.0001)
    embedding_finetuning_weight_decay (float) : L2 regularisation parameter for 
        when embedding model is being finetuned. (Default: 0)
    embedding_finetuning_num_epochs (int) : Number of total epochs to run for finetuning.
        (Default : 20)
    embedding_finetuning_loader_type (str) : Can either be 'single' or 'balanced'. If 'single',
        a loader that simply shuffles the data with a batch size of 128 images will be performed.
        If 'balanced', images will be sampled so that the number of images from each class in
        a batch are the same. If necessary, the balanced sampler will re-use images from a class
        if it does not have enough images. (Default : 'balanced')
    embedding_train_* : The embedding_train_* parameters are the same as the embedding_finetuning_*
        parameters except that they are used when the embedding model is initally trained on the
        training set. (Default values: lr = 0.00001, weight_decay = 0.0005, num_epochs = 5, 
        loader_type = 'single')
    balanced_loader_num_classes (int) : Number of classes to sample from at
        each batch of the balanced loader. Capped at number of classes in train_data.
        Note, batch size of balanced loader is num_classes * num_samples. 
        (Default : 20)
    balanced_loader_num_samples (int) : Number of images to sample from each
        class per batch of the balanced loader. Note, batch size of balanced
        loader is num_classes * num_samples. (Default : 10)
    """
    # .format(al_strat = AL_STRATEGY_TYPES, loss_type = LOSS_TYPES, finetune_strat = FINETUNING_TYPES)
    
    # Validate Input
    validate_choice('active_learning_strategy', active_learning_strategy, glob_vars.AL_STRATEGY_TYPES)
    validate_choice('embedding_loss_type', embedding_loss_type, glob_vars.LOSS_TYPES)
    validate_choice('embedding_loss_strategy', embedding_loss_data_strategy, glob_vars.EMBEDDING_LOSS_DATA_STRATEGIES)
    validate_choice('embedding_arch', embedding_arch, glob_vars.ARCHITECTURES)
    
    # Check if a GPU is available
    device = "cuda" 
    if not torch.cuda.is_available():
        device = "cpu"
        logging.warning('No GPU was detected; using CPU instead')
    
    # Set up work dir
    if output_dir == 'default':
        output_dir = os.path.join(os.getcwd(), 'Active Learning Files')
    
    # Set up checkpoint manager
    chp = CheckpointManager(output_dir, device = device)
    
    # If checkpoint exists and should be used
    if use_checkpoints and chp.validate_checkpoint():
    #if False:
        
        # Log info that program will start from checkpoint
        logging.info('Starting from checkpoint')
        
        # Load dataset and models from checkpoint 
        env, classifier = chp.load_checkpoint(
            active_learning_strategy = active_learning_strategy,
            optim_params = {
                'lr' : embedding_finetuning_lr,
                'weight_decay' : embedding_finetuning_weight_decay,
            }
        )
    
    # Otherwise, set up data loader and train new model on labelled data
    else:
        
        # Log that checkpoint will not be used
        logging.info(
            """
            Will not start from checkpoint either 
            because it should not be used or could 
            not be found
            """
        )
        
        # Setup the target dataset on training data
        dataset = ExtendedImageFolder(train_data)
        
        # Instantiate embedding model
        if embedding_loss_type.lower() == 'softmax':
            embedding_model = SoftmaxNet(
                architecture = embedding_arch, 
                feat_dim = feat_dim, 
                num_classes = len(dataset.classes), 
                use_pretrained= use_pretrained
            )
        else:
            embedding_model = NormalizedEmbeddingNet(
                architecture = embedding_arch, 
                feat_dim = feat_dim, 
                use_pretrained= use_pretrained
            )
            
        # Set up embedding model with parallel processing
        embedding_model = torch.nn.DataParallel(embedding_model)
        
        # Set up engine for training embedding model
        optimizer = torch.optim.Adam( # Optimiser object
            embedding_model.parameters(), 
            lr = embedding_train_lr, 
            weight_decay = embedding_train_weight_decay
        )
        criterion = getCriterion( # Loss criterion
            loss_type = embedding_loss_type, 
            strategy = embedding_loss_data_strategy, 
            margin = embedding_loss_margin
        ) 
        engine= Engine(
            device = device, 
            model = embedding_model, 
            criterion = criterion, 
            optimizer = optimizer,
            arch_type = embedding_arch,
            feat_dim = feat_dim
        )
        
        # Setup an active learning environment (Manages data handling)
        env = ActiveLearningEnvironment(
            dataset = dataset, 
            engine = engine,
            active_learning_strategy = active_learning_strategy
        )
        
        # Add unlabelled data to the dataset
        env.add_data(root_dir = unlabelled_data, data_type = 'unlabelled')
        
        # If Validation data exists
        if validation_data is not None:
            
            # Add validation data to dataset
            env.add_data(root_dir = validation_data, data_type = 'val')
        
        # Train Embedding model
        env.train_embedding_model(
            epochs = embedding_train_num_epochs,
            loader_type = embedding_train_loader_type, 
            num_classes = balanced_loader_num_classes,
            num_samples = balanced_loader_num_samples,
            num_workers = num_workers
        )
        
        # Extract embedding
        env.updateEmbedding(
            normalize = normalize_embedding,
            batch_size = extract_embedding_batch_size, 
            num_workers=num_workers
        )
        
        # Create a classifier (WILL NEED TO MODIFY THIS BIT FOR A CUSTOM CLASSIFIER)
        classifier = MLPClassifier(hidden_layer_sizes=(150, 100), alpha=0.0001, max_iter= 2000)
        
        # Train Classifier
        X_train, y_train = env.get_train_set()
        classifier.fit(X_train, y_train)
        
        # Setup Engine for finetuning
        env.update_optimizer(
            lr = embedding_finetuning_lr, 
            weight_decay = embedding_finetuning_weight_decay, 
        )
        
        
        # Save Checkpoint
        chp.save_checkpoint(env, classifier)
    
    # Object for inputting labels
    lblr = LabelRetriever(
        unlabelled_dir = unlabelled_data,
        work_dir = output_dir,
        data_paths = env.get_path_csv('default')
    )
    
    
    while env.get_pool_size('default') > 0:
        
        # If test data exists
        if env.val_pool_exists() and validate_model:
            
            # Check performance
            X_test, y_test = env.get_test_set()
            acc = classifier.score(X_test, y_test) # Overall Accuracy
            y_pred = classifier.predict(X_test)
            cm = confusion_matrix(y_test, y_pred) # Confusion matrix
            norm_mat = confusion_matrix(y_test, y_pred, normalize = 'true') # Row normed conf mat (main diagonal shows recall)
            macro_pc_acc = precision_score(y_test, y_pred, average = 'macro', zero_division = 0) # Macro Precision
            macro_rec_acc = recall_score(y_test, y_pred, average = 'macro', zero_division = 0) # Macro Recall
            micro_pc_acc = precision_score(y_test, y_pred, average = 'micro', zero_division = 0) # Micro Precision
            micro_rec_acc = recall_score(y_test, y_pred, average = 'micro', zero_division = 0) # Micro Recall
            
            # Get classes in confusion matrix
            test_classes = np.array(env.dataset.classes)
            test_classes = test_classes[np.unique(np.concatenate((y_test, y_pred)))].tolist()
            
            # Record accuracy
            chp.record_test_results(
                name = 'train_set_size_' + str(env.get_pool_size('active')),
                conf_mat = cm,
                norm_mat = norm_mat,
                conf_mat_classes = test_classes,
                test_metrics = {
                    'accuracy' : acc,
                    'micro_precision' : micro_pc_acc,
                    'micro_recall' : micro_rec_acc,
                    'macro_precision' : macro_pc_acc,
                    'macro_recall' : macro_rec_acc
                }
            )
            
        # If not currently waiting for labels
        if not lblr.label_request_exist():
            
            # Find best images to label using active learning
            imgs_to_label = env.sample_labels(N = active_batch, model = classifier)
            
            # Save Request for labels
            lblr.save_label_request(imgs_to_label)
        
        else:
            
            # Load label request 
            imgs_to_label = lblr.load_label_request()
        
        # Ask and wait for labels
        new_labels = lblr.request_labels(imgs_to_label)
        
        # Assign new labels
        env.add_new_labels(new_labels)
        
        # Finetune the embedding model and calculate new embedding values
        if env.num_new_labels >= embedding_finetuning_period : 
            
            env.train_embedding_model(
                epochs = embedding_finetuning_num_epochs,
                loader_type = embedding_finetuning_loader_type, 
                num_classes = balanced_loader_num_classes,
                num_samples = balanced_loader_num_samples,
                num_workers = num_workers
            )
            
            
            env.updateEmbedding(
                normalize = normalize_embedding,
                batch_size=extract_embedding_batch_size, 
                num_workers=num_workers
            )
        
        # gather labeled pool and train the classifier
        X_train, y_train = env.get_train_set()
        
        classifier.fit(X_train, y_train)
        
        # If embedding was updated
        if env.num_new_labels >= embedding_finetuning_period :
            
            # Reset counter for number of new labels
            env.num_new_labels = 0 
        
        # Delete labels file
        lblr.delete_label_request()
        
        # Save Checkpoint
        chp.save_checkpoint(env, classifier)
        
    logging.info('No unlabelled images remain. Terminating program')
    
    
    # Try to convert model to ONNX
    
    
if __name__ == '__main__':
    main() 