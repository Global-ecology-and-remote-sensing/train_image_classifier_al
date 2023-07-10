# Classes used by the main program

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import logging
import time
import warnings
from textwrap import dedent

from camera_trap_al.deep_learning.utils import save_embedding_model, load_embedding_model
from camera_trap_al.deep_learning.networks import NormalizedEmbeddingNet, SoftmaxNet
from camera_trap_al.deep_learning.data_loader import ExtendedImageFolder
from camera_trap_al.deep_learning.engine import Engine
from camera_trap_al.deep_learning.utils import getCriterion
from camera_trap_al.deep_learning.active_learning_manager import ActiveLearningEnvironment
from os.path import join as jpth

from matplotlib import pyplot as plt
from matplotlib import colors

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skl2onnx import to_onnx

def plot_conf_mat(count_mat, norm_mat, categories, show_plot = True, path = None):
    """
    Plots a confusion matrix as an annotated heatmap. Copied
    from https://www.geeksforgeeks.org/how-to-draw-2d-heatmap-using-matplotlib-in-python/
    
    count_mat  : Confusion Matrix of value counts
    
    norm_mat (2-D numpy array): normalised confusion matrix
    """
    
    num_categories = count_mat.shape[0]
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
   
    # Create a custom color map
    # with blue and green colors
    colors_list = ['#0099ff', '#33cc33']
    cmap = colors.ListedColormap(colors_list)
     
    # Plot the heatmap with custom colors and annotations
    ax.imshow(norm_mat, vmin = 0, vmax = 1, cmap='summer')
    for i in range(count_mat.shape[0]):
        for j in range(count_mat.shape[1]):
            plt.annotate(str(count_mat[i][j]), xy=(j, i),
                         ha='center', va='center', color='black')
    
    
    # Add category names to X and Y axes
    ax.set_xticks(list(range(num_categories)))
    ax.set_yticks(list(range(num_categories)))
    
    ax.set_xticklabels(categories)
    ax.set_yticklabels(categories)
    
    plt.xticks(rotation=90, ha='right')
    
    plt.sca(ax)
    
    # Add colorbar
    #cbar = plt.colorbar(ticks=[0, 0.5, 1])
    #cbar.ax.set_yticklabels(['0', '0.5', '1'])
     
    # Set plot title and axis labels
    plt.title("Camera Trap Confusion Matrix", fontsize = 18)
    plt.xlabel("Predicted Label", fontsize = 16)
    plt.ylabel("True Label", fontsize = 16)
    
    if path is not None:
        plt.savefig(path)
    
    
    # Display the plot
    if show_plot:
        plt.show()


class CheckpointManager():
    """
    Object for saving, loading checkpoint data
    and managing checkpoint files for program that
    trains a classifier using active learning
    """

    def __init__(self, root_dir, device):
        """
        Set up directory structure of output folder
        which will also contain checkpoints
        
        root_dir : Directory where files will be stored
        device (str): Device that the program uses to
            train models (Can either be 'cuda' or 'cpu')
        """
        
        # Save Device
        assert device in ['cuda', 'cpu']
        self.device = device
        
        # Directories used by CheckpointManager
        self.dirs = {}
        self.dirs['root'] = root_dir
        self.dirs['data'] = jpth(self.dirs['root'], 'data') # Data Loader, record of labels, request for labels
        self.dirs['embedding'] = jpth(self.dirs['root'], 'embedding') # Embedding and its model
        self.dirs['classifier'] = jpth(self.dirs['root'], 'classifier') # AL indices and classifier
        self.dirs['test_results'] = jpth(self.dirs['root'], 'test_results') # Validation results
        self.dirs['export'] = jpth(self.dirs['root'], 'export') # Trained model export files
        
        # File Paths for restoring state of training loop
        self.file_paths = {}
        self.file_paths['embedding'] = jpth(self.dirs['embedding'], 'embedding.npy')
        self.file_paths['embedding_model'] = jpth(self.dirs['embedding'], 'embedding_model_weights.pt')
        self.file_paths['pools'] = jpth(self.dirs['classifier'], 'active_learning_pools.json')
        self.file_paths['classifier'] = jpth(self.dirs['classifier'], 'classifier.pkl')
        self.file_paths['data_loader'] = jpth(self.dirs['data'], 'data_loader.pt')
        self.file_paths['label_record'] = jpth(self.dirs['data'], 'labels.csv')
        
        # File paths for exporting trained model
        self.file_paths['export_embedding_model'] = jpth(self.dirs['export'], 'embedding_model_weights.pt')
        self.file_paths['onnx_classifier'] = jpth(self.dirs['export'], 'classifier.onnx')
        self.file_paths['dataset_mean'] = jpth(self.dirs['export'], 'dataset_mean.npy')
        self.file_paths['dataset_std'] = jpth(self.dirs['export'], 'dataset_std.npy')
        
        # Make all subfolders in root_dir
        self.make_dirs()
    
    def make_dirs(self):
        """
        Create all subfolders in root directory if they
        do not already exist
        """
        
        # For every subfolder
        for directory in self.dirs.values():
            
            # Make subfolder if it doesn't exist
            if not os.path.isdir(directory): os.mkdir(directory)
    
    def validate_checkpoint(self):
        """
        Returns True if all checkpoint files exist
        and returns False otherwise
        """
        
        # First, assume that checkpoint is valid
        valid = True
        
        # Loop through all checkpoint files
        for file_path in self.file_paths.values():
            
            # If file does not exist
            if not os.path.exists(file_path):
                
                # Checkpoint is not valid
                valid = False
                break
        
        # Return True iff checkpoint exists
        return valid
    
    def save_checkpoint(self, env, classifier, output_labels = True):
        """
        Saves model, dataloader and objects used to
        train model. Does not save hyperparameters
        of embedding model's optimiser
        
        env (ActiveLearningEnvironment object) : Object
            containing (almost) all of the states in
            the active learning process
        embedding_arch (string) : Name of type of architecture
            of embedding used
        output_labels (bool) : If True, record of all 
            labelled data will be outputted to a CSV
        """
        logging.info('Saving checkpoint...')
        
        # Save data loader
        torch.save(
            obj = env.dataset,
            f = self.file_paths['data_loader']
        )
        
        # Save dataset fitted mean
        np.save(
            file = self.file_paths['dataset_mean'],
            arr = env.dataset.mean
        )
        
        # Save dataset fitted std
        np.save(
            file = self.file_paths['dataset_std'],
            arr = env.dataset.std
        )
        
        # Save embedding
        np.save(
            file = self.file_paths['embedding'],
            arr = env.embedding
        )
        
        # Pack embedding model
        packed_embedding_model = {
        
            # Model
            'arch_name' : env.engine.arch_type,
            'embedding_model_state_dict' : env.engine.model.state_dict(),
            'feat_dim' : env.engine.feat_dim,
            'num_classes' : len(env.dataset.classes),
            'classes' : env.dataset.classes,
            'class_to_idx' : env.dataset.class_to_idx,
            
            # Optimiser
            'optimizer_state_dict' : env.engine.optimizer.state_dict(),
            
            # Loss criterion
            'loss_type' : env.engine.criterion.loss_type,
            'loss_func_strategy' : env.engine.criterion.strategy,
            'loss_func_margin' : env.engine.criterion.margin,
        }
        
        # Save embedding model checkpoint
        torch.save(
            obj = packed_embedding_model,
            f = self.file_paths['embedding_model']
        )
        
        # Export embedding model
        torch.save(
            obj = packed_embedding_model,
            f = self.file_paths['export_embedding_model']
        )
        
        # Save AL pools
        with open(self.file_paths['pools'], 'w') as f:
            f.write(
                json.dumps({
                    'active' : env.active_pool,
                    'default' : env.default_pool,
                    'val' : env.val_pool,
                    'num_new_labels' : env.num_new_labels
                })
            )
        
        # Pickle classifier
        with open(self.file_paths['classifier'], 'wb') as f:
            pickle.dump(classifier, f)
        
        # Export classifier
        onx = to_onnx(classifier, env.embedding)
        with open(self.file_paths['onnx_classifier'], "wb") as f:
            f.write(onx.SerializeToString())
        
        # If labels should be saved
        if output_labels:
            
            # Save labels as CSV
            df = env.dataset.get_snapshot()
            df.to_csv(self.file_paths['label_record'], index = False)
        
        
        logging.info('Checkpoint saved successfully')
        
        
    def load_checkpoint(self, active_learning_strategy, optim_params = None):
        """
        Loads objects that are needed to perform active learning
        
        optim_params (dict) : If not None, keyworded paramters
            in optim_params will be unpacked and passed to
            the embedding model's Adam optimiser
        """
        
        # Load dataset
        dataset = torch.load(
            f = self.file_paths['data_loader']
        )
        
        # Load embedding
        embedding = np.load(
            file = self.file_paths['embedding']
        )
        
        # Load embedding model and its training objects
        embedding_model_settings = torch.load(
            f = self.file_paths['embedding_model'],
            map_location = torch.device(self.device)
        )
        
        # Load AL pools
        with open(self.file_paths['pools'], 'r') as f:
            
            al_pools = json.load(f)
        
        # Unpack embedding finetuning counter from al_pools
        num_new_labels = al_pools.pop('num_new_labels')
        
        # Load classifier
        with open(self.file_paths['classifier'], 'rb') as f:
            classifier = pickle.load(f)
        
        
        # Instantiate embedding model
        if embedding_model_settings['loss_type'].lower() == 'softmax':
            embedding_model = SoftmaxNet(
                architecture = embedding_model_settings['arch_name'], 
                feat_dim = embedding_model_settings['feat_dim'], 
                num_classes = len(dataset.classes), 
                use_pretrained = False
            )
        else:
            embedding_model = NormalizedEmbeddingNet(
                architecture = embedding_model_settings['arch_name'], 
                feat_dim = embedding_model_settings['feat_dim'], 
                use_pretrained = False
            )
        # Set up embedding model with parallel processing
        embedding_model = torch.nn.DataParallel(embedding_model)
        
        # Assign saved weights to embedding model
        embedding_model.load_state_dict(embedding_model_settings['embedding_model_state_dict'])
        
        # Set up engine for training embedding model
        optimizer = torch.optim.Adam( # Optimiser object
            embedding_model.parameters(), 
            **optim_params
        )
        
        optimizer.load_state_dict(embedding_model_settings['optimizer_state_dict'])
        
        criterion = getCriterion( # Loss criterion
            loss_type = embedding_model_settings['loss_type'], 
            strategy = embedding_model_settings['loss_func_strategy'], 
            margin = embedding_model_settings['loss_func_margin']
        )
        engine = Engine(
            device = self.device, 
            model = embedding_model, 
            criterion = criterion, 
            optimizer = optimizer,
            arch_type = embedding_model_settings['arch_name'],
            feat_dim = embedding_model_settings['feat_dim']
        )
        
        # Setup an active learning environment
        env = ActiveLearningEnvironment(
            dataset = dataset, 
            engine = engine,
            active_learning_strategy = active_learning_strategy,
            data_pools = al_pools,
            num_new_labels = num_new_labels,
            embedding = embedding
        )
        
        # Update the AL sampler on the embedding
        env.update_sampler()
        
        # Return AL environment and saved classifier
        return env, classifier
    
    def record_test_results(self, name, test_metrics, conf_mat = None, norm_mat = None, conf_mat_classes = None):
        """
        Records test results in the test_results folder
        
        name (string) : Name of test
        test_metrixs (dict) : Metrics used in test results
        conf_mat (NumPy array) : Confusion matrix of test results
        """
        
        # Make and Get directory for this record
        results_dir = jpth(self.dirs['test_results'], name)
        if not os.path.isdir(results_dir): os.mkdir(results_dir)
        
        # If a confusion matrix exists
        if conf_mat is not None and conf_mat_classes is not None:
            
            # Save confusion matrix
            np.save(
                file = jpth(results_dir,'confusion_matrix.npy'), 
                arr = conf_mat
            )
            
            # Add confusion matrix classes to test metrics
            test_metrics['conf_mat_classes'] = conf_mat_classes
            
            if norm_mat is not None:
            
                # Save plot of confusion matrix
                plot_conf_mat(
                    count_mat = conf_mat, 
                    norm_mat = norm_mat, 
                    categories = conf_mat_classes, 
                    show_plot = False, 
                    path = jpth(results_dir,'confusion_matrix.png')
                )
            
                
        
        # Save test metrics
        with open(jpth(results_dir, 'metrics.json'), 'w') as f:
            f.write(
                json.dumps(test_metrics)
            )
    

class LabelRetriever():
    """
    Handles label request to and retrieval from 
    the user as well as data validation
    """
    
    def __init__(self, unlabelled_dir, work_dir, data_paths):
        """
        unlabelled_dir (path or string) : Directory that
            contains unlabelled data. Must be the same directory
            that the Timelapse template file used to label
            the data was initially saved in.
        work_dir (path or string): Directory where requests
            for labels will be stored
        data_paths (Pandas Dataframe) : Filepaths of dataset.
            The index must be called 'index'
            and have the indices of the images as represented
            in the active and default pools. It must also have
            one column
            called 'path' that contains the absolute file paths
            of the images
        """
        def extract_timelapse_paths(sr, root_dir):
            """
            Extracts filepaths that are compatible with
            timelapse when it comes to updateting a
            database using a CSV
            
            sr (pandas series) : Must contain an element
                called 'path' which is the absolute file
                path that will be converted to a dataframe
            root_dir (path or string) : Directory that
                the file path will be made relative to.
                Absolute file path must lie within root_dir
            """
            
            # Make path relative
            relative_path = os.path.relpath(sr['path'], root_dir)
            
            # Split file and directory
            head, tail = os.path.split(relative_path)
            
            # Assign data to series according to
            # Timelapse's format
            sr['RelativePath'] = head
            sr['File'] = tail
            
            return sr
        
        # Directories
        self.unlabelled_root_dir = unlabelled_dir
        self.work_dir = work_dir
        self.label_bin_dir = jpth(self.work_dir, 'new_labels_bin') # Folder for user to place new labels
        self.label_request_dir = jpth(self.work_dir, 'data')
        
        # Files
        self.label_request_path = jpth(self.label_request_dir, 'label_request.json')
        self.timelapse_selector_path = jpth(self.work_dir, 'timelapse_selector.csv')
        
        # Make Directories
        if not os.path.isdir(self.label_bin_dir): os.mkdir(self.label_bin_dir)
        if not os.path.isdir(self.label_request_dir): os.mkdir(self.label_request_dir)
        
        # Dataframe used to submit label requests
        self.df_lr_template = data_paths.copy()
        
        # Convert filepaths dataframe to a 
        # version that's compatible with Timelapse
        self.df_lr_template = data_paths.apply(
            lambda x : extract_timelapse_paths(x, self.unlabelled_root_dir), 
            axis = 1
        )
        
        # Add column for selecting images for labelling
        self.df_lr_template['Selected'] = False
        
        # Stores new label input
        self.df_new_labels = None
        
        
    def save_label_request(self, label_request):
        """
        label_request : Program's request for
            labels if the program is waiting for
            labels
        """
        
        # Save request for labels as JSON
        with open(self.label_request_path, 'w') as f:
            f.write(
                json.dumps(label_request)
            )
    
    def label_request_exist(self):
        """
        Returns True if a request for labels
        has yet to be fulfilled
        """
        
        return os.path.exists(self.label_request_path)
        
    def load_label_request(self):
        """
        Loads request for labels. Will cause an error
        if request for labels does not exist
        """
        
        with open(self.label_request_path, 'r') as f:
            
            label_request = json.load(f)
        
        return label_request
    
    def delete_label_request(self):
        """
        Deletes request for labels
        """
        
        os.remove(self.label_request_path)
    
    def labels_file_valid(self, lr_indices, overwrite_new_labels = True):
        """
        Validates file that contains labels for
        new data. Function will only look
        at the first file that it finds in the
        labels bin
        
        lr_indices (list of int): Indices of 
            images that need labels
        overwrite_new_labels (bool) : If true, the file
            will be loaded as a pandas dataframe
            and saved as one of the object's
            attributes after it has been validated.
            Will overwrite any previously loaded
            new labels file
        
        Returns True if file is valid, returns
        False otherwise
        """
        
        
        if not os.path.isdir(self.label_bin_dir):
            print('Labels bin does not exist')
            return False
        
        
        # Search for labels file
        bin_files = os.listdir(self.label_bin_dir)
        
        # Stores path to new labels
        new_labels_path = None
        
        # Search through files in bin
        for file in bin_files:
            
            # If file is a CSV file
            if os.path.splitext(file)[1].lower() == '.csv':
                
                # Take that as the new labels
                new_labels_path = jpth(self.label_bin_dir, file)
                break
        
        # If no CSV files were found
        if new_labels_path is None:
            
            # File invalid
            print('There are no CSV files in the labels bin')
            return False
        
        # Try to load new labels
        try:
            
            df_new_labels = pd.read_csv(new_labels_path)
        
        except:
            
            print(dedent(
            """
                An error occurred while trying to load the
                first CSV in the labels bin. Please ensure
                that only the file for the new labels lies
                in the bin, only data and column names
                are recorded in the file and that it is encoded 
                using UTF-8 which uses only English characters.
            """))
            return False
            
        # Check that file has all necessary columns
        necessary_columns = ['File', 'RelativePath', 'Species']
        for col in necessary_columns:
            
            # If column doesn't exist
            if not (col in df_new_labels.columns) : 
                
                # File invalid
                print('{} column does not exist in new labels CSV'.format(col))
                return False
        
        # Recover absolute file paths of images
        df_new_labels['path'] = df_new_labels.apply(
            lambda x : jpth(
                self.unlabelled_root_dir,
                x['RelativePath'],
                x['File']
            ),
            axis = 1
        )
        
        # Remove rows that don't have labels
        df_new_labels.dropna(subset = 'Species', inplace = True)
        
        # Get filepaths of images that need labels
        sr_paths_to_be_labelled = self.df_lr_template.loc[lr_indices, 'path'].copy()
        paths_to_be_labelled = set(sr_paths_to_be_labelled.tolist())
        
        # Get paths of images that have labels
        paths_that_have_labels = df_new_labels['path'].tolist()
        
        # If not all images that need labels have labels
        if not(paths_to_be_labelled.issubset(paths_that_have_labels)):
            
            # Get images that weren't labelled
            paths_need_labels = list(paths_to_be_labelled.difference(paths_that_have_labels))
            
            # File invalid
            print(dedent(
                """
                    Not all label requests were fulfilled. Please label the
                    following images:
                    
                    {}
                """.format(paths_need_labels)
            ))
            return False
        
        # Get only the labels for the data that needs labels
        df_new_labels = df_new_labels[df_new_labels['path'].isin(paths_to_be_labelled)]
        
        # If an image has multiple new labels
        if df_new_labels['path'].duplicated().any():
            
            # File invalid
            print('The following images have multiple labels. Please ensure that each image has only 1 label')
            print(df_new_labels.loc[df_new_labels['path'].duplicated(), 'path'].to_list())
            return False
        
        # If record of new labels should be overwritten
        if overwrite_new_labels:
            
            # Add labels to images that need labels and keep a record of it
            self.df_new_labels = pd.DataFrame(sr_paths_to_be_labelled).reset_index().merge(
                df_new_labels,
                how = 'left',
                on = 'path'
            ).set_index('index')
            
            print('Labels were loaded successfully')
        
        # File is valid if it wasn't found to be
        # invalid
        return True
        
    
    def request_labels(self, label_request):
        """
        Asks and waits for labels from the user
        """
        
        # Create Timelapse selection file
        self.submit_label_request(label_request)
        
        # Asks user to place labels in label bin
        print(dedent(
            """
            Please place exported Timelapse CSV in the labels bin at
            {}
            """.format(self.label_bin_dir)
        ))
        
        while True:
            
            # Wait for user to input labels
            input('Press enter when you have placed the labels in the labels bin')
            
            
            # If inputted labels are valid
            if self.labels_file_valid(label_request):
                
                # Unpack new labels (in labels_file_valid method)
                
                # Stop asking for labels
                break
            
        # Return unpacked labelled data
        return self.unpack_new_labels()
    
    def unpack_new_labels(self, df_new_labels = None):
        """
        Converts dataframe of new labels
        to a format that can be read by
        the rest of the program
        
        df_new_labels (Pandas dataframe): New labels.
            Must have column 'Species' that contains
            the labels for the data as strings and
            the index must be the image's index
            as it appears in the default pool. If None,
            will use the dataframe stored in the object
        """
        
        if df_new_labels is None:
            df_new_labels = self.df_new_labels
        
        
        return df_new_labels['Species'].to_dict()
        
    
    def submit_label_request(self, label_request):
        """
        Creates CSV file used to select data
        in Timelapse
        
        label_request (List of int): Indices
            to label as they appear in the default
            pool
        """
        
        # If label selector file already exist
        if os.path.exists(self.timelapse_selector_path):
            
            # Ensure that file can be edited
            file_editable = False
            
            while not file_editable:
            
                try:
                    
                    # Check that CSV file can be edited (Does not work for linux)
                    os.rename(self.timelapse_selector_path, self.timelapse_selector_path)
                    
                    # If it can be edited, record that it can be edited
                    file_editable = True
                    
                except:
                
                    input(dedent(
                        """
                            CSV for selecting timelapse data could not be edited.
                            Please close the application that is using the file
                            and then press Enter.
                            
                            The file is located at
                            {}
                        """.format(self.timelapse_selector_path)
                    ))
        
        # Reset image selector flags
        self.df_lr_template['Selected'] = False
        
        # Build dataframe for highlighting images for labelling
        df_label_update = pd.DataFrame(
            data = {'Selected' : [True] * len(label_request)},
            index = label_request
        )
        df_label_update.index.name = 'index'
        
        # Flag images for labelling
        self.df_lr_template.update(df_label_update)
        
        # Submit request for labels
        self.df_lr_template.to_csv(
            self.timelapse_selector_path,
            index = False,
            columns = ['File', 'RelativePath', 'Selected']
        )
        
        # Send message telling user that Timelapse selector
        # Has been updated
        print(dedent(
            """
                Timelapse data selector file has been updated.
                Please import it into the Timelapse database.
            """
        ))