# Global variables of programs in main_scripts.py
from camera_trap_al.deep_learning.networks import models

"""
------------------------------------------------------------------------------
            Hyperparameters for Training and Finetuning Model
------------------------------------------------------------------------------
"""
# Options for training strategies
LOSS_TYPES = ['softmax', 'triplet', 'siamese']
AL_STRATEGY_TYPES = ['uniform', 'graph_density', 'entropy', 'confidence',
     'kcenter', 'margin', 'informative_diverse', 'margin_cluster_mean', 'hierarchical']
EMBEDDING_LOSS_DATA_STRATEGIES = ['hardest', 'random', 'semi_hard', 'hard_pair']

"""
------------------------------------------------------------------------------
                Hyperparameters for Model Architecture
------------------------------------------------------------------------------
"""
# Options of architectures for base model
ARCHITECTURES = sorted(name for name in models.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(models.__dict__[name]))


# Dataset types
DATASET_TYPES = ['val', 'train', 'unlabelled']

# Active learning pool types
AL_POOL_TYPES = ['active', 'default', 'val']
