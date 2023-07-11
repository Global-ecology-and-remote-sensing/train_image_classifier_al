# Train Image Classifier Using Active Learning

This package provides scripts for training an image classifier
using active learning (AL). It uses the approach described in the
paper called "A deep active learning system for species 
identification and counting in camera trap images" by Norouzaddeh
et al., which tries to improve the efficieny of the training
algorithm by performing active learning on the features extracted
from the data rather than the images themselves (see 
[acknowledgements](#acknowledgements) for more details). 

The
package is an adaptation of the scripts that they used in their
experiment. It differs from its source code as it was was designed 
to perform active learning in a practical setting. As such, it
provides code to load-in unlabelled data and a way to label data
using Timelapse. 

Included in this package is a function called
"run_active_learning". It trains a model using the approach in
the paper by Norouzaddeh et al. with some additional features
such as providing checkpoints, testing the model if a test set
has been provided and saving the model in an interoperable format.
If you wish to use your own custom model architecture and training
algorithm then this package could still prove useful by providing
objects that can load-in unlabelled data, label data and keeping
track of what images belong to which datasets (more information
can be found in 
[Designing your own model](#designing-your-own-model)


**Comment** Why use active learning, active learning's slow computation on
raw images, package uses Norouzaddeh's approach in their paper, it was
adapted from the code that they used in their paper so that it can be
used outside of an experiment. It provides a main script "run_active_learning"
for running active learning but it also provides the objects, active_learning_environment,
data_set and engine which users can use for building their own image processing model.
It also uses both timelapse and AL algorithms designed by Google.

## Why Use Active Learning?

Training neural networks to classify images typically require very
large databases of manually labelled images. In some contexts,
labelling these images can be far more expensive than the
computational resources that are required to train the model. One
way to reduce this cost is to label the images as the model is
being trained. The idea behind this is to choose the best images
to label at each iteration of the training algorithm as to provide
the model with the most information at each stage. Algorithms that
employ this technique are often classed under "active learning".

## Quick Start

**Comment** What you need to run the program, the code that you need to run, 
how to prepare the data, how to label
things with timelapse. Using logging package to see info messages

### Labelling with Timelapse

## Training Algorithm

**Comment** Parameters of main script (No elaboration as that comes later), choices for training algorithm,
I think that when I elaborate, I should use pandas' example of have the parameter and then talk
about it?

### Data

**Comment** Format of data folders, train set needs all labels, unlabelled needs to be
in subfolders, validation set of classes can be a subset of the training ones

### Active learning

**Comment** AL batch size, available sampling methods

### Embedding model

**Comment** Available architectures, triplet loss vs softmax, 
(I don't know what the triplet loss hyperparameters do)

### Train and finetune embedding

**Comment** Adam sampler, available hyperparameters, importance of num_epochs, balanced vs simple loader

### Classifier

**Comment** No way to edit classifier as it's hardcoded into the source code :(

## Working Folder

**Comment** This is where all files generated by the package are
saved

### Checkpoint folders

**Comment** Needed for checkpoints, don't change or move these
folders if you want to load from checkpoint

### Label bin

**Comment** Covered by previous section but described here for completion

### Validation results

**Comment** Stored in test results folder, to save model for
latest test results, you'll have to copy the export
folder before you submit the labels for that AL batch,
Does not exist if validation doesn't exist

### Exporting model

**Comment** Trained model for latest test results

## Designing your own model

**Comment** Brief comment on which classes might be useful
and that you can use run_active_learning.py as an example

## Cite

**Comment** How to cite the package

## Acknowledgments

**Comment** Acknwoledgments to Megadetector, Norouzaddeh,
Google team for AL algorithms