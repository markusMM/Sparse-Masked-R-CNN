"""
Mask R-CNN
Configurations and data loading code for the Maritime dataset
This is an sub-class created using the original class.
Written by Adrian Llopart
"""

from config import Config
import skimage.io
from scripts.ex_train_maritime import main
import os
import glob
import numpy as np
from sparse_mask_rcnn import utils

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# Maritime Configuration Class
# It overwrites the original values from the original class
class MaritimeConfig(Config):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = "Maritime"  # Override in sub-classes

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 2

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch.
    # A bigger number improves accuracy of validation stats, but slows
    # down the training.
    VALIDATION_STEPS = 1

    # Number of classification classes (including background)
    NUM_CLASSES = 6  # For background + my_classes

    # Input image resing
    # Images are resized such that the smallest side is >= IMAGE_MIN_DIM and
    # the longest side is <= IMAGE_MAX_DIM. In case both conditions can't
    # be satisfied together the IMAGE_MAX_DIM is enforced.
    IMAGE_MIN_DIM = 448
    IMAGE_MAX_DIM = 512

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 50

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 50


    # Image mean (RGB)
    MEAN_PIXEL = np.array([152.5409961171337, 151.54973003671023, 150.04029676639962])

    # TODO think it over
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9


class MaritimeDataset(utils.data.Dataset):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:

    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...

    See COCODataset and ShapesDataset as examples.
    """
    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def load_maritime(self, dataset_dir, dataset_type):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("maritime", 1, "buoy")
        self.add_class("maritime", 2, "land")
        self.add_class("maritime", 3, "sea")
        self.add_class("maritime", 4, "ship")
        self.add_class("maritime", 5, "sky")


        # Get folders
        dataset_dir = os.path.join(dataset_dir, dataset_type)
        examples_paths = sorted([os.path.join(dataset_dir,f) for f in os.listdir(dataset_dir)])
        number_of_examples = len(examples_paths)

        # Add images
        for example, example_path in enumerate(examples_paths):
            image_path = os.path.join(example_path,'rgb.jpg')
            self.add_image("maritime", image_id=example, path=image_path)

    def load_image(self, image_id):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])

        #print(self.image_info[image_id]['path'])
        
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.

        instance_masks = []
        class_ids = [] 

        labels_path = os.path.dirname(self.image_info[image_id]['path'])
        labels_path = os.path.join(labels_path, 'labels')

        #Get all .png files in the folder
        file_paths = os.path.join(labels_path,'*.png')
        file_paths = sorted(glob.glob(file_paths))

        #Add mask to instance_masks and append the class name found in the filename
        for file_path in file_paths:
            for cat in self.class_names:
                if cat in file_path:
                    mask = skimage.io.imread(file_path)
                    instance_masks.append(mask)
                    class_ids.append(self.class_names.index(cat))
                    #print("Filename loaded: ", file_path)
                    #print("Class loaded: ", cat)

        #Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(MaritimeDataset, self).load_mask(image_id)
