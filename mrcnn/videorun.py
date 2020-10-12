"""
Mask R-CNN
Runs Mask R-CNN on trained vocalfold model.  Then calculates angle and produces a picture/video of TVF

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 vocalfold.py train --dataset=/path/to/vocalfold/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 vocalfold.py train --dataset=/path/to/vocalfold/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 vocalfold.py train --dataset=/path/to/vocalfold/dataset --weights=imagenet

    # Apply color splash to an image
    python3 vocalfold.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 vocalfold.py splash --weights=last --video=<URL or path to file>
"""

import json
import datetime
import numpy as np
import skimage.draw
import skimage.io

import os
import cv2
import sys
import random
import math
import re
import time
import skimage
import glob
import itertools
import logging
import json
from collections import OrderedDict
import numpy as np
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import pylab
from imgaug import augmenters as iaa
from mrcnn import visualize

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class VocalFoldConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "vocalfold"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 3

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + vocalfold

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class VocalFoldDataset(utils.Dataset):

    def load_vocalfold(self, dataset_dir, subset):
        """Load a subset of the VocalFold dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("vocalfold", 1, "vocalfold")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Load annotations
        # VGG Image Annotator (up to version 1.6) saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        # We mostly care about the x and y coordinates of each region
        # Note: In VIA 2.0, regions was changed from a dict to a list.
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. These are stores in the
            # shape_attributes (see json format above)
            # The if condition is needed to support VIA versions 1.x and 2.x.
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']] 

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "vocalfold",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a vocalfold dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "vocalfold":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "vocalfold":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = VocalFoldDataset()
    dataset_train.load_vocalfold(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = VocalFoldDataset()
    dataset_val.load_vocalfold(args.dataset, "val")
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Affine(rotate=(-60 , 60)),
    ])

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=150,
                augmentation=augmentation,
                layers='heads')



def display_stuff(mask):
  from google.colab.patches import cv2_imshow
  import imutils
  import pylab
  import cv2

  labels = [0,1]
  angles = []
  ellipse_angle = []
  chiral = []
 
  for i in labels:
    image = mask[:,:,i]
    test = pylab.axis('off') # no axis
    pylab.axes(frameon=False, xticks=[], yticks=[]) # Remove the white edge
    pylab.margins(0)
    pylab.tight_layout(pad=0)
    pylab.imshow(image)

    test_name = "test_" + "\[i]" + ".png"

    pylab.savefig('test.png', bbox_inches=None, pad_inches=0)

    im = cv2.imread('test.png')
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Find the contour that corresponds to the tvf
    cnt = min(contours, key=cv2.contourArea)

    #cnt = contours[1] - this was old version
    cv2.drawContours(im, cnt, -1, (0,255,0), 3)

    #Fit a line to the moment of inertia of the contour
    rows,cols = im.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)

    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(im,(cols-1,righty),(0,lefty),(0,255,0),2)

    # Attempt fitting an ellipse
    ellipse = cv2.fitEllipse(cnt)

    # Find the angle of rotation for the ellipse.  Re-adjust axes to make sure angles are calculated appropriately
    if ellipse[2] > 90:
      e_angle = ellipse[2] - 180
      ellipse_angle.append(e_angle)
      print(e_angle)
    else:
      ellipse_angle.append(ellipse[2])
      print(ellipse_angle)
        
    im = cv2.ellipse(im,ellipse,(255, 0,0),2)

    # Determine the most extreme points along the contour
    extLeft = tuple(cnt[cnt[:, :, 0].argmin()][0])
    #extRight = tuple(cnt[cnt[:, :, 0].argmax()][0])
    #extTop = tuple(cnt[cnt[:, :, 1].argmin()][0])
    #extBot = tuple(cnt[cnt[:, :, 1].argmax()][0])
    chiral.append(extLeft[0])

    # cv2.drawContours(image, [cnt], -1, (0, 255, 255), 2)
    #cv2.circle(im, extLeft, 3, (0, 0, 255), -1)
    #cv2.circle(im, extRight, 3, (0, 255, 0), -1)
    #cv2.circle(im, extTop, 3, (255, 0, 0), -1)
    #cv2.circle(im, extBot, 3, (255, 255, 0), -1)

    angle = int(math.atan((righty-lefty)/((cols-1)-0))*180/math.pi)
    
    # Re-adjust angle for weird axes system
    if angle > 0:
      angle = angle*-1

    angles.append(angle)
    

    cv2_imshow(im)  

  # Standard format for reporting (Left TVF then Right TVF)  
  if chiral[0] > chiral[1]:
    chiral.reverse()
    angles.reverse()
    ellipse_angle.reverse()

  return angles, ellipse_angle, chiral

def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    
    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax

def color_splash(image, r):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    #TODO: print prob and number of items
    prob = r['scores']
    number = r['class_ids']
    mask = r['masks']

    print(prob, number)

    #Calculates the angles between each TVF
    angles, ellipse_angle, chiral = display_stuff(mask)
    print(angles, ellipse_angle, chiral)

    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Color splash
        splash = color_splash(image, r)
        #Display image
        ax = get_ax(1)
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            dataset.class_names, r['scores'], ax=ax,
                            title="Predictions")
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
        vwriter2 = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r)
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                ax = get_ax(1)
                a = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                            'TVF', r['scores'], ax=ax)
                vwriter2.write(a)
                count += 1
        vwriter.release()
        vwriter2.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect vocalfolds.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/vocalfold/dataset/",
                        help='Directory of the VocalFold dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video,\
               "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = VocalFoldConfig()
    else:
        class InferenceConfig(VocalFoldConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
