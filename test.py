import numpy as np
import matplotlib.image as mpimg
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from imgaug import augmenters as iaa


dude = '/home/cyrano5614/Documents/webcam/dude.jpg'
hand = '/home/cyrano5614/Documents/webcam/hands.jpg'


def preprocess(image):

    img = mpimg.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64))

    return img

def augmentation(image):

    img = image[..., np.newaxis]
    seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), # crop images from each side by 0 to 16px (randomly chosen)
    # iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0)) # blur images with a sigma of 0 to 3.0
    ])

    images_aug = np.empty([1000, 64, 64, 1])

    for i in range(1000):
        image = img
        output = seq.augment_images([image])
        output = np.array(output)
        images_aug[i] = output

    return images_aug


def augment_hog(img):

    images = augmentation(img)
    hog_output = np.empty([1000, 2916])

    for i, image in enumerate(images):
        img_hog = np.squeeze(image, axis=(2,))
        img_hog = hog_features(img_hog)
        hog_output[i] = img_hog


def hog_features(img):
    hog_array = hog(img)

    return hog_array


def plot_images(images):

    plt.figure(figsize=(20,10))

    for i in range(12):
        img_draw = np.squeeze(images[i], axis=(2,))
        plt.subplot(3, 4, i+1)
        plt.imshow(img_draw)

    plt.show()


def visualize(img_path):
    plot_images(augmentation(preprocess(img_path)))

# visualize('/home/cyrano5614/Documents/webcam/dude.jpg')
