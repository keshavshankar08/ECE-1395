from kmeans_multiple import *
from skimage.io import imread, imsave
from skimage.transform import resize
import numpy as np

def Segment_kmeans(im_in, K, iters, R):
    # downsample the image to avoid long runtime and convert to double format
    image_resized = resize(im_in, (100, 100), anti_aliasing=True) # imresize is depricated so used this
    image_double = image_resized.astype(np.float64) / 255.0
    
    # reshape to 2d of dim numPixels x 3 (for color image)
    X = np.reshape(image_resized, (image_double.shape[0] * image_double.shape[1], 3))
    
    # run kmeans on image
    ids, means, ssd = kmneans_multiple(X, K, iters, R)

    # recolor image based on cluster belongingness
    image_recolored = np.zeros_like(X)
    for k in range(K):
        image_recolored[ids == k] = means[k]
    
    # reshape to original size (100 x 100 downsampled)
    image_recolored = np.reshape(image_recolored, (image_double.shape[0], image_double.shape[1], 3))
    
    # convert image to uint8
    im_out = (image_recolored * 255).astype(np.uint8)
    
    return im_out