# USAGE
# python index_images.py --images ..\..\datasets\caltech101 --tree vptree.pickle --hashes hashes.pickle
from pyimagesearch.parallel_hashing import *
from imutils import paths
import argparse
import pickle
import vptree
import cv2

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True, type=str, help='Path to input directory of images')
ap.add_argument('-t', '--tree', required=True, type=str, help='Path to output VP-tree')
ap.add_argument('-a', '--hashes', required=True, type=str, help='Path to output hashes directory')
args = vars(ap.parse_args())

# Grab the paths to the input images and initialize the hashes
image_paths = list(paths.list_images(args['images']))
hashes = {}

# loop over the image paths
for i, image_path in enumerate(image_paths):
    # load the input image
    if (i + 1) % 100 == 0:
        print(f'[INFO] processing images {i + 1}/{len(image_paths)}...')
    image = cv2.imread(image_path)

    # compute the hash for the image and convert it
    h = dhash(image)
    h = convert_hash(h)

    # update the hashes dictionary
    l = hashes.get(h, [])
    l.append(image_path)
    hashes[h] = l

# build the VP-tree
print('[INFO] building VP-Tree...')
points = list(hashes.keys())
tree = vptree.VPTree(points, hamming)

# serialize the Vp-Tree to disk
print('[INFO] serializing VP-Tree...')
f = open(args['tree'], 'wb')
f.write(pickle.dumps(tree))
f.close()

# serialize the hashes to dictionary
print('[INFO] serialzing hashes...')
f = open(args['hashes'], 'wb')
f.write(pickle.dumps(hashes))
f.close()