# USAGE
# python search.py --tree vptree.pickle --hashes hashes.pickle --query queries\accordion.jpg
from pyimagesearch.parallel_hashing import *
import argparse
import pickle
import time
import cv2

# construct argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-t', '--tree', required=True, type=str, help='Path to output VP-tree')
ap.add_argument('-a', '--hashes', required=True, type=str, help='Path to output hashes directory')
ap.add_argument('-q', '--query', required=True, type=str, help='Path to input query image')
ap.add_argument('-d', '--distance', type=int, default=10, help='Maximum hamming distance')
args = vars(ap.parse_args())

# load the VP-Tree and hashes dictionary
print('[INFO] loading VP-Tree and hashes...')
tree = pickle.loads(open(args['tree'], 'rb').read())
hashes = pickle.loads(open(args['hashes'], 'rb').read())

# load the input query image
image = cv2.imread(args['query'])
cv2.imshow('Query', image)

# compute the hash for the query image, then convert it
query_hash = dhash(image)
query_hash = convert_hash(query_hash)

# perform the search
print('[INFO] performing search...')
start = time.time()
results = tree.get_all_in_range(query_hash, args['distance'])
results = sorted(results)
print(f'[INFO] search took {time.time() - start} seconds')

# loop over the results
for d, h in results:
    # grab all image paths in our dataset with the same hash
    result_paths = hashes.get(h, [])
    print(f'[INFO] {len(result_paths)} total image(s) with d: {d}, h: {h}')

    # loop over the result paths
    for result_path in result_paths:
        # load the result image and display it to our screen
        result = cv2.imread(result_path)
        cv2.imshow('Result', result)
        cv2.waitKey(0)