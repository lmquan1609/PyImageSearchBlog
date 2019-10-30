import numpy as np
import pickle
import cv2
import vptree

def dhash(image, hash_size=8):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # resize the input image, add width so we can compute the horizontal gradient
    resized = cv2.resize(gray, (hash_size + 1, hash_size))

    # compute the horizontal gradient between adjacent columns pixels
    diff = resized[:, 1:] > resized[:, :-1]

    # convert the differene image to a hash
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def convert_hash(h):
    return int(np.array(h, dtype='float64'))

def hamming(a, b):
    return bin(int(a) ^ int(b)).count('1')

def chunk(l, n):
    # loop over the list in n-sized chunks
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process_images(payload):
    # display the process ID for debugging and initialize the hashes dictionary
    print(f"[INFO] starting process {payload['id']}...")
    hashes = {}

    # loop over the image paths
    for image_path in payload['input_paths']:
        # load the input image
        image = cv2.imread(image_path)

        # compute the hash for the image and convert it
        h = dhash(image)
        h = convert_hash(h)

        # update the hashes dictionary
        l = hashes.get(h, [])
        l.append(image_path)
        hashes[h] = l

    # serialize the hashes dictionary to disk using the supplied output path
    print(f"[INFO] process {payload['id']} serializing hashes...")
    f = open(payload['output_path'], 'wb')
    f.write(pickle.dumps(hashes))
    f.close()