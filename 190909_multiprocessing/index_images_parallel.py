# USAGE
# python index_images_parallel.py --images ..\..\datasets\caltech101 --output temp_output --hashes hashes.pickle
from pyimagesearch.parallel_hashing import process_images, chunk
from multiprocessing import Pool, cpu_count
from imutils import paths
import numpy as np
import argparse
import pickle
import os

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--images', required=True, type=str, help='Path to input directory of images')
    ap.add_argument('-o', '--output', required=True, type=str, help='Path to output directory to store intermediate files')
    ap.add_argument('-a', '--hashes', required=True, type=str, help='Path to output hashes directory')
    ap.add_argument('-p', '--procs', type=int, default=-1, help='# of processes to spin up')
    args = vars(ap.parse_args())

    # determine the number of concurrent processes to launch when distributing the load across the system, then create the list of process IDs
    procs = args['procs'] if args['procs'] > 0 else cpu_count()
    proc_ids = list(range(procs))

    # grab the paths to the input images, then determine the number of images each process will handle
    print('[INFO] grabbing image paths...')
    all_image_paths = list(paths.list_images(args['images']))
    num_images_per_proc = int(np.ceil(len(all_image_paths) / float(procs)))

    # chunk the image paths into N equal sets, one set of image paths for each individual process
    chunked_paths = list(chunk(all_image_paths, num_images_per_proc))

    # initialize the list of payloads
    payloads = []

    # loop over the set chunked image paths
    for i, image_paths in enumerate(chunked_paths):
        # construct the path to the output intermediary file for the current process
        output_path = os.path.sep.join([args['output'], f'proc_{i}.pickle'])

        # construct a dictionary of data for the payload, then add it to the payloads list
        data = {
            'id': i,
            'input_paths': image_paths,
            'output_path': output_path
        }
        payloads.append(data)

    # construct and launch the processing pool
    print(f'[INFO] launching pool using {procs} processes...')
    pool = Pool(processes=procs)
    pool.map(process_images, payloads)

    # close the pool and wait for all processes to finish
    print('[INFO] waiting for processes to finish...')
    pool.close()
    pool.join()
    print('[INFO] multiprocessing complete')

    # initialize our combined hashes dictionary
    print('[INFO] combining hashes...')
    hashes = {}

    for path in paths.list_files(args['output'], validExts=('.pickle')):
        # load the contents of the dictionary
        data = pickle.loads(open(path, 'rb').read())

        # loop over the hashes and image paths in the dictionary
        for temp_h, temp_paths in data.items():
            # grab all image paths with current hash, add in the image paths for the current pickle file, and then update our hashes dictionary
            image_paths = hashes.get(temp_h, [])
            image_paths.extend(temp_paths)
            hashes[temp_h] = image_paths

    # serialize the hashes dictionary to disk
    print('[INFO] serializing hashes...')
    f = open(args['hashes'], 'wb')
    f.write(pickle.dumps(hashes))
    f.close()
