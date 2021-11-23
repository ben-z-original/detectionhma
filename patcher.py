import numpy as np


def split_in_chunks(img, size=448, pad=0):
    """ Splits the images into patcher (resp. tiles). """
    h, w, c = img.shape

    # number of required patches
    steps_h = h // (size + 1) + 1
    steps_w = w // (size + 1) + 1

    # create padding (top, bottom, left, right)
    img_tmp = np.pad(img, ((pad, size - h % size + pad), (pad, size - w % size + pad), (0, 0)), mode='reflect')

    # container for patches
    patches = []

    # loop over all patches
    for i in range(steps_h):
        for j in range(steps_w):
            # obtain patch with padding and append to results
            patch = img_tmp[i * size:(i + 1) * size + 2 * pad, j * size:(j + 1) * size + 2 * pad, :]
            patches.append(patch)

    return patches


def merge_from_chunks(patches, targ_h, targ_w, targ_c=1, size=448, pad=0):
    """ Merges the patches (resp. tiles) into one image"""
    # determine temporary size
    steps_h = targ_h // (size + 1) + 1
    steps_w = targ_w // (size + 1) + 1

    # create dummy for result
    img_merged = np.zeros((steps_h * size, steps_w * size, targ_c))

    # loop over all patches
    for i in range(steps_h):
        for j in range(steps_w):
            # get patch and remove padding
            patch = patches[i * steps_w + j].reshape(size+2*pad, size+2*pad, -1)
            patch = patch[pad:size+pad, pad:size+pad, :]

            # paste patch
            img_merged[i * size:(i + 1) * size, j * size:(j + 1) * size, :] = patch

    # remove irrelevant regions
    img_merged = img_merged[0:targ_h, 0:targ_w, :]

    # remove dimension
    if targ_c == 1:
        img_merged = img_merged[...,0]

    return img_merged
