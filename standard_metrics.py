
import numpy as np
import torch
from skimage import feature
from scipy import ndimage


def evaluateDepthStandard(predDepths, gtDepths, masks=None, printInfo=False):
    """Evaluate depth reconstruction accuracy"""

    if isinstance(masks, type(None)):
        masks = gtDepths > 1e-4

    numPixels = float(masks.sum())

    rmse = np.sqrt((pow(predDepths - gtDepths, 2) * masks).sum() / numPixels)
    rmse_log = np.sqrt(
        (pow(np.log(np.maximum(predDepths, 1e-4)) - np.log(np.maximum(gtDepths, 1e-4)), 2) * masks).sum() / numPixels)
    log10 = (np.abs(
        np.log10(np.maximum(predDepths, 1e-4)) - np.log10(np.maximum(gtDepths, 1e-4))) * masks).sum() / numPixels
    rel = (np.abs(predDepths - gtDepths) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels
    rel_sqr = (pow(predDepths - gtDepths, 2) / np.maximum(gtDepths, 1e-4) * masks).sum() / numPixels

    deltas = np.maximum(predDepths / np.maximum(gtDepths, 1e-4), gtDepths / np.maximum(predDepths, 1e-4)) + (
                1 - masks.astype(np.float32)) * 10000
    accuracy_1 = (deltas < 1.25).sum() / numPixels
    accuracy_2 = (deltas < pow(1.25, 2)).sum() / numPixels
    accuracy_3 = (deltas < pow(1.25, 3)).sum() / numPixels
    if printInfo:
        print(('depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))
        pass

    if numPixels == 0:
        print(('Zero mask depth statistics', rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3))

    return [rel, rel_sqr, log10, rmse, rmse_log, accuracy_1, accuracy_2, accuracy_3, numPixels]



def calculateTrueErrors(errors, pixels):
    ''' This method calculates real dataset statistics instead of per image. 
        Normally, they should be equal, yet, depth maps from sensors often have
        sensor noise, so incomplete/0/NaN depth values. This function considers
        this sensor noise by calculating only on valid pixels.

    '''

    total_pixels = np.sum(pixels)
    rel, relsqr, log10, rmse, rmselog, a1, a2, a3 = 0, 0, 0, 0, 0, 0, 0, 0
    for i in range(len(pixels)):
        no_pixels = pixels[i]
        if no_pixels == 0:
            continue
        rel += errors[i][0] * no_pixels / total_pixels
        relsqr += errors[i][1] * no_pixels / total_pixels
        log10 += errors[i][2] * no_pixels / total_pixels
        rmse += np.power(errors[i][3], 2) * no_pixels / total_pixels
        rmselog += np.power(errors[i][4], 2) * no_pixels / total_pixels
        a1 += errors[i][5] * no_pixels / total_pixels
        a2 += errors[i][6] * no_pixels / total_pixels
        a3 += errors[i][7] * no_pixels / total_pixels
        # print("no pixels, rel, rmse: ", no_pixels, rel, rmse)

    rmse = np.sqrt(rmse)
    rmselog = np.sqrt(rmselog)

    print("Total pixels: ", total_pixels)
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(rel, relsqr, log10,
                                                                                                rmse,
                                                                                                rmselog, a1, a2, a3))




