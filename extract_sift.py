# from a single image, extract SIFT features from it
# by Carmen Carrano 6/16/2014 LLNL
import cv2
import numpy as np
import math
import os
from matplotlib import pyplot as plt


# Uses the grid adapted feature detection to limit the Number of detected keypoints
def get_sift(img, show=1, maxfeatures=500, xgrid=1, ygrid=1):
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift_det = cv2.FeatureDetector_create("SIFT")
    sift_det_grid = cv2.GridAdaptedFeatureDetector(sift_det, maxfeatures, xgrid, ygrid)
    sift_desc = cv2.DescriptorExtractor_create("SIFT")
    fs = sift_det_grid.detect(img2)

    (keypoints, descriptors) = sift_desc.compute(img2, fs)
    
    if show == 1:
        img_wSIFT = img.copy()
        d_blue = cv2.cv.RGB(25, 15, 100)
        l_blue = cv2.cv.RGB(200, 200, 250)
        for f in fs:
            cv2.circle(img_wSIFT, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), l_blue, 2, cv2.CV_AA)
            cv2.circle(img_wSIFT, (int(f.pt[0]), int(f.pt[1])), int(f.size/2), d_blue, 1, cv2.CV_AA)
            ori = math.radians(f.angle) 
            tx = math.cos(ori) * 0 - math.sin(ori) * (f.size/2)
            ty = math.sin(ori) * 0 + math.cos(ori) * (f.size/2)
            tx += f.pt[0]
            ty += f.pt[1]
            cv2.line(img_wSIFT, (int(f.pt[0]), int(f.pt[1])), (int(tx), int(ty)), l_blue, 2, cv2.CV_AA)
            cv2.line(img_wSIFT, (int(f.pt[0]), int(f.pt[1])), (int(tx), int(ty)), d_blue, 1, cv2.CV_AA)

        plt.imshow(img_wSIFT[:, :, ::-1])
        plt.show()

    return keypoints, descriptors


def extract_sift(im_name):
    im = cv2.imread(im_name)
    out_name = im_name.split('.')[0]
    out = out_name + '.SIFT'
    feat_file = open(out, 'w')
    kp, desc = get_sift(im, show=0, maxfeatures=500)  # set to whatever you want to have as max SIFt points
    # These silly joins convert the numpy 2d array into a string with space delimiting.
    # Name, imgrows, imgcols, number of keypoints, list of ( keypoint location (x,y), keypoint size,
    # keypoint strength(float), keypoint angle,keypoint octave, descriptor (128 floats per keypoint) )
    out_str = im_name + ',' + str(np.size(im, 0)) + ',' + str(np.size(im, 1)) + ',' + str(np.size(kp))
    for i in range(np.size(kp)):
        ktmp = kp[i]
        dtmp = desc[i]
        out_str += ',' + str(int(round(ktmp.pt[0]))) + ',' + str(int(round(ktmp.pt[1]))) + ',' + str(ktmp.size) + ',' + str(ktmp.response) + ',' + str(int(ktmp.angle)) + ',' + str(int(ktmp.octave)) + ','
        out_str += ' '.join(map(str, np.ndarray.tolist(dtmp.flatten())))

    out_str += '\n'
    feat_file.write(out_str)
    feat_file.close()
    return kp, desc


def file_to_sift(filename):
    kp = []
    desc = []
    with open(filename) as f:
        feat_str = f.read().strip()
        cols = feat_str.split(',')

        im_name = cols[0]
        im_size = map(int, (cols[1], cols[2]))
        kp_size = int(cols[3])
        for i in range(kp_size):
            base_idx = 4 + i*7
            kp_desc = map(float, cols[base_idx:base_idx+6])
            x = int(kp_desc[0])
            y = int(kp_desc[1])
            kp_size = kp_desc[2]
            kp_response = kp_desc[3]
            kp_angle = int(kp_desc[4])
            kp_octave = int(kp_desc[5])
            
            ktmp = cv2.KeyPoint(x=x, y=y, _size=kp_size, _angle=kp_angle, _response=kp_response, _octave=kp_octave)
            kp.append(ktmp)

            dtmp = cols[base_idx+6]
            dtmp = map(float, dtmp.split())

            desc.append(dtmp)

    return kp, np.array(desc, dtype=np.float32)

def file_to_sift(filename, dir, image_hash):
    kp = []
    desc = []
    with open(dir + '/' + filename) as f:
        for line in f:
            feat_str = line.strip()
            cols = feat_str.split(',')

            im_name = cols[0]
            if im_name != image_hash:
                continue

            kp_size = int(cols[3])
            for i in range(kp_size):
                base_idx = 4 + i*7
                kp_desc = map(float, cols[base_idx:base_idx+6])
                x = int(kp_desc[0])
                y = int(kp_desc[1])
                kp_size = kp_desc[2]
                kp_response = kp_desc[3]
                kp_angle = int(kp_desc[4])
                kp_octave = int(kp_desc[5])

                ktmp = cv2.KeyPoint(x=x, y=y, _size=kp_size, _angle=kp_angle, _response=kp_response, _octave=kp_octave)
                kp.append(ktmp)

                dtmp = cols[base_idx+6]
                dtmp = map(float, dtmp.split())

                desc.append(dtmp)

    return kp, np.array(desc)

extract_sift('dog.jpg')  # Replace filename here to test the script with some image