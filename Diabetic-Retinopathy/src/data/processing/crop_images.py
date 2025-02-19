import os
from os.path import join
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageFilter
from PIL import ImageDraw, ImageChops
from multiprocessing import Process
import cv2

#Change to local paths, where you have downloaded the data
DATA_ROOT_DIR = "/scratchk/ehealth/optha/"
APTOS_ROOT_DIR = join(DATA_ROOT_DIR, "aptos")
EYEPACS_ROOT_DIR = join(DATA_ROOT_DIR, "eyepacs")
ODIR_ROOT_DIR = join(DATA_ROOT_DIR, "odir")
ANNOTATED_ROOT_DIR = join(DATA_ROOT_DIR, "annotated")
AIIMS_DELHI_ROOT_DIR = join(DATA_ROOT_DIR, "aiims_delhi")

src = join(AIIMS_DELHI_ROOT_DIR, 'images/ophtha_images')
tgt = join(AIIMS_DELHI_ROOT_DIR, 'data/processed_images_cv2_test')

def main():
    jobs = []
    for root, _, imgs in os.walk(src):
        for img in tqdm(imgs):
            src_path = os.path.join(root, img)
            tgt_dir = root.replace(src, tgt)
            if not os.path.exists(tgt_dir):
                os.makedirs(tgt_dir)
            tgt_path = os.path.join(tgt_dir, img)
            jobs.append((src_path, tgt_path, 1024))

    procs = []
    job_size = len(jobs) // 8
    for i in range(8):
        if i < 7:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:(i + 1) * job_size])))
        else:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:])))

    for p in procs:
        p.start()

    for p in procs:
        p.join()


def crop_image_from_gray(img,tol=7):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
    #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1,img2,img3],axis=-1)
    #         print(img.shape)
        return img


def circle_crop(img, sigmaX=10):   
    """
    Create circular crop around image centre    
    """    
    img = cv2.imread(img)
    img = crop_image_from_gray(img)    
    img = img[:, :, ::-1]
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    # img=cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def convert_list(i, jobs):
    for j, job in enumerate(jobs):
        if j % 100 == 0:
            print('worker{} has finished {}.'.format(i, j))
        convert(*job)


def convert(fname, tgt_path, crop_size):
    im = (Image.fromarray(circle_crop(fname))).resize([crop_size, crop_size], Image.ANTIALIAS).convert('RGB')

    save(im, tgt_path)


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def convert_square(fname, crop_size):
    img = Image.open(fname)
    bbox = square_bbox(img)
    cropped = img.crop(bbox)
    # resized = img.resize([crop_size, crop_size])
    return cropped


def get_convert_fname(fname, extension, directory, convert_directory):
    return fname.replace('JPG', extension).replace(directory,
                                                   convert_directory)


def process(args):
    fun, arg = args
    directory, convert_directory, fname, crop_size, extension = arg
    convert_fname = get_convert_fname(fname, extension, directory,
                                      convert_directory)
    if not os.path.exists(convert_fname):
        img = fun(fname, crop_size)
        save(img, convert_fname)


def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
