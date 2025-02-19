import os
from os.path import join
import numpy as np
import cv2

# cv2.setNumThreads(1)

def crop_image_from_gray(img,tol=15):
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
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def padded_crop(img, size=512):
    img = crop_image_from_gray(img)    
    img = resize_with_pad(img, size)
    return img[..., ::-1]

def resize_with_pad(image, size, padding_color = (0, 0, 0)):
    original_shape = (image.shape[1], image.shape[0])
    new_shape = (max(image.shape[1], image.shape[0]), max(image.shape[1], image.shape[0]))

    delta_w = new_shape[0] - original_shape[0]
    delta_h = new_shape[1] - original_shape[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,None,value=padding_color)
    return cv2.resize(image, (size,size), interpolation=cv2.INTER_AREA)