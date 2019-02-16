import numpy as np
import cv2
from PIL import ImageGrab


def grab_screen(_Driver = None):
    screen = np.array(ImageGrab.grab(bbox = (40,180,440,400)))    
    image = procces_img(screen)
    return image

def procces_img(image):
    image = cv2.resize(image, (0,0), fx = 0.15, fy = 0.10)
    #Crops out dinosaur from image
    image = image[2:38,10:50]
    image = cv2.Canny(image, threshold1 = 100, threshold2 = 200)
    return image