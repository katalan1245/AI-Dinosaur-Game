import numpy as np
from mss import mss
import cv2
import keyboard
import os
from time import sleep
import shutil

def preprocessing(img):
    dino_width = 45
    img = img[:,dino_width:]
    cv2.Canny(img,threshold1 = 100, threshold2 = 200)
    return img

def start():
    T = 0
    
    sct = mss()
    
    coordinates = {
        "left": 650,
        "top": 320,
        "width": 165,
        "height": 54
    }
        
    
    if os.path.exists('./images'):
            shutil.rmtree('./images')
    os.mkdir(r'./images')
    
    with open('./actions.csv', 'w') as csv:
        
        frame = 0
        
        while True:
            img = preprocessing(np.array(sct.grab(coordinates)))
            if keyboard.is_pressed('up arrow'):
                cv2.imwrite('./images/frame_{0}.jpg'.format(frame), img)
                csv.write('1\n')
                print('jump write')
                frame += 1
                sleep(T)
            
            if keyboard.is_pressed('down arrow'):
                cv2.imwrite('./images/frame_{0}.jpg'.format(frame), img)
                csv.write('2\n')
                print('duck')
                frame += 1
                sleep(T)
                
            if keyboard.is_pressed('n'): # do nothing
                cv2.imwrite('./images/frame_{0}.jpg'.format(frame), img)
                csv.write('0\n')
                print('nothing')
                frame += 1
                sleep(T)
                
            # break the video feed
            if keyboard.is_pressed('esc'):
                csv.close()
                cv2.destroyAllWindows()
                return -1