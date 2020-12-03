from keras.models import load_model
from mss import mss
import cv2
import numpy as np
import time

model = load_model('dino_ai_weights_post_train.h5')

start = time.time()
def predict(game):
    
    # configuration for image capture
    sct = mss()
    
    dino_width = 45
    
    coordinates = {
        "left": 650,
        "top": 300,
        "width": 165,
        "height": 54
    }
    
    # image capture
    img = np.array(sct.grab(coordinates))
    
    # cropping, edge detection, resizing to fit excpected model input
    img = img[:,dino_width:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.Canny(img, threshold1 = 100, threshold2 = 200)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    img = img[np.newaxis, :, :, np.newaxis]
    # model prediction
    y_prob = model.predict(img)
    prediction = y_prob.argmax(axis = -1)
    
    print(prediction)
    if prediction == 1:
        # jump
        game.send_keys(u'\ue013')
        print('YEEP YEEP')
        time.sleep(0) #0.2
        """if prediction == 2:
        # duck
        game.send_keys(u'\ue015')
        print("DUCKING") """
    if prediction == 0:
        # do nothing
        print("I'M CHILLING")
        pass
