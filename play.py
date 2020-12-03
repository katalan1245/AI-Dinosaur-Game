from selenium import webdriver
import time
from capture import capture_feed
import os
import keyboard
import sys
# Check for existing model

is_trained = os.path.exists('./dino_ai_weights_post_train.h5')
if is_trained:
    ans = input('You have a trained model.\nWould you like to capture new data? [y/n]: ')
    if ans.lower() in ['yes', 'ye', 'y']:
        is_trained = False

# Setup the game

print('Firing Browser!')
options = webdriver.ChromeOptions()
options.add_argument('--start-maximized')
driver = webdriver.Chrome(options = options)
driver.get("https://chromedino.com/")
time.sleep(2)

# Main page to send key commands to
page = driver.find_element_by_class_name('offline')

# Start game
page.send_keys(u'\ue00d')

while True:
    if is_trained:
        from network import ai_player
        # controls the dinosaur
        ai_player.predict(page)
        if keyboard.is_pressed('esc'):
            print('Good Game!')
            sys.exit(0)
    else:
        # capture the game data
        print('Capturing Data')
        print('Press Escape to stop')
        if capture_feed.start() == -1:
            break
print('Data captured')