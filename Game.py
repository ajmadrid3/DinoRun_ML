# Selenium interfacing between Python and Chrome
import os
import time
from Path import game_url, chrome_driver_path
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys

class Game:
    # __init__(): Launch Chrome using attributes in chrome_options
    def __init__(self, custom_config = True):
        chrome_options = Options()
        chrome_options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path = chrome_driver_path, chrome_options = chrome_options)
        self._driver.set_window_position(x = -10, y = 0)
        self._driver.set_window_size(200, 300)
        self._driver.get(os.path.abspath(game_url))

        if custom_config:
            self._driver.execute_script("Runner.congif.ACCELERATION=0")

    # get_crashed(): Returns true if Dino hit an obstacle
    def get_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed")

    # get_playing(): Returns true if the game is running, false if it crashed or paused
    def get_playing(self):
        return self._driver.execute_script("return Runner.instance_.playing")

    # restart(): Sends a signal to restart the game
    def restart(self):
        self._driver.execute_script("Runner.instance_.restart()")
        time.sleep(0.25)

    # press_up(): Sends a signal to press the up arrow key
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)

    # get_score(): Gets the current score
    def get_score(self):
        score_array = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        score = ''.join(score_array)
        return int(score)

    # pause(): Sends a signal to pause the game
    def pause(self):
        return self._driver.execute_script("return Runner.instance_.stop()")

    # resume(): Sends a signal to unpause the game
    def resume(self):
        return self._driver.execute_script("return Runner.instance_.play()")

    # end(): Sends a signal to close the browser and end the game
    def end(self):
        self._driver.close()