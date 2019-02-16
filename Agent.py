# Controls Dino
class DinoAgent:
    # __init__(): starts the game by performing one jump
    def __init__(self, game):
        self._game = game
        self.jump()
        time.sleep(0.5)

    def is_running(self):
        return self._game.get_playing()

    def is_crashed(self):
        return self._game.get_crashed()

    def jump(self):
        self._game.press_up()

    def duck(self):
        self._game.press_down()