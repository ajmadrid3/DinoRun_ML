# Used by the network to perform actions and getting new states
class Game_state:
    def __init__(self, agent, game):
        self._agent = agent
        self._game = game

    # get_state(): Accepts an array of actions and performs the action on the Dino
    #              Returns the new state, the reward, and if the game ended
    def get_state(self, actions):
        score = self._game.get_score()
        reward = 0.1 * score / 10
        is_over = False
        if actions[1] == 1:
            self._agent.jump()
            reward = 0.1 * score / 11
        image = grab_screen()

        if self._agent.is_crashed():
            self._game.restart()
            reward = -11 / score
            is_over = True

        return image, reward, is_over