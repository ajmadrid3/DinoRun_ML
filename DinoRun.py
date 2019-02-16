from Game import Game
from Agent import DinoAgent
from Game_state import Game_state
from buildModel import buildmodel
from trainNetwork import trainNetwork

def playGame(observer = False):
    game = Game()
    dino = DinoAgent(game)
    game_state = Game_state(dino, game)
    model = buildmodel()
    trainNetwork(model, game_state)