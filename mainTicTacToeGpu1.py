from Coach import Coach
# from othello.OthelloGame import OthelloGame as Game
# from othello.pytorch.NNet import NNetWrapper as nn
from tictactoe.TicTacToeGame import TicTacToeGame as Game
from tictactoe.TicTacToeGame import display as display
from tictactoe.keras.NNetGPU1 import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 100,
    'numEps': 50,
    'tempThreshold': 15,
    'updateThreshold': 0.55,
    'maxlenOfQueue': 400000,
    'numMCTSSims': 25,
    'arenaCompare': 100 ,
    'cpuct': 1,

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('temp','ttt6x6.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'reverseArena': True,

})

if __name__=="__main__":
    g = Game(6)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args, display)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
