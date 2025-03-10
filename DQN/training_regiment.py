


class progressive:
    def __init__(self):
        """ Class designed to train an agent from the lowest difficulty
        to different and variying higher difficult situations.
        In other words, we want to train the agent on a 2x2 grid first, then 
        keep increasing the size of the grid till a sufficient size we want.
        
        While it trains, we get to a point where it does well, and then increase the 
        size, while trying to make sure it does not overfit to that specific size."""