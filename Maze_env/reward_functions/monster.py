from Maze_env.reward_functions.basic import BasicRewardFun


class MonsterRewards(BasicRewardFun):
    def __init__(self,**kwargs):

        super().__init__(**kwargs)

    def __setup__(self):
        rewards = {
            'CAPTURE': 0.0,
            'GET_CLOSER_CONSTANT' : 0.0,
            'GET_CLOSER' : 0.0
        }
        return rewards | super().__setup__()