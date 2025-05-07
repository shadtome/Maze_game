from Maze_env.wrappers.reward_wrappers.runner_rewards import MazeRunnerRewards



class HungerGamesRewards(MazeRunnerRewards):

    def __init__(self, env, rewards_dist):
        super().__init__(env, rewards_dist)


    def step(self, action):
        new_obs, reward, terminated, truncated, info = super().step(action)

        for object in info['type_of_objects']:
            for k in range(info['n_'+object]):
                # --- check neighborhoods for goals and other agents --- #
                index_agent = +1000
                for d in ['CENTER','UP', 'DOWN','LEFT','RIGHT','UP_LEFT','UP_RIGHT','DOWN_LEFT','DOWN_RIGHT']:
                    try:
                        index_agent = min(index_agent,info[object + f'_{k}'][f'{d}_vision'].index(1))
                    except ValueError:
                        None
                # -- discourge getting closer to other agents -- #
                if index_agent!=1000:
                    reward[object][k] += self.rewards_dist[object]['TOO_CLOSE_CONSTANT'] + self.rewards_dist[object]['TOO_CLOSE']/(1 + index_agent)
                else:
                    reward[object][k] += self.rewards_dist[object]['NO_NEIGHBORS']

                if info[object + f'_{k}']['dead']:
                    reward[object][k] += self.rewards_dist[object]['HIT_OTHER']

                # -- normalize the rewards compared to the size of the maze -- #
                reward[object][k] = reward[object][k]/(info['max_pos']+1)
                #reward[k] = np.tanh(reward[k])

                self.cum_rewards[object][k] += reward[object][k]
        return new_obs, reward, terminated, truncated, info