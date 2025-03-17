

class DistanceLevels:
    def __init__(self,max_dist,n_levels):
        self.max_dist = max_dist 

        self.dist_rate = self.__setup_dist__(max_dist,n_levels)
        self.start_dist = self.dist_rate

    def __setup_dist__(self, max_dist, n_levels):
        rate = max_dist//n_levels
        if rate == 0:
            return 1
        else:
            return rate
        
    def get_level(self,dist):
        if dist <= self.max_dist:
            return dist//self.dist_rate - 1
        else:
            return -1
        
    def step(self):
        if self.start_dist<=self.max_dist:
            self.start_dist += self.dist_rate
        