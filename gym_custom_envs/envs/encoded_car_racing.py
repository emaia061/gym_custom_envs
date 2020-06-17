import gym
import torch
import gym.envs.box2d.car_racing as cr
from gym import spaces

class EncodedCarRacing(cr.CarRacing){

    def __init__(self, encoder_path, verbose=1):
        super(EncodedCarRacing, self).__init__(verbose) # call CarRacing init
        self.encoder = torch.load(encoder_path) # raises exception if not available
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=32, dtype=np.float32)

    def step():
        if action is not None:
            self.car.steer(-action[0])
            self.car.gas(action[1])
            self.car.brake(action[2])

        self.car.step(1.0/cr.FPS)
        self.world.Step(1.0/cr.FPS, 6*30, 2*30)
        self.t += 1.0/cr.FPS

        self.state = self.render("state_pixels") # returns ndarray
        self.state = self.encoder(torch.transforms.ToTensor(self.state))
        step_reward = 0
        done = False
        if action is not None: # First step without action, called from reset()
            self.reward -= 0.1
            # We actually don't want to count fuel spent, we want car to be faster.
            # self.reward -=  10 * self.car.fuel_spent / ENGINE_POWER
            self.car.fuel_spent = 0.0
            step_reward = self.reward - self.prev_reward
            self.prev_reward = self.reward
            if self.tile_visited_count == len(self.track):
                done = True
            x, y = self.car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                done = True
                step_reward = -100

        return self.state, step_reward, done, {}

}
