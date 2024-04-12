import numpy as np
GAMMA = 0.9

class State:
    def __init__(self, name, reward):
        self.name = name
        self.reward = reward
        self.actions = {}
        self.expected_rewards = []
        self.expected_reward = reward

    def add_transition(self, action, state, probability):
        if action in self.actions:
            self.actions[action].append((state, probability))
        else:
            self.actions[action] = [(state, probability)]
    
    # Called at the end of iteration
    def update_expected_reward(self):
        self.expected_rewards.append(self.expected_reward)
    
    # Called to calculate expected reward
    def calculate_expected_reward(self):
        # Maximum rewarding action
        maximum_reward = -1 * np.inf
        for action in self.actions:
            action_reward = 0
            for state, probability in self.actions[action]:
                action_reward += state.expected_rewards[-1] * probability
            maximum_reward = np.maximum(action_reward, maximum_reward)
        
        expected_reward = self.reward + GAMMA * maximum_reward
        
        if np.isclose(self.expected_reward, expected_reward):
            return True
        else:
            self.expected_reward = expected_reward
            return False


if __name__ == "__main__":
    sun = State("sun", 4)
    wind = State("wind", 0)
    hail = State("hail", -8)
    
    sun.add_transition("a", wind, 0.5)
    sun.add_transition("a", sun, 0.5)
    
    wind.add_transition("a", sun, 0.5)
    wind.add_transition("a", hail, 0.5)
    
    hail.add_transition("a", wind, 0.5)
    hail.add_transition("a", hail, 0.5)
    
    states = [sun, wind, hail]
    shouldStopCheck = [False] * len(states)
    
    print("Iteration", end="\t\t")
    for state in states:
        print(f"{state.name}", end="\t\t")
    print()
    
    i = 0
    while not all(shouldStopCheck):
        for state in states:
            state.update_expected_reward()
        
        print(f"{i}", end="\t\t")
        for state in states:
            print(f"{state.expected_reward: .2f}", end="\t\t")
        print()

        for index, state in enumerate(states):
            shouldStopCheck[index] = state.calculate_expected_reward()
        
        i += 1
