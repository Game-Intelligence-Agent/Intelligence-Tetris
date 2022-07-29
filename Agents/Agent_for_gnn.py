from os import stat
import torch

import numpy as np
import random
from collections import deque

from .Agent import Agent

from Models.utils import build_optimizer


# Deep Q Learning Agent + Maximin
#
# This version only provides only value per input,
# that indicates the score expected in that state.
# This is because the algorithm will try to find the
# best final state for the combinations of possible states,
# in constrast to the traditional way of finding the best
# action for a particular state.
class Agent_for_gnn(Agent):

    '''Deep Q Learning Agent + Maximin

    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''

    ## 每个state都应该是一个graph，graph里面带有mask来标识wall以及tetris

    def __init__(self, state_size, tb_handler, device, mem_size = 10000, discount = 0.95,
                 epsilon = 1, epsilon_min = 0, epsilon_stop_episode = 500, replay_start_size=None):


        self.state_size = state_size
        ## save state as graph
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (epsilon_stop_episode)
        self.tb_handler = tb_handler
        self.device = device if torch.cuda.is_available() else 'cpu'
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        self.model = self._build_model()

    def tranform(self, state):

        
        return state


    def train(self, batch_size = 32, epochs = 3, times = 0, optimizer_params = {}):

        '''Trains the agent'''
        n = len(self.memory)
    
        if n >= self.replay_start_size and n >= batch_size:

            scheduler, optimizer = build_optimizer(optimizer_params, self.tb.model.parameters())

            for epoch in range(epochs):

                batch = random.sample(self.memory, batch_size)

                # Get the expected score for the next states, in batch (better performance)
                next_states = np.array([x[1] for x in batch])
                with torch.no_grad():
                    next_qs = [x[0] for x in self.model.predict(next_states)]

                x = []
                y = []

                # Build xy structure to fit the model in batch (better performance)
                for i, (state, _, reward, done) in enumerate(batch):
                    if not done:
                        # Partial Q formula
                        new_q = reward + self.discount * next_qs[i]
                    else:
                        new_q = reward

                    x.append(state)
                    y.append(new_q)

                x = torch.Tensor(x).to(self.device)
                y = torch.Tensor(y).to(self.device).reshape(-1, 1)

                # Fit the model to the given values
                self.tb_handler.model.train()
                optimizer.zero_grad()
                self.tb_handler.add_graph(x)
                preds = self.tb_handler.model(x).reshape(-1, 1)

                loss = self.tb_handler.model.loss(preds, y)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                    
                self.tb_handler.show_params(times * epochs + epoch)
                self.tb_handler.add_scalar(self, loss.item(), times * epochs + epoch, 'loss')

            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
