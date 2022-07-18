from Agents import Agent
from Games import Tetris
from Models.Parameters import hyper_loader
from Models.Wrappers import *

import random
from Models.utils.tensorboard_handler import tb_handler
from tqdm import tqdm
from datetime import datetime
from statistics import mean, median


def get_args():
    import argparse
    parser = argparse.ArgumentParser("""Parameters for Tetris Agent""")
    parser.add_argument("--hyper_name", type = str, default = 'cnn_hyper', help = 'file name for hyper parameters')
    args = parser.parse_args()

    args.agent_hyper = hyper_loader('agent_hyper')
    args.train_hyper = hyper_loader(args.hyper_name)

    return args

# Run dqn with Tetris
def main():

    ## params
    args = get_args()
    log_every = args.train_hyper['train_params']['log_every']

    ## env
    env = Tetris()

    ## model
    model = eval(args.train_hyper['wrapper_name'])(**args.train_hyper['model_params'])
    tb = tb_handler('./Models/runs/', args.train_hyper['wrapper_name'], model)

    ## agent
    agent = Agent(state_size = env.get_state_size(), tb_handler = tb, **args['agent_hyper'])

    scores = []

    for episode in tqdm(range(args.train_hyper['train_params']['episodes'])):

        current_state = env.reset()
        done = False
        steps = 0

        if args.train_hyper['train_params']['render_every'] and episode % args.train_hyper['train_params']['render_every'] == 0:
            render = True
        else:
            render = False

        # Game
        while not done and (not args.train_hyper['train_params']['max_steps'] or steps < args.train_hyper['train_params']['max_steps']):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render = render, render_delay = args.train_hyper['train_params']['render_delay'])
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % args.train_hyper['train_params']['train_every'] == 0:
            agent.train(batch_size = args.train_hyper['train_params']['batch_size'], epochs = args.train_hyper['train_params']['epochs'])

        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])


            print(f'Episode {episode}----------------------------------------------\n[Scores For Last {log_every} Games] Avg Score: {avg_score}, Max Score: {max_score}, Min Score: {min_score}')


if __name__ == "__main__":
    main()
