from Agents import *
from Games import Tetris
from Models.Parameters import hyper_loader
from Models.Wrappers import *

import random
from copy import deepcopy
from Models.utils.tensorboard_handler import tb_handler
from tqdm import tqdm
from datetime import datetime
from statistics import mean, median


def get_args():
    import argparse
    parser = argparse.ArgumentParser("""Parameters for Tetris Agent""")
    parser.add_argument("--hyper_name", type = str, default = 'gnn_hyper', help = 'file name for hyper parameters')
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
    model = eval(args.train_hyper['wrapper_name'])(**args.train_hyper['model_params']).to(args.train_hyper['train_params']['device'])
    tb = tb_handler('./Models/runs/', args.train_hyper['wrapper_name'], model)

    # print(tb.model.state_dict())

    ## agentprint
    agent = Agent(tb_handler = tb, device = args.train_hyper['train_params']['device'], **args.agent_hyper)

    scores = []

    train_time, best_max, best_avg, best_min = 0, 0, 0, 100
    best_model = None

    for episode in range(args.train_hyper['train_params']['episodes']):

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
            best_action, best_state = agent.best_state(next_states)

            reward, done = env.play(best_action[0], best_action[1], render = render, render_delay = args.train_hyper['train_params']['render_delay'])
            
            agent.add_to_memory(current_state, best_state, reward, done)
            current_state = best_state
            steps += 1

        scores.append(env.get_game_score())

        # Train
        if episode % args.train_hyper['train_params']['train_every'] == 0:
            train_time += agent.train(batch_size = args.train_hyper['train_params']['batch_size'], epochs = args.train_hyper['train_params']['epochs'], times = train_time, optimizer_params = args.train_hyper['optimizer_params'])

        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            min_score = min(scores[-log_every:])
            max_score = max(scores[-log_every:])

            tb.add_scalar(avg_score, episode, f'Avg Score every {log_every} Episode')
            tb.add_scalar(max_score, episode, f'Max Score every {log_every} Episode')
            tb.add_scalar(min_score, episode, f'Min Score every {log_every} Episode')

            print(f'Episode {episode}----------------------------------------------\n[Scores For Last {log_every} Games] Avg Score: {avg_score}, Max Score: {max_score}, Min Score: {min_score}')

            if best_max <= max_score:
                best_max = max_score
                best_model = deepcopy(agent.tb_handler.model)

        if train_time > 0 and train_time % args.train_hyper['train_params']['save_every'] == 0:
            # print(best_model.state_dict())
            best_model.save_parameters()

if __name__ == "__main__":
    main()
