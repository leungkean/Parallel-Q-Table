import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
from copy import deepcopy
import argparse

from afa.data import load_unsupervised_split_as_numpy, load_supervised_split_as_numpy
from afa.environments.dataset_manager import EnvironmentDatasetManager
from afa.environments.core import DirectClassificationEnv

import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle
import time

from multiprocessing import Process, Lock, cpu_count, current_process, Barrier
from multiprocessing.managers import BaseManager

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='molecule_20',  help='Dataset used for environment')
parser.add_argument('--lr', type=float, default=1.0,          help='Learning rate')
parser.add_argument('--y', type=float, default=0.99,           help='Discount factor (gamma)')
parser.add_argument('--eps', type=float, default=1.0,          help='Epsilon for epsilon-greedy policy')
parser.add_argument('--eps_decay', type=float, default=1-5e-5, help='Epsilon decay')
parser.add_argument('--min_eps', type=float, default=1e-3,     help='Minimum epsilon')
parser.add_argument('--cost', type=float, default=0.001,        help='Acquisition cost')

args = parser.parse_args()

# Episode counter
CUR_EPISODE = 0

def make_env(data_select):
    features, targets = load_supervised_split_as_numpy(args.dataset, data_select)
    dataset_manager = EnvironmentDatasetManager(
            features, targets
    )

    env = DirectClassificationEnv(
            dataset_manager,
            incorrect_reward=-1.0,
            acquisition_cost=args.cost,
    )
    return env

class ReplayBuffer:
    def __init__(self, capacity=20000, batch_size=512):
        self.buffer = deque(maxlen=capacity)
        self.batch_size = batch_size

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        return random.sample(self.buffer, self.batch_size)

    def size(self):
        return len(self.buffer)

    def shuffle(self):
        random.shuffle(self.buffer)

# Global Storage
class StorageAgent:
    def __init__(self, num_workers):
        self.q_table = {}
        self.visit = {}
        self.num_workers = num_workers
        self.LocalRandGen = np.random.RandomState()
        self.replay_buffer = ReplayBuffer()
        self.valid_env = make_env('validation')
        self.best_acc = 0.0

    def get_row(self, state, action_dim):
        if state not in self.q_table:
            self.q_table[state] = np.zeros(action_dim, dtype=np.float32) - 1.
            self.visit[state] = np.zeros(action_dim, dtype=np.int32)
        return self.q_table[state]

    def update(self, update): 
        state, action, reward, next_state, done, action_dim = update 
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(action_dim, dtype=np.float32) - 1.
            self.visit[next_state] = np.zeros(action_dim, dtype=np.int32)

        self.visit[state][action] += 1

        if not done: 
            self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward + args.y * np.max(self.q_table[next_state]) - self.q_table[state][action]) 
        else: 
            self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward - self.q_table[state][action])

        self.replay_buffer.put(state, action, reward, next_state, done)

    def replay(self, iters=10):
        if self.replay_buffer.size() <= self.replay_buffer.batch_size: return
        for _ in range(iters):
            samples = self.replay_buffer.sample()
            for sample in samples:
                state, action, reward, next_state, done = sample
                if not done:
                    self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward + args.y * np.max(self.q_table[next_state]) - self.q_table[state][action])
                else:
                    self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward - self.q_table[state][action])
                self.visit[state][action] += 1

        self.replay_buffer.shuffle()

    def choose_action_hard(self, state, actions_remain):
        # Choose action based on the Q-table 
        if state not in self.q_table:
            return self.LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(self.q_table[state]))
        for action in best_action:
            if action in actions_remain:
                return action
            
    # Validation for valid_test = 0, test for valid_test = 1
    def eval(self, valid_test=0, test_size=640): 
        if valid_test == 0:
            env = self.valid_env
            dataset = 'Validation'
        else: 
            env = make_env('test')
            dataset = 'Test'
        state_dim = env.observation_space['observed'].shape[0]
        action_dim = env.action_space.n
        acc = [] 
        feat_length = [] 

        for _ in range(test_size):
            s = tuple(env.reset()['observed'].astype(np.byte)) 
            d = False 

            total_reward = 0 
            actions_remain = list(range(action_dim))
            next_action = self.choose_action_hard(s, actions_remain)
            actions_remain.remove(next_action)
            feat_acq = 0

            while not d: 
                # Choose action based on argmax of Q-table 
                action = next_action 
                # Get new state and reward from environment 
                s, r, d, info = env.step(action) 
                s = tuple(s['observed'].astype(np.byte)) 
                total_reward += r 
                # Get next action 
                next_action = self.choose_action_hard(s, actions_remain) 
                actions_remain.remove(next_action)
                feat_acq += 1

            acc.append(action-state_dim == info['target'])
            feat_length.append(feat_acq)

        print(f"{dataset} Accuracy: {np.mean(acc)}, Features Acquired: {np.mean(feat_length)}, Q-Table Size: {len(self.q_table)}")

        eval_acc = np.mean(acc)
        if eval_acc > self.best_acc:
            self.best_acc = eval_acc

        return eval_acc

    def get_best_acc(self):
        return self.best_acc

class StorageManager(BaseManager):
    pass

class Agent:
    def __init__(self): 
        #self.num_workers = cpu_count()
        self.num_workers = 4
        self.barrier = Barrier(self.num_workers)

    def run(self): 
        start_time = time.time()
        StorageManager.register('StorageAgent', StorageAgent)
        with StorageManager() as manager:
            global_storage = manager.StorageAgent(self.num_workers)

            workers = [WorkerAgent(self.barrier, self.num_workers) for _ in range(self.num_workers)]
            processes = [Process(target=worker.train, args=(global_storage,)) for worker in workers]

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            print(f"Training time: {time.time() - start_time} seconds")

            global_storage.eval(valid_test=1)

class WorkerAgent:
    def __init__(self, barrier, num_workers, update_freq=5000, replay_freq=1000):
        # Multiprocessing
        self.barrier = barrier
        self.num_workers = num_workers
        self.update_freq = update_freq
        self.replay_freq = replay_freq

        # Initialize environment
        self.env = make_env('train')
        self.state_dim = self.env.observation_space['observed'].shape[0]
        self.action_dim = self.env.action_space.n
        self.LocalRandGen = np.random.RandomState()

    def choose_action(self, q_row, actions_remain):
        # Choose action based on the Q-table 
        if self.LocalRandGen.random() < args.eps or np.count_nonzero(q_row == -1) == q_row.shape[0]:
            return self.LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(q_row))
        for action in best_action:
            if action in actions_remain:
                return action

    def episode(self, global_storage):
        # Reset environment
        state = tuple(self.env.reset()['observed'].astype(np.byte))
        done = False
        actions_remain = list(range(self.action_dim))

        while not done:
            q_row = global_storage.get_row(state, self.action_dim)
            # Choose action using epsilon-greedy policy
            action = self.choose_action(q_row, actions_remain)
            actions_remain.remove(action)

            # Take action
            new_state, reward, done, info = self.env.step(action)
            new_state = tuple(new_state['observed'].astype(np.byte))

            ############################ Update ############################ 

            update = (state, action, reward, new_state, done, self.action_dim)
            global_storage.update(update)

            #################################################################

            # Update state
            state = deepcopy(new_state)

    # Global tables are only set if the worker's tables are better in terms of validation accuracy
    # This is done to prevent the worker from overwriting the global tables with worse tables
    def train(self, global_storage):
        global CUR_EPISODE

        valid_acc = 0.
        while valid_acc < 0.7:
            self.episode(global_storage)
            
            # Experience replay
            if CUR_EPISODE % self.replay_freq == 0 and CUR_EPISODE > 0:
                global_storage.replay()

            if CUR_EPISODE % self.update_freq == 0 and CUR_EPISODE > 0:
                # Wait for all processes to finish updating
                self.barrier.wait()
                
                print(f'Episode: {CUR_EPISODE}, Current Process {current_process()}')
                global_storage.eval()

                self.barrier.wait()

                valid_acc = global_storage.get_best_acc()

            # Decay epsilon 
            args.eps *= args.eps_decay 
            args.eps = max(args.eps, args.min_eps) 

            CUR_EPISODE += 1


def main():
    agent = Agent()
    agent.run()

if __name__ == '__main__':
    main()
