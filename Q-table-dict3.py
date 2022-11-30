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

# Global Storage
class StorageAgent:
    def __init__(self, num_workers):
        self.q_table = {}
        self.visit = {}
        self.num_workers = num_workers
        self.LocalRandGen = np.random.RandomState()
        # Choose a random process to update global Q-table and visit count
        self.buffer = deque()
        self.best_accuracy = 0.
        self.num_updates = 0

    def update_tables(self, q_table, visit, buffer, valid_accuracy):
        if self.num_updates == self.num_workers: 
            self.num_updates = 0
            self.best_accuracy = 0.
        # 1/num_workers chance of updating the global Q-table and visit count
        if valid_accuracy > self.best_accuracy:
            self.best_accuracy = valid_accuracy
            self.q_table = q_table
            self.visit = visit
            self.buffer = buffer
            self.best_accuracy = valid_accuracy

        self.num_updates += 1

    def choose_action_hard(self, state, actions_remain):
        # Choose action based on the Q-table 
        if state not in self.q_table:
            return self.LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(self.q_table[state]))
        for action in best_action:
            if action in actions_remain:
                return action
            
    def test(self, test_size=640): 
        env = make_env('test')
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

        print(f"Test Accuracy: {np.mean(acc)}") 
        print(f"Test Feature Length: {np.mean(feat_length)}") 

        return np.mean(acc), np.mean(feat_length)

    # Get global Q-table and global visit count
    def get_tables(self):
        return self.q_table, self.visit, self.buffer, self.best_accuracy

class StorageManager(BaseManager):
    pass

class Agent:
    def __init__(self): 
        #self.num_workers = cpu_count()
        self.num_workers = 1
        self.barrier = Barrier(self.num_workers)

    def run(self): 
        start_time = time.time()
        StorageManager.register('StorageAgent', StorageAgent)
        with StorageManager() as manager:
            global_storage = manager.StorageAgent(self.num_workers)

            workers = [WorkerAgent(self.barrier) for _ in range(self.num_workers)]
            processes = [Process(target=worker.train, args=(global_storage,)) for worker in workers]

            for process in processes:
                process.start()

            for process in processes:
                process.join()

            print(f"Training time: {time.time() - start_time} seconds")

            global_storage.test()

class ReplayBuffer:
    def __init__(self, capacity=10000, batch_size=512):
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

    def get_buffer(self):
        return self.buffer

    def set_buffer(self, buffer):
        self.buffer = buffer

class WorkerAgent:
    def __init__(self, barrier, update_freq=5000, pretrain=10000):
        # Multiprocessing
        self.barrier = barrier
        self.update_freq = update_freq
        self.pretrain = pretrain

        # Initialize environment
        self.env = make_env('train')
        self.valid_env = make_env('validation')
        self.state_dim = self.env.observation_space['observed'].shape[0]
        self.action_dim = self.env.action_space.n
        self.LocalRandGen = np.random.RandomState()

        # Create Q-table
        self.q_table = {} 
        self.q_table[tuple(np.zeros(self.state_dim, dtype=np.byte))] = np.zeros(self.action_dim) - 1.

        # Create visit count table
        self.visit = {}
        self.visit[tuple(np.zeros(self.state_dim, dtype=np.byte))] = np.zeros(self.action_dim, dtype=np.int32)

        # Initialize Replay
        self.replay_buffer = ReplayBuffer()

    def choose_action(self, q_row, actions_remain):
        # Choose action based on the Q-table 
        if self.LocalRandGen.random() < args.eps or np.count_nonzero(q_row == -1) == q_row.shape[0]:
            return self.LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(q_row))
        for action in best_action:
            if action in actions_remain:
                return action

    def choose_action_hard(self, state, actions_remain):
        if state not in self.q_table:
            return self.LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(self.q_table[state]))
        for action in best_action:
            if action in actions_remain:
                return action

    def replay(self, iters=10):
        if self.replay_buffer.size() <= self.replay_buffer.batch_size: return
        for _ in range(iters):
            samples = self.replay_buffer.sample()
            for sample in samples:
                state, action, reward, next_state, done = sample
                if not done:
                    self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward + args.y * np.max(self.q_table[next_state]) - self.q_table[state][action])
                    self.visit[state][action] += 1
                else:
                    self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward - self.q_table[state][action])
                    self.visit[state][action] += 1

    def episode(self):
        # Reset environment
        state = tuple(self.env.reset()['observed'].astype(np.byte))
        done = False
        total_reward = 0
        actions_remain = list(range(self.action_dim))
        feat_acq = 0
        correct = 0

        while True:
            # Choose action using epsilon-greedy policy
            action = self.choose_action(self.q_table[state], actions_remain)
            actions_remain.remove(action)
            if state not in self.visit:
                self.visit[state] = np.zeros(self.action_dim, dtype=np.int32)
            self.visit[state][action] += 1

            # Take action
            new_state, reward, done, info = self.env.step(action)
            new_state = tuple(new_state['observed'].astype(np.byte))
            total_reward += reward

            ############################ Update ############################

            if new_state not in self.q_table:
                self.q_table[new_state] = np.zeros(self.action_dim, dtype=np.float32) - 1.

            self.replay_buffer.put(state, action, reward, new_state, done)

            # Update Q-table
            if not done: 
                self.q_table[state][action] += (args.lr/self.visit[state][action])*(reward + args.y*np.max(self.q_table[new_state]) - self.q_table[state][action]) 
                if action < self.state_dim: 
                    feat_acq += 1
            else:
                self.q_table[state][action] += (args.lr/self.visit[state][action])*(reward - self.q_table[state][action])
                correct = int(action-self.state_dim == info['target'])
                break

            #################################################################

            # Update state
            state = deepcopy(new_state)

        return total_reward, feat_acq, correct


    # Global tables are only set if the worker's tables are better in terms of validation accuracy
    # This is done to prevent the worker from overwriting the global tables with worse tables
    def train(self, global_storage):
        global CUR_EPISODE

        valid_acc = 0.
        while valid_acc < 0.7: 
            reward, feat_acq, correct = self.episode()

            # Replay past experiences
            self.replay()

            if CUR_EPISODE % 1000 == 0: print(f'Episode: {CUR_EPISODE}, Q-Table Size: {len(self.q_table)}')

            if CUR_EPISODE % self.update_freq == 0 and CUR_EPISODE > self.pretrain:
                print(f'Current Process {current_process()}')
                global_storage.update_tables(self.q_table, self.visit, self.replay_buffer.get_buffer(), self.valid()[0])

                # Wait for all processes to finish updating
                self.barrier.wait()

                self.q_table, self.visit, temp_buffer, valid_acc = global_storage.get_tables()
                self.replay_buffer.set_buffer(temp_buffer) 
                self.replay_buffer.shuffle()
                print(f"Chosen Accuracy: {valid_acc}")

            # Decay epsilon 
            args.eps *= args.eps_decay 
            args.eps = max(args.eps, args.min_eps) 

            CUR_EPISODE += 1

    def valid(self, valid_size=640): 
        acc = [] 
        feat_length = [] 

        for _ in range(valid_size):
            s = tuple(self.valid_env.reset()['observed'].astype(np.byte)) 
            d = False 

            total_reward = 0 
            next_action = np.argmax(self.q_table[s])
            actions_remain = list(range(self.action_dim))
            actions_remain.remove(next_action)
            feat_acq = 0

            while not d: 
                # Choose action based on argmax of Q-table 
                action = next_action 
                # Get new state and reward from environment 
                s, r, d, info = self.valid_env.step(action) 
                s = tuple(s['observed'].astype(np.byte)) 
                total_reward += r 
                # Get next action 
                next_action = self.choose_action_hard(s, actions_remain)
                actions_remain.remove(next_action)
                if action < self.state_dim: feat_acq += 1

            acc.append(action-self.state_dim == info['target'])
            feat_length.append(feat_acq)

        print(f"Validation Accuracy: {np.mean(acc)}") 
        print(f"Validation Feature Length: {np.mean(feat_length)}") 

        return np.mean(acc), np.mean(feat_length)



def main():
    agent = Agent()
    agent.run()

if __name__ == '__main__':
    main()
