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

from multiprocessing import Process, Lock, cpu_count, current_process

# Hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='molecule_20',  help='Dataset used for environment')
parser.add_argument('--lr', type=float, default=1.0,          help='Learning rate')
parser.add_argument('--y', type=float, default=0.99,           help='Discount factor (gamma)')
parser.add_argument('--eps', type=float, default=1.0,          help='Epsilon for epsilon-greedy policy')
parser.add_argument('--eps_decay', type=float, default=1-5e-5, help='Epsilon decay')
parser.add_argument('--min_eps', type=float, default=5e-3,     help='Minimum epsilon')
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
    def __init__(self):
        self.q_table = {}
        self.visit = {}
        self.valid_accuracies = []
        self.save_episode = []

    # Train global Q-table and global visit count using exeriences collected by local agents
    def train(self, workerBuffer, state_dim, action_dim):
        for state, action, reward, next_state, done in workerBuffer:
            if state not in self.q_table:
                self.q_table[state] = np.zeros(action_dim, dtype=np.float32) - 1.
                self.visit[state] = np.zeros(action_dim, dtype=np.int32)
            if next_state not in self.q_table:
                self.q_table[next_state] = np.zeros(action_dim, dtype=np.float32) - 1.
                self.visit[next_state] = np.zeros(action_dim, dtype=np.int32)

            self.visit[state][action] += 1

            if not done:
                self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward + args.y * np.max(self.q_table[next_state]) - self.q_table[state][action])
            else:
                self.q_table[state][action] += (args.lr/self.visit[state][action]) * (reward - self.q_table[state][action])

    def save_accuracy(self, acc):
        self.valid_accuracies.append(acc)
        self.save_episode.append(CUR_EPISODE)

    def get_accuracy(self):
        return self.valid_accuracies, self.save_episode

    def choose_action_hard(self, state, actions_remain):
        LocalRandGen = np.random.RandomState()
        # Choose action based on the Q-table 
        if state not in self.q_table:
            return LocalRandGen.choice(actions_remain)
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
        return self.q_table, self.visit

class Agent:
    def __init__(self): 
        self.num_workers = cpu_count()
        self.global_storage = StorageAgent()

    def train(self): 
        workers = [WorkerAgent(self.global_storage) for _ in range(self.num_workers)] 
        processes = [Process(target=w.train, args=()) for w in workers]

        start_time = time.time()

        for p in processes:
            p.start()

        for p in processes:
            p.join()

        print(f"Training time: {time.time() - start_time} seconds")
        valid_acc, save_eps = self.global_storage.get_accuracy()
        return valid_acc, save_eps

    def test(self):
        return self.global_storage.test()


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

class WorkerAgent:
    def __init__(self, global_storage, update_freq=1000, update_threshold=0):
        # Multiprocessing
        self.lock = Lock()
        self.update_freq = update_freq
        self.update_threshold = update_threshold

        # Initialize environment
        self.env = make_env('train')
        self.valid_env = make_env('validation')
        self.state_dim = self.env.observation_space['observed'].shape[0]
        self.action_dim = self.env.action_space.n
        self.global_storage = global_storage

        # Create Q-table
        self.q_table = {} 
        self.q_table[tuple(np.zeros(self.state_dim, dtype=np.byte))] = np.zeros(self.action_dim) - 1.

        # Create visit count table
        self.visit = {}
        self.visit[tuple(np.zeros(self.state_dim, dtype=np.byte))] = np.zeros(self.action_dim, dtype=np.int32)

        # Initialize Replay
        self.replay_buffer = ReplayBuffer()

    def choose_action(self, q_row, actions_remain):
        LocalRandGen = np.random.RandomState()
        # Choose action based on the Q-table 
        if LocalRandGen.random() < args.eps or np.count_nonzero(q_row == -1) == q_row.shape[0]:
            return LocalRandGen.choice(actions_remain)
        best_action = np.flip(np.argsort(q_row))
        for action in best_action:
            if action in actions_remain:
                return action

    def choose_action_hard(self, state, actions_remain):
        LocalRandGen = np.random.RandomState()
        if state not in self.q_table:
            return LocalRandGen.choice(actions_remain)
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

            self.replay_buffer.put(state, action, reward, new_state, done)

            # Initialize Q-table row for new state
            if new_state not in self.q_table:
                self.q_table[new_state] = np.zeros(self.action_dim) - 1.

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
    def train(self): 
        global CUR_EPISODE

        valid_acc = 0.
        #while valid_acc < 0.7: 
        while CUR_EPISODE <= 100000:
            reward, feat_acq, correct = self.episode()

            # Replay past experiences
            self.replay()

            if CUR_EPISODE % self.update_freq == 0 and CUR_EPISODE > self.update_threshold:
                self.lock.acquire() 
                ############################ Protected ############################
                self.global_storage.train(self.replay_buffer.get_buffer(), self.state_dim, self.action_dim)
                self.q_table, self.visit = self.global_storage.get_tables()
                print(f'Episode {CUR_EPISODE}: Current Process {current_process()}')
                valid_acc = self.valid()[0]
                self.global_storage.save_accuracy(valid_acc)
                ###################################################################
                self.lock.release()

            self.replay_buffer.shuffle()
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
        print(f"Q-table Size: {len(self.q_table)}")

        return np.mean(acc), np.mean(feat_length)



def main():
    agent = Agent()
    valid_acc, save_eps = agent.train()
    np.save('valid_acc.npy', np.array(valid_acc, dtype=np.float32))
    np.save('save_eps.npy', np.array(save_eps, dtype=np.int32))
    agent.test()

if __name__ == '__main__':
    main()
