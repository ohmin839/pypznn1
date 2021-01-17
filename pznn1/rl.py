import argparse
import copy
import os
import random
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
tf.compat.v1.disable_eager_execution()

from pznn1.core import Puzzle, Direction
from pznn1.util import dir2idx, idx2dir, reverse_shuffle

Experience = namedtuple("Experience",
                        ["s", "a", "r", "n_s", "d"])

class Environment:
    def __init__(self, size):
        self.steps = 0
        self.shuffles = 0
        self.puzzle = Puzzle(size)
        self.limit = 50

    def reset(self, shuffles):
        self.steps = 0
        self.shuffles = shuffles
        directions = reverse_shuffle(self.puzzle, shuffles)
        return self.puzzle.get_grid(), directions

    def step(self, direction):
        self.steps += 1
        if not self.puzzle.can_move(direction):
            grid = self.puzzle.get_grid()
            return grid, -1, True, self.steps
        else:
            grid = self.puzzle.move(direction).get_grid()
            if self.puzzle.has_completed():
                return grid, 1, True, self.steps
            elif self.steps >= self.limit:
                return grid, -(1e-3), True, self.steps
            else:
                return grid, -(1e-3), False, self.steps

class Observer:
    def __init__(self, env):
        self.env = env

    def reset(self, shuffles):
        grid, directions = self.env.reset(shuffles)
        h, w = grid.shape
        return grid.reshape(1, h, w, 1), [dir2idx(d) for d in directions]

    def step(self, action):
        direction = idx2dir(action)
        grid, reward, done, step = self.env.step(direction)
        h, w = grid.shape
        return grid.reshape(1, h, w, 1), reward, done, step

class Student:
    def __init__(self):
        self.actions = [dir2idx(d) for d in Direction.get_directions()]
        self.model = None
        self.updater = None

    def init_training_phase1(self, size, optimizer):
        self.set_model(size)
        self.model.compile(optimizer, loss="sparse_categorical_crossentropy")

    def init_training_phase2(self, size, optimizer):
        self.reload_model()
        self.set_policy_gradient_updater(optimizer)

    def set_model(self, size):
        self.actions = [dir2idx(d) for d in Direction.get_directions()]
        normal = K.initializers.glorot_normal()
        self.model = K.models.Sequential([
            K.layers.Conv2D(input_shape=(size, size, 1),
                            filters=len(self.actions)**3,
                            kernel_size=size-1, padding="same",
                            kernel_initializer=normal,
                            activation="relu"),
            K.layers.Conv2D(filters=len(self.actions)**3,
                            kernel_size=size-1, padding="same",
                            kernel_initializer=normal,
                            activation="relu"),
            K.layers.Conv2D(filters=len(self.actions)**3,
                            kernel_size=size-1, padding="same",
                            kernel_initializer=normal,
                            activation="relu"),
            K.layers.Flatten(),
            K.layers.Dense(units=len(self.actions)**2,
                           kernel_initializer=normal,
                           activation="relu"),
            K.layers.Dense(units=len(self.actions),
                           kernel_initializer=normal,
                           activation="softmax"),
        ])

    def reload_model(self, file_path="model.h5"):
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.model = K.models.load_model(file_path)

    def set_policy_gradient_updater(self, optimizer):
        actions = tf.compat.v1.placeholder(shape=(None), dtype="int32")
        rewards = tf.compat.v1.placeholder(shape=(None), dtype="float32")
        one_hot_actions = tf.one_hot(actions, len(self.actions), axis=1)
        action_probs = self.model.output
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs,
                                              axis=1)
        clipped = tf.clip_by_value(selected_action_probs, 1e-10, 1.0)
        loss = - tf.math.log(clipped) * rewards
        loss = tf.reduce_mean(loss)

        updates = optimizer.get_updates(loss=loss,
                                        params=self.model.trainable_weights)
        self.updater = K.backend.function(
                                        inputs=[self.model.input,
                                                actions, rewards],
                                        outputs=[loss],
                                        updates=updates)

    def update_by_crossentropy(self, states, actions):
        self.model.train_on_batch(
                np.vstack(states),
                np.array(actions))

    def update_by_policy_gradient(self, policy_experiences):
        length = len(policy_experiences)
        batch = random.sample(policy_experiences, length)
        states = [e.s for e in batch]
        actions = [e.a for e in batch]
        rewards = [e.r for e in batch]
        self.updater([np.vstack(states),
                      np.array(actions),
                      np.array(rewards)])

    def estimate(self, state):
        return self.model.predict(state)[0]

    def policy(self, state):
        estimateds = self.estimate(state)
        action = np.random.choice(self.actions, size=1, p=estimateds)[0]
        return action

    def save(self, file_path="model.h5"):
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.model.save(file_path, overwrite=True, include_optimizer=False)

    def test(self, obs, episodes, shuffles):
        successes = 0
        for e_i in range(episodes):
            state, answer_actions = obs.reset(shuffles)
            done = False
            while not done:
                action = self.policy(state)
                next_state, reward, done, steps = obs.step(action)
                state = next_state
            else:
                if reward == 1:
                    successes += 1
        else:
            success_rate = successes * 100 / episodes
            return success_rate

    def demo(self, size, shuffles):
        env = Environment(size)
        obs = Observer(env)
        pz = env.puzzle

        self.reload_model()
        model = self.model

        state, _ = obs.reset(shuffles)
        done = False
        steps = 0

        while not done:
            action = self.policy(state)
            predicted = model.predict(pz.get_grid().reshape(1, size, size, 1))
            print(env.puzzle, idx2dir(action), predicted)
            state, reward, done, steps = obs.step(action)
        print(env.puzzle)
        if reward == 1:
            print(f"Conguatulations!({steps} steps)")
        elif reward == -1:
            print(f"Failed!({steps} steps)")
        else:
            print(f"Exceeded limit steps!({steps} steps)")

class Trainer:
    def __init__(self, size, lr=1e-3):
        self.size = size
        self.optimizer = K.optimizers.Adam(lr=lr)

    def train_phase1(self, obs, student, gamma, episodes, shuffles, report_interval=1000):
        actions = []
        states = []
        student.init_training_phase1(self.size, self.optimizer)
        for e_i in range(1, episodes+1):
            state, answer_actions = obs.reset(shuffles)
            done = False
            steps = 0
            while not done:
                states.append(state)
                action_t = answer_actions[steps]
                action_y = student.policy(state)
                state, reward, done, steps = obs.step(action_t)
                actions.append(action_t)
            else:
                student.update_by_crossentropy(states, actions)
                actions = []
                states = []
                if e_i % report_interval == 0:
                    print(f"episodes: {e_i}")
        else:
            success_rate = student.test(obs, 1000, shuffles)
            print(f"episodes: {report_interval}, success_rate: {success_rate:3.2f}%")

    def train_phase2(self, obs, student, gamma, episodes, shuffles, report_interval=100000):
        student.init_training_phase2(self.size, self.optimizer)
        x = []
        y = []
        for e_i in range(1, episodes+1):
            state, answer_actions = obs.reset(shuffles)
            done = False
            experiences = []
            episode_reward = 0
            steps = 0
            while not done:
                action = answer_actions[steps]
                next_state, reward, done, steps = obs.step(action)
                e = Experience(state, action, reward, next_state, done)
                experiences.append(e)
                state = next_state
                episode_reward += reward
            else:
                policy_experiences = []
                rewards = [e.r for e in experiences]
                if rewards[-1] == 1:
                    for t, e in enumerate(experiences):
                        s, a, r, n_s, d = e
                        d_r = [_r * (gamma ** i) for i, _r in enumerate(rewards[t:])]
                        d_r = sum(d_r)
                        d_e = Experience(s, a, d_r, n_s, d)
                        policy_experiences.append(d_e)
                elif rewards[-1] == -1:
                    policy_experiences.append(experiences[-1])
                if len(policy_experiences) > 0:
                    student.update_by_policy_gradient(policy_experiences)
                if e_i % report_interval == 0:
                    success_rate = student.test(obs, 1000, shuffles)
                    print(f"episodes: {e_i}, success_rate: {success_rate:3.2f}%")
                    x.append(e_i)
                    y.append(success_rate)
        else:
            plt.plot(x, y)
            plt.show()

def main(size=3, gamma=0.999, lr=1e-3, episodes=1000, shuffles=10, action="play"):
    env = Environment(size)
    obs = Observer(env)
    student = Student()
    if action == "train1":
        trainer = Trainer(size, lr)
        trainer.train_phase1(
                obs, student,
                gamma, episodes, shuffles)
        student.save()
    elif action == "train2":
        trainer = Trainer(size, lr)
        trainer.train_phase2(
                obs, student,
                gamma, episodes, shuffles)
        student.save()
    elif action == "test":
        self.reload_model()
        student.test(obs, episodes, shuffles)
    elif action == "demo":
        student.demo(size, shuffles)
    else:
        print(f"{action} is not supported option.")
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinforcement lerning against NxN Puzzle")
    parser.add_argument("--size", type=int, default=3,
                        help="specify the size of puzzle")
    parser.add_argument("--gamma", type=float, default=0.999,
                        help="")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="")
    parser.add_argument("--shuffles", type=int, default=10,
                        help="")
    parser.add_argument("--action", default="test",
                        help="")
    args = parser.parse_args()

    main(
        size=args.size, gamma=args.gamma, lr=args.lr,
        episodes=args.episodes, shuffles=args.shuffles,
        action=args.action)

