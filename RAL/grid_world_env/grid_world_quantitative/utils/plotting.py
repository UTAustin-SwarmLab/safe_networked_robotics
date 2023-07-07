from cProfile import label
from sre_parse import State
from turtle import color
import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "game_status"])
TestStats_pm = namedtuple("TestStats_pm",["robot", "adversary", "reward", "game_status"])

def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    rewards_smoothed.to_csv('src/data/learning_curve_w_shield.csv', index=False)
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(np.cumsum(stats.episode_lengths), np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show()

    fig4 = plt.figure(figsize=(10,5))
    plt.bar(['Won', 'Lost', 'Tie'], stats.game_status.tolist())
    plt.xlabel("Game Status")
    plt.ylabel("No of events")
    plt.title("Training: Game status at end of each episode")
    if noshow:
        plt.close(fig4)
    else:
        plt.show()

    return fig1, fig2, fig3, fig4

def plot_test_stats_pm(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig, axes = plt.subplots(1,1)
    axes.plot(*zip(*stats.robot), label = 'robot path',color='blue')
    axes.plot(*zip(*stats.adversary), label = 'adversary path', color='red')
    axes.plot(*stats.robot[-1], marker='o',color='blue')
    axes.plot(*stats.adversary[-1], marker='o',color='red')
    axes.set_xlabel("X-Axis")
    axes.set_ylabel("Y-Axis")
    axes.set_title("Paths")
    axes.set_xlim([-1, 10])
    axes.set_ylim([-1, 10])
    axes.grid()
    axes.legend()

    if noshow:
        plt.close(fig)
    else:
        plt.show()

    fig1 = plt.figure(figsize=(10,5))
    plt.bar(['Won', 'Lost', 'Tie'], stats.game_status.tolist())
    plt.xlabel("Game Status")
    plt.ylabel("No of events")
    plt.title("Testing: Game status at end of each episode")
    if noshow:
        plt.close(fig1)
    else:
        plt.show()

    return fig, fig1
