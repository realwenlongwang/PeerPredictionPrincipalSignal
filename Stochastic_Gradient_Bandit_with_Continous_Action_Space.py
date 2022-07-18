#!/usr/bin/env python
# coding: utf-8

# # Contextual Bandits Agent with Policy Gradient Method (Stochastic) in Prediction Markets Problem
# ---
# This is a program that simulates an agent who trades in a prediction market. The problem that the prediction market aims to solve is to predict the real distribution of a random variable. We define the random variable as the colour of a bucket. The problem design comes from a human-subjective experiment for decision markets.

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from main import stochastic_training
from Environment import *
plt.rcParams.update({'font.size': 20})
import pyarrow.feather as feather
import os
from scipy.ndimage import uniform_filter1d


# In[4]:


training_platform = TrainingPlatform.Python
agent_num=3
action_num=2
signal_size =1
learning_rate_theta = 1e-4 / signal_size
learning_rate_wv = 1e-4 / signal_size
memory_size = 16
batch_size = 16
training_episodes = int(8e4)
decay_rate = 0
beta1 = 0.9
beta2 = 0.9999
algorithm = Algorithm.REGULAR
learning_std = False
fixed_std = 0.3
# Bucket parameters
pr_red_ball_red_bucket = 2/3
pr_red_ball_blue_bucket = 1/3
# prior_red_list = logit([2/3, 1/2, 1/3])
prior_red_list = None
preferred_colour_pr_list = [0.99, 0.01]
score_func = ScoreFunction.LOG
decision_rule = DecisionRule.DETERMINISTIC
agent_list = []
evaluation_step = 10
weights_init = WeightsInit.RANDOM
report_order = ReportOrder.FIXED
signal_size_list = np.ones(shape=agent_num, dtype=int) * signal_size

test_times = 3

for i in range(test_times):
    print(f'Test {i}')
    metric_dict = stochastic_training(
                                 training_platform, agent_list, learning_rate_theta, learning_rate_wv,
                                 memory_size, batch_size, training_episodes,
                                 decay_rate, beta1, beta2, algorithm, learning_std,
                                 fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                 prior_red_list, agent_num, action_num, score_func, decision_rule,
                                 preferred_colour_pr_list, evaluation_step, weights_init, report_order, signal_size_list)

    sequence_number = 1
    parent_folder_name = 'Deterministic' # 'Deterministic' or 'Stochastic'

    dir_name = r'ag{}_ac{}_sig{}_lrt{}_te{}_{}_{}_{}_{}_{}_{}_{}_{}/'.format(agent_num, action_num, signal_size, learning_rate_theta,
                                        training_episodes, algorithm.name,
                                        'normal' if prior_red_list is None else 'list',
                                        preferred_colour_pr_list, score_func.name,
                                        decision_rule.name, weights_init.name, report_order.name, sequence_number)

    dir_path = r'./data/{}/{}'.format(parent_folder_name, dir_name)

    while os.path.exists(dir_path):
        sequence_number += 1
        dir_name = r'ag{}_ac{}_sig{}_lrt{}_te{}_{}_{}_{}_{}_{}_{}_{}_{}/'.format(agent_num, action_num, signal_size, learning_rate_theta,
                                        training_episodes, algorithm.name,
                                        'normal' if prior_red_list is None else 'list',
                                        preferred_colour_pr_list, score_func.name,
                                        decision_rule.name, weights_init.name, report_order.name, sequence_number)
        dir_path = r'./data/{}/{}'.format(parent_folder_name, dir_name)



    os.makedirs(dir_path)

    backwards_index = -int(1e6)

    fig, ax = plt.subplots(figsize=(15,4))
    ax.plot(metric_dict['loss'],'.',markersize=0.1, label='loss')
    average_loss = uniform_filter1d(metric_dict['loss'],size=100)
    ax.plot(average_loss,'.', markersize=0.1, label='running window average')
    lgnd = ax.legend(loc='upper right',markerscale=100)
    xlabels = ax.get_xticks().tolist()
    xlabels = [str(xlabel)[0] for xlabel in xlabels]
    ax.set_xticklabels(xlabels)
    ax.set_xlabel('Millon Steps')
    fig.suptitle('Root Mean Squared Log Loss')
    plt.savefig(dir_path + 'loss_dotsplot.png')

    metric_df = pd.DataFrame(metric_dict)
    with open(dir_path + r'metric', 'wb') as f:
        feather.write_feather(metric_df, f)

    check_point_index = int(1e6)
    metric_df.head(check_point_index)['loss'].describe().to_csv(dir_path + 'head_loss_describe_table.csv')
    metric_df.tail(check_point_index)['loss'].describe().to_csv(dir_path + 'tail_loss_describe_table.csv')

    rolling_window_size = 10000
    rolling_df = metric_df.loc[:, ['loss', 'dm_outcome', 'bayesian_outcome', 'dr_outcome']].rolling(rolling_window_size, center=True).mean()

    fig,ax1 = plt.subplots(figsize=(15,4))
    ax2 = ax1.twinx()
    ax1.plot(rolling_df['loss'], label='Loss')
    ax2.plot(rolling_df['dm_outcome'],'g', label='Decision Markets outcome')
    ax2.plot(rolling_df['bayesian_outcome'], 'r', label='Bayesian outcome')
    # ax2.plot(rolling_df['dr_outcome'], 'b.',markersize=0.01)
    ax2.set_ylim([0.5,0.75])
    ax1.legend(loc='lower left', bbox_to_anchor=(0,1))
    ax2.legend(loc='lower right', bbox_to_anchor=(1,1))
    xlabels = ax1.get_xticks().tolist()
    xlabels = [str(xlabel)[0] for xlabel in xlabels]
    ax1.set_xticklabels(xlabels)
    ax1.set_xlabel('Millon Steps')
    fig.suptitle('Rolling Window Metrics with Window Size ' + str(rolling_window_size), y=1.3)
    plt.savefig(dir_path + 'loss_reward_plot.png')

    feature_num = 3

    fig, axs=plt.subplots(agent_num, action_num, figsize=(15, 5 * agent_num), sharex=True, sharey=True, squeeze=False)


    for agent, ag_no in zip(agent_list, range(agent_num)):
        mean_weights_df_list = agent.mean_weights_history_df()
        reward_df = agent.reward_history_dataframe()
        with open(dir_path + f'{agent.name}_score', 'wb') as f:
            feather.write_feather(reward_df, f)
        # Save the weights
        for no in range(2):
            with open(dir_path + f'{agent.name}_bucket{no}_weights', 'wb') as f:
                feather.write_feather(mean_weights_df_list[no], f)

        for df, ac_no in zip(mean_weights_df_list, range(2)):

            axs[ag_no, ac_no].plot(df.iloc[1:, 0 * feature_num + 0],
                                   label='Bucket0 Red weight', color='red')
            axs[ag_no, ac_no].plot(df.iloc[1:, 0 * feature_num + 1],
                                   label='Bucket0 Blue weight', color='blue')
            axs[ag_no, ac_no].plot(df.iloc[1:, 0 * feature_num + 2],
                                   label='Bucket0 Prior weight', color='green')

            axs[ag_no, ac_no].plot(df.iloc[1:, 1 * feature_num + 0],
                                   label='Bucket1 Red weight', color='darkred')
            axs[ag_no, ac_no].plot(df.iloc[1:, 1 * feature_num + 1],
                                   label='Bucket1 Blue weight', color='darkblue')
            axs[ag_no, ac_no].plot(df.iloc[1:, 1 * feature_num + 2],
                                   label='Bucket1 Prior weight', color='darkgreen')

    for r_no in range(agent_num):
        axs[r_no, 0].set_ylabel(f'Agent {r_no}')
        for c_no in range(action_num):
            if r_no == 0:
                axs[r_no, c_no].set_title(f'Bucket {c_no}')
            axs[r_no, c_no].hlines(y=np.log(pr_red_ball_red_bucket / pr_red_ball_blue_bucket), xmin=0,

                                xmax=len(df), colors='red',
                                         linestyles='dashdot')
            axs[r_no, c_no].hlines(
                    y=np.log((1 - pr_red_ball_red_bucket) / (1 - pr_red_ball_blue_bucket)), xmin=0,
                    xmax=len(df), colors='blue',
                    linestyles='dashdot')
            axs[r_no, c_no].hlines(y=1, xmin=0, xmax=len(df), colors='green', linestyles='dashdot')
            axs[r_no, c_no].hlines(y=0, xmin=0, xmax=len(df), colors='black', linestyles='dashdot')

    xlabels = axs[-1, 0].get_xticks().tolist()
    xlabels = [str(xlabel)[0] for xlabel in xlabels]
    axs[-1,0].set_xticklabels(xlabels)


    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.95, 0.95), ncol=1)
    fig.text(0.5, 0.08, 'Million Steps', ha='center', va='center', fontsize=25)
    # fig.text(0.75, 1, 'Bucket 1', ha='center', va='center', fontsize=20)


    plt.savefig(dir_path + 'mean_weights.png', bbox_inches='tight')



