import numpy as np
from Environment import *
from scipy.special import logit, expit
import traceback
from tqdm.notebook import tnrange
from tqdm import trange
from PolicyGradientAgent import StochasticGradientAgent, DeterministicGradientAgent
# import line_profiler


def stochastic_training_notebook(agent_list, learning_rate_theta, learning_rate_wv,
                                 memory_size, batch_size, training_episodes,
                                 decay_rate, beta1, beta2, algorithm, learning_std,
                                 fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                 prior_red_list, agent_num, action_num, score_func, decision_rule,
                                 preferred_colour_pr_list, evaluation_step, weights_init, report_order, signal_size_list):

    assert len(signal_size_list) == agent_num, "The length of signal_size_list should equal to the number of agents."

    if not agent_list:
        for i in range(agent_num):
            agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                            learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                            memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                            learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                            algorithm=algorithm, weights_init=weights_init)
            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

    loss_list = []
    dm_outcome_list = []
    prior_outcome_list = []
    nb_outcome_list = []

    for t in tnrange(training_episodes):
        if report_order == ReportOrder.RANDOM:
            np.random.shuffle(agent_list)
        dm_outcome, prior_outcome, nb_outcome, loss = stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket,
                                                    pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func,
                                                    decision_rule, preferred_colour_pr_list, signal_size_list)
        dm_outcome_list.append(dm_outcome)
        prior_outcome_list.append(prior_outcome)
        nb_outcome_list.append(nb_outcome)
        loss_list.append(loss)

    return agent_list, dm_outcome_list, prior_outcome_list, nb_outcome_list, loss_list


def stochastic_training(learning_rate_theta, learning_rate_wv,
                        memory_size, batch_size, training_episodes,
                        decay_rate, beta1, beta2, algorithm, learning_std,
                        fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                        prior_red_list, agent_num, action_num, score_func, decision_rule,
                        preferred_colour_pr_list, evaluation_step, weight_init, report_order, signal_size_list):
    agent_list = []

    assert len(signal_size_list) == action_num, "The length of signal_size_list should equal to the number of agents."

    for i in range(agent_num):
        agent = StochasticGradientAgent(feature_num=3, action_num=action_num,
                                        learning_rate_theta=learning_rate_theta, learning_rate_wv=learning_rate_wv,
                                        memory_size=memory_size, batch_size=batch_size, beta1=beta1, beta2=beta2,
                                        learning_std=learning_std, fixed_std=fixed_std, name='agent' + str(i),
                                        algorithm=algorithm, weights_init=weight_init)
        agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
        agent_list.append(agent)

    loss_list = []
    dm_outcome_list = []
    prior_outcome_list = []
    nb_outcome_list = []

    for t in trange(training_episodes):
        if report_order == ReportOrder.RANDOM:
            np.random.shuffle(agent_list)
        dm_outcome, prior_outcome, nb_outcome, loss = stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket,
                                                    pr_red_ball_blue_bucket, agent_list, t, decay_rate, score_func,
                                                    decision_rule, preferred_colour_pr_list, signal_size_list)
        dm_outcome_list.append(dm_outcome)
        prior_outcome_list.append(prior_outcome)
        nb_outcome_list.append(nb_outcome)
        loss_list.append(loss)


    return agent_list, dm_outcome_list, prior_outcome_list, loss_list


def stochastic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket, agent_list,
                                t, decay_rate, score_func, decision_rule, preferred_colour_pr_list, signal_size_list):
    if prior_red_list is None:
        # prior_red_instances = np.random.uniform(size=action_num)
        prior_red_instances = np.random.normal(loc=0, scale=0.3, size=action_num)
    else:
        prior_red_instances = np.random.choice(prior_red_list, size=action_num)

    buckets = MultiBuckets(action_num, expit(prior_red_instances), pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    dm = DecisionMarket(action_num, prior_red_instances, decision_rule, preferred_colour=BucketColour.RED,
                        preferred_colour_pr_list=preferred_colour_pr_list)

    experience_list = []

    logit_pos = dm.read_current_pred()
    for agent, signal_size in zip(agent_list, signal_size_list):
        signal_mat = buckets.signal(signal_size, t)
        current_predictions = dm.read_current_pred()
        prior_index = np.arange(start=2, stop=3 * action_num, step=3)
        signal_mat[0, prior_index] = current_predictions
        signal_array, h_array, mean_array, std_array = agent.report(signal_mat, t)
        dm.report(h_array, mean_array)
        experience_list.append([t, signal_array.copy(), h_array.copy(), mean_array.copy(), std_array.copy()])
        signal_mat[0, prior_index] = logit_pos
        logit_pos = BayesianUpdateMat(signal_mat, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)

    rewards_array, arm = dm.resolve(score_func, buckets.bucket_list)

    # learning
    for agent, reward_array, experience in zip(agent_list, rewards_array, experience_list):

        experience.append(reward_array)
        agent.store_experience(*experience)
        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)

    final_prediction = expit(dm.read_current_pred())
    loss = np.sum(np.square(expit(logit_pos) - final_prediction))
    dm_outcome = buckets.bucket_list[arm].colour == BucketColour.RED
    prior_arm = np.argmax(prior_red_instances)
    prior_outcome = buckets.bucket_list[prior_arm].colour == BucketColour.RED
    nb_arm = np.argmax(expit(logit_pos))
    nb_outcome = buckets.bucket_list[nb_arm].colour == BucketColour.RED

    return dm_outcome, prior_outcome, nb_outcome, loss


def deterministic_training_notebook(
        agent_list, feature_num, action_num,
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning,
        fixed_std, score_func, decision_rule, preferred_colour_pr_list,
        evaluation_step, weights_init, report_order, signal_size_list):

    if not agent_list:
        for i in range(agent_num):
            agent = DeterministicGradientAgent(
                feature_num=feature_num, action_num=action_num,
                learning_rate_theta=learning_rate_theta,
                learning_rate_wv=learning_rate_wv,
                learning_rate_wq=learning_rate_wq,
                memory_size=memory_size,
                batch_size=batch_size,
                beta1=beta1,
                beta2=beta2,
                name='agent' + str(i),
                algorithm=algorithm,
                weights_init=weights_init
            )
            agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
            agent_list.append(agent)

    explorer = Explorer(feature_num=3, action_num=action_num, learning=explorer_learning, init_learning_rate=3e-4,
                        min_std=0.1)

    loss_list = []
    dm_outcome_list = []
    prior_outcome_list = []
    nb_outcome_list = []

    for t in tnrange(training_episodes):
        if report_order == ReportOrder.RANDOM:
            np.random.shuffle(agent_list)
        dm_outcome, prior_outcome, nb_outcome, loss = deterministic_iterative_policy(
            action_num, prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std, score_func, decision_rule,
            preferred_colour_pr_list, signal_size_list
        )
        dm_outcome_list.append(dm_outcome)
        prior_outcome_list.append(prior_outcome)
        nb_outcome_list.append(nb_outcome)
        loss_list.append(loss)

    return agent_list, dm_outcome_list, prior_outcome_list, nb_outcome_list, loss_list


def deterministic_training(
        feature_num, action_num,
        learning_rate_theta, learning_rate_wv, learning_rate_wq,
        memory_size, batch_size, training_episodes,
        decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
        pr_red_ball_blue_bucket, prior_red_list, agent_num, explorer_learning,
        fixed_std, score_func, decision_rule, preferred_colour_pr_list,
        evaluation_step, weight_init, report_order, signal_size_list):
    agent_list = []

    for i in range(agent_num):
        agent = DeterministicGradientAgent(
            feature_num=feature_num, action_num=action_num, learning_rate_theta=learning_rate_theta,
            learning_rate_wv=learning_rate_wv, learning_rate_wq=learning_rate_wq,
            memory_size=memory_size, batch_size=batch_size,
            beta1=beta1, beta2=beta2, name='agent' + str(i),
            algorithm=algorithm, weights_init=weight_init
        )
        agent.evaluation_init(pr_red_ball_red_bucket, pr_red_ball_blue_bucket, evaluation_step)
        agent_list.append(agent)

    loss_list = []
    dm_outcome_list = []
    prior_outcome_list = []
    nb_outcome_list = []

    explorer = Explorer(feature_num=3, action_num=action_num, learning=explorer_learning, init_learning_rate=3e-4,
                        min_std=0.1)

    for t in trange(training_episodes):
        if report_order == ReportOrder.RANDOM:
            np.random.shuffle(agent_list)
        dm_outcome, prior_outcome, nb_outcome, loss = deterministic_iterative_policy(
            action_num, prior_red_list, pr_red_ball_red_bucket,
            pr_red_ball_blue_bucket, agent_list, explorer,
            t, decay_rate, fixed_std, score_func, decision_rule, preferred_colour_pr_list, signal_size_list
        )
        dm_outcome_list.append(dm_outcome)
        prior_outcome_list.append(prior_outcome)
        nb_outcome_list.append(nb_outcome)
        loss_list.append(loss)

    return agent_list, dm_outcome_list, prior_outcome_list, loss_list

def deterministic_iterative_policy(action_num, prior_red_list, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
                                   agent_list, explorer, t, decay_rate, fixed_std, score_func, decision_rule,
                                   preferred_colour_pr_list, signal_size_list):
    if prior_red_list is None:
        # prior_red_instances = np.random.uniform(size=action_num)
        prior_red_instances = np.random.normal(loc=0, scale=0.3, size=action_num)
    else:
        prior_red_instances = np.random.choice(prior_red_list, size=action_num)
    # Prepare a bucket and a prediction market
    buckets = MultiBuckets(action_num, expit(prior_red_instances), pr_red_ball_red_bucket, pr_red_ball_blue_bucket)
    dm = DecisionMarket(action_num, prior_red_instances, decision_rule, preferred_colour=BucketColour.RED,
                        preferred_colour_pr_list=preferred_colour_pr_list)

    experience_list = []

    # nb_predictions = dm.read_current_pred()
    logit_pos = dm.read_current_pred()
    for agent, signal_size in zip(agent_list, signal_size_list):
        signal_mat = buckets.signal(signal_size, t)
        current_predictions = dm.read_current_pred()
        prior_index = np.arange(start=2, stop=3 * action_num, step=3)
        signal_mat[0, prior_index] = current_predictions
        signal_array, mean_array = agent.report(signal_mat, t)
        explorer.set_parameters(mean_array=mean_array, fixed_std=fixed_std)
        e_h_array = explorer.report(signal_array)
        dm.report(e_h_array, mean_array)
        experience_list.append([t, signal_array, e_h_array, mean_array])
        signal_mat[0, prior_index] = logit_pos
        logit_pos = BayesianUpdateMat(signal_mat, pr_red_ball_red_bucket, pr_red_ball_blue_bucket)

    rewards_array, arm = dm.resolve(score_func, buckets.bucket_list)

    # learning
    for agent, reward_array, experience in zip(agent_list, rewards_array, experience_list):

        experience.append(reward_array)
        agent.store_experience(*experience)
        # explorer.update(reward_array, signal_array)

        try:
            agent.batch_update(t)
        except AssertionError:
            tb = traceback.format_exc()
            print(tb)

        agent.learning_rate_decay(epoch=t, decay_rate=decay_rate)
        #     if explorer.learning:
        #         explorer.learning_rate_decay(epoch=t, decay_rate=0.001)

    final_predictions = expit(dm.read_current_pred())
    loss = np.sum(np.square(expit(logit_pos) - final_predictions))
    dm_outcome = buckets.bucket_list[arm].colour == BucketColour.RED
    prior_arm = np.argmax(prior_red_instances)
    prior_outcome = buckets.bucket_list[prior_arm].colour == BucketColour.RED
    nb_arm = np.argmax(expit(logit_pos))
    nb_outcome = buckets.bucket_list[nb_arm].colour == BucketColour.RED

    return dm_outcome, prior_outcome, nb_outcome, loss


if __name__ == '__main__':
    # learning_rate_theta = 1e-4
    # learning_rate_wv = 1e-4
    # memory_size = 16
    # batch_size = 16
    # training_episodes = int(1e6)
    # decay_rate = 0
    # beta1 = 0.9
    # beta2 = 0.9999
    # # Algorithm: adam, momentum, regular
    # algorithm = Algorithm.REGULAR
    # learning_std = False
    # fixed_std = 0.3
    # # Bucket parameters
    # pr_red_ball_red_bucket = 2 / 3
    # pr_red_ball_blue_bucket = 1 / 3
    # prior_red_list = logit([3 / 4, 1 / 4])
    # agent_num = 2
    # action_num = 2
    # preferred_colour_pr_list = [0.9, 0.1]
    # score_func = ScoreFunction.LOG
    # decision_rule = DecisionRule.DETERMINISTIC
    # evaluation_step = 1
    # weights_init = WeightsInit.ZERO
    # report_order = ReportOrder.RANDOM
    # signal_size_list = [1, 1]
    #
    # stochastic_training(learning_rate_theta, learning_rate_wv,
    #                     memory_size, batch_size, training_episodes,
    #                     decay_rate, beta1, beta2, algorithm, learning_std,
    #                     fixed_std, pr_red_ball_red_bucket, pr_red_ball_blue_bucket,
    #                     prior_red_list, agent_num, action_num, score_func, decision_rule,
    #                     preferred_colour_pr_list, evaluation_step, weights_init, report_order, signal_size_list)

    feature_num = 3
    action_num = 2
    learning_rate_theta = 1e-4
    decay_rate = 0  # 0.001
    learning_rate_wv = 1e-4
    learning_rate_wq = 1e-2
    memory_size = 16
    batch_size = 16
    training_episodes = 900000
    beta1 = 0.9
    beta2 = 0.9999
    fixed_std = 0.3
    # Algorithm: adam, momentum, regular
    algorithm = Algorithm.REGULAR
    # Bucket parameters
    prior_red_list = logit([3 / 4, 1 / 4])
    pr_red_ball_red_bucket = 2 / 3
    pr_red_ball_blue_bucket = 1 / 3
    agent_num = 1

    explorer_learning = False
    decision_rule = DecisionRule.STOCHASTIC
    score_func = ScoreFunction.LOG
    preferred_colour_pr_list = [0.8, 0.2]
    weights_init = WeightsInit.ZERO
    report_order = ReportOrder.RANDOM
    signal_size_list = [2]
    evaluation_step = 1

    agent_list = deterministic_training(     feature_num, action_num, learning_rate_theta, learning_rate_wv, learning_rate_wq,
                                             memory_size, batch_size, training_episodes,
                                             decay_rate, beta1, beta2, algorithm, pr_red_ball_red_bucket,
                                             pr_red_ball_blue_bucket, prior_red_list, agent_num,
                                             explorer_learning, fixed_std, score_func, decision_rule, preferred_colour_pr_list, evaluation_step, weights_init, report_order, signal_size_list)


