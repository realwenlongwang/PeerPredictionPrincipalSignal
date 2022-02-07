import numpy as np
from scipy.special import expit, logit
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from numpy.random import Generator, PCG64


class Ball(Enum):
    RED = 0
    BLUE = 1


class BucketColour(Enum):
    RED = 0
    BLUE = 1


class Algorithm(Enum):
    REGULAR = 'regular'
    MOMENTUM = 'momentum'
    ADAM = 'adam'


class ScoreFunction(Enum):
    LOG = 0
    QUADRATIC = 1


class DecisionRule(Enum):
    STOCHASTIC = 0
    DETERMINISTIC = 1

class WeightsInit(Enum):
    ZERO = 'zero initialisation'
    RANDOM = 'random initialisation'



def NaiveBayesOneIter(prior, signal, bucket_no, pr_rs_ru, pr_rs_bu):
    posterior = np.array(prior)
    if signal.name == 'RED':
        posterior[bucket_no] = prior[bucket_no] * pr_rs_ru / (
                    prior[bucket_no] * pr_rs_ru + (1 - prior[bucket_no]) * pr_rs_bu)
    else:
        posterior[bucket_no] = prior[bucket_no] * (1 - pr_rs_ru) / (
                    prior[bucket_no] * (1 - pr_rs_ru) + (1 - prior[bucket_no]) * (1 - pr_rs_bu))

    return posterior

class PredictionMarket:

    def __init__(self, no, prior_red):
        self.no = no
        self.current_prediction = [prior_red, 1 - prior_red]
        self.sampled_prediction_history = [self.current_prediction.copy()]
        self.mean_prediction_history = [self.current_prediction.copy()]


    def report(self, sampled_prediction, mean_prediction):
        # Record the contract if multiple traders.
        self.sampled_prediction_history.append(sampled_prediction.copy())
        self.mean_prediction_history.append(mean_prediction.copy())
        self.current_prediction = mean_prediction.copy()

    def log_resolve(self, materialised_index):
        score_list = []
        for i in range(len(self.sampled_prediction_history) - 1): # minus the initial report
            scores = np.log(self.sampled_prediction_history[i + 1]) - np.log(self.mean_prediction_history[i])
            score_list.append(scores.copy()[materialised_index])
        return np.array(score_list)

    def brier_resolve(self, materialised_index):
        score_list = []
        for i in range(len(self.sampled_prediction_history) - 1):
            current_scores = self.sampled_prediction_history[i + 1][materialised_index] - np.sum(np.square(self.sampled_prediction_history[i + 1])) / 2
            previous_scores = self.mean_prediction_history[i][materialised_index] - np.sum(np.square(self.mean_prediction_history[i])) / 2
            score_list.append(current_scores - previous_scores)
        return np.array(score_list)

    def resolve(self, score_func, materialised_index):
        if score_func == ScoreFunction.LOG:
            return self.log_resolve(materialised_index)
        elif score_func == ScoreFunction.QUADRATIC:
            return self.brier_resolve(materialised_index)
        else:
            raise ValueError('The score function does not exist.')


class DecisionMarket:

    def __init__(self, action_num, prior_red_instances, decision_rule, preferred_colour, preferred_colour_pr_list):
        self.conditional_market_num = action_num
        self.conditional_market_list = list(PredictionMarket(no, prior_red) for no, prior_red in
                                            zip(range(self.conditional_market_num), prior_red_instances))
        self.preferred_colour = preferred_colour
        self.decision_rule = decision_rule
        self.preferred_colour_pr_list = preferred_colour_pr_list

    def report(self, sampled_predictions, mean_predictions):
        for pm, pi, mu in zip(self.conditional_market_list, sampled_predictions[0], mean_predictions[0]):
            pm.report([np.asscalar(pi), 1 - np.asscalar(pi)], [np.asscalar(mu), 1 - np.asscalar(mu)])

    def log_resolve(self, buckets):
        if self.decision_rule == DecisionRule.DETERMINISTIC or self.conditional_market_num == 1:
            current_price_list = self.read_current_pred()
            index = np.argmax(current_price_list)
            conditional_market, bucket = self.conditional_market_list[index], buckets[index]
            agent_num = len(conditional_market.sampled_prediction_history) - 1  # minus the initial report
            reward_array = np.zeros(shape=(agent_num, self.conditional_market_num))
            reward_array[:, index] = conditional_market.log_resolve(bucket.colour.value)
            return reward_array, index
        else:
            # TODO: consider move the generator initialisation outside and init once only. May speed up training.
            generator = Generator(PCG64())
            sorted_market_list = sorted(self.conditional_market_list, key=lambda market: market.current_prediction[0],
                                        reverse=True)
            pr, conditional_market = generator.choice(list(zip(self.preferred_colour_pr_list, sorted_market_list)),
                                                      p=self.preferred_colour_pr_list)
            index = conditional_market.no
            bucket = buckets[index]
            agent_num = len(conditional_market.sampled_prediction_history) - 1 # minus the initial report
            reward_array = np.zeros(shape=(agent_num, self.conditional_market_num))
            reward_array[:, index] = conditional_market.log_resolve(bucket.colour.value) / pr
            return reward_array, index

    def brier_resolve(self, buckets):
        if self.decision_rule == DecisionRule.DETERMINISTIC  or self.conditional_market_num == 1:
            current_price_list = self.read_current_pred()
            index = np.argmax(current_price_list)
            conditional_market, bucket = self.conditional_market_list[index], buckets[index]
            agent_num = len(conditional_market.sampled_prediction_history) - 1  # minus the initial report
            reward_array = np.zeros(shape=(agent_num, self.conditional_market_num))
            reward_array[:, index] = conditional_market.brier_resolve(bucket.colour.value)
            return reward_array, index
        else:
            # TODO: consider move the generator initialisation outside and init once only. May speed up training.
            generator = Generator(PCG64())
            sorted_market_list = sorted(self.conditional_market_list, key=lambda market: market.current_prediction[0],
                                        reverse=True)
            pr, conditional_market = generator.choice(list(zip(self.preferred_colour_pr_list, sorted_market_list)),
                                                      p=self.preferred_colour_pr_list)
            index = conditional_market.no
            bucket = buckets[index]
            agent_num = len(conditional_market.sampled_prediction_history) - 1 # minus the initial report
            reward_array = np.zeros(shape=(agent_num, self.conditional_market_num))
            reward_array[:, index] = conditional_market.brier_resolve(bucket.colour.value) / pr
            return reward_array, index

    def resolve(self, score_func, buckets):
        if score_func == ScoreFunction.LOG:
            return self.log_resolve(buckets)
        elif score_func == ScoreFunction.QUADRATIC:
            return self.brier_resolve(buckets)
        else:
            raise ValueError('The score function does not exist.')



    def read_current_pred(self):
        current_price_list = list(pm.current_prediction[0] for pm in self.conditional_market_list)

        return current_price_list


class Bucket:
    def __init__(self, no, prior_red, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
        assert prior_red >= 0, 'Prior can not be negative!'
        assert prior_red <= 1, 'Prior can not greater than one!'
        assert pr_red_ball_red_bucket >= 0, 'Prior can not be negative!'
        assert pr_red_ball_red_bucket <= 1, 'Prior can not greater than one!'
        assert pr_red_ball_blue_bucket >= 0, 'Prior can not be negative!'
        assert pr_red_ball_blue_bucket <= 1, 'Prior can not greater than one!'

        self.no = no
        self.prior_red = prior_red
        self.pr_red_ball_red_bucket = pr_red_ball_red_bucket
        self.pr_red_ball_blue_bucket = pr_red_ball_blue_bucket
        self.colour = np.random.choice([BucketColour.RED, BucketColour.BLUE], p=(self.prior_red, 1 - self.prior_red))

    def signal(self):
        if self.colour == BucketColour.RED:
            ball_distribution = (self.pr_red_ball_red_bucket, 1 - self.pr_red_ball_red_bucket)
        elif self.colour == BucketColour.BLUE:
            ball_distribution = (self.pr_red_ball_blue_bucket, 1 - self.pr_red_ball_blue_bucket)
        else:
            raise ValueError('Bucket colour incorrect, colour is ' + str(self.colour.name))
        return np.random.choice([Ball.RED, Ball.BLUE], p=ball_distribution)


class MultiBuckets:
    def __init__(self, bucket_num, prior_red_instances, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
        self.pr_red_ball_red_bucket = pr_red_ball_red_bucket
        self.pr_red_ball_blue_bucket = pr_red_ball_blue_bucket
        self.bucket_list = []
        self.index = np.random.randint(bucket_num)
        self.bucket_num = bucket_num
        for no, prior_red in zip(range(bucket_num), prior_red_instances):
            self.bucket_list.append(Bucket(no, prior_red, pr_red_ball_red_bucket, pr_red_ball_blue_bucket))

    def signal(self, t):

        # Randomly select a bucket
        bucket = np.random.choice(self.bucket_list)

        # # Always select a different bucket for each agent
        # bucket = self.bucket_list[self.index % self.bucket_num]
        # self.index += 1
        return bucket.no, bucket.signal()


class Explorer:
    def __init__(self, feature_num, action_num, learning=True, init_learning_rate=0.001, min_std=0.3):
        self.action_num = action_num
        self.mean_array = np.zeros(shape=(1, action_num))
        self.std_array = np.ones(shape=(1, action_num))
        self.theta_std = np.zeros((feature_num * action_num, action_num))
        self.init_learning_rate = init_learning_rate
        self.learning_rate = init_learning_rate
        self.learning = learning
        self.min_std = min_std
        self.h_array = list()

    def set_parameters(self, mean_array, fixed_std):
        self.mean_array = mean_array.copy()
        self.std_array = np.ones(shape=(1, self.action_num)) * fixed_std

    def learning_rate_decay(self, epoch, decay_rate):
        self.learning_rate = 1 / (1 + decay_rate * epoch) * self.init_learning_rate
        return self.learning_rate

    def report(self, signal_array):
        if self.learning:
            self.std_array = np.exp(np.matmul(signal_array, self.theta_std))
            if self.std_array < self.min_std:
                self.std_array = self.min_std
        self.h_array = np.random.normal(loc=self.mean_array, scale=self.std_array)
        return self.h_array.copy()

    def update(self, reward, signal_array):
        if self.learning:
            gradient_std = np.matmul(signal_array, reward * (
                        np.power(self.h_array - self.mean_array, 2) / np.power(self.std_array, 2) - 1))
            self.theta_std += self.learning_rate * gradient_std


def analytical_best_report_ru_rs(pr_ru, pr_rs_ru, pr_rs_bu):
    """
    :param pr_ru: float
        Prior probability of red urn
    :param pr_rs_ru: float
        Conditional probability of red ball signal_array given red urn
    :param pr_rs_bu: float
        conditional probability of red ball signal_array given blue urn
    :return: float
        Conditional probability of red urn given red ball signal_array
    """
    joint_distribution_ru_rs = pr_ru * pr_rs_ru
    joint_distribution_bu_rs = (1 - pr_ru) * pr_rs_bu
    return joint_distribution_ru_rs / (joint_distribution_ru_rs + joint_distribution_bu_rs)


def analytical_best_report_ru_bs(pr_ru, pr_bs_ru, pr_bs_bu):
    """
    :param pr_ru: float
        Prior probability of red urn
    :param pr_bs_ru: float
        Conditional probability of blue ball signal_array given red urn
    :param pr_bs_bu: float
        conditional probability of blue ball signal_array given blue urn
    :return: float
        Conditional probability of red urn given blue ball signal_array
    """

    joint_distribution_ru_bs = pr_ru * pr_bs_ru
    joint_distribution_bu_bs = (1 - pr_ru) * pr_bs_bu
    return joint_distribution_ru_bs / (joint_distribution_ru_bs + joint_distribution_bu_bs)


def expected_log_reward_red_ball(actual_pr_ru_rs, estimated_pr_ru_rs, pr_ru):
    """
    This function compute the expected logarithmic reward_array given a red ball signal_array
    :param actual_pr_ru_rs: float
        Ground truth probability of conditional probability of red urn given a red ball signal_array
    :param estimated_pr_ru_rs: float
        Estimated probability of conditional probability of red urn given a red ball signal_array
    :param pr_ru: float
        Prior probability of a red urn
    :return: float
        expected logarithmic reward_array given red signal_array
    """
    return actual_pr_ru_rs * (np.log(estimated_pr_ru_rs) - np.log(pr_ru)) + (1 - actual_pr_ru_rs) * (
            np.log(1 - estimated_pr_ru_rs) - np.log(1 - pr_ru))


def expected_quadratic_reward_red_ball(actual_pr_ru_rs, estimated_pr_ru_rs, pr_ru):
    """
    This function compute the expected logarithmic reward_array given a red ball signal_array
    :param actual_pr_ru_rs: float
        Ground truth probability of conditional probability of red urn given a red ball signal_array
    :param estimated_pr_ru_rs: float
        Estimated probability of conditional probability of red urn given a red ball signal_array
    :param pr_ru: float
        Prior probability of a red urn
    :return: float
        expected logarithmic reward_array given red signal_array
    """
    return actual_pr_ru_rs * (np.log(estimated_pr_ru_rs) - np.log(pr_ru)) + (1 - actual_pr_ru_rs) * (
            np.log(1 - estimated_pr_ru_rs) - np.log(1 - pr_ru))


def expected_log_reward_blue_ball(actual_pr_ru_bs, estimated_pr_ru_bs, pr_ru):
    """
    This function compute the expected logarithmic reward_array given a blue ball signal_array
    :param actual_pr_ru_bs: float
        Ground truth probability of conditional probability of red urn given a blue ball signal_array
    :param estimated_pr_ru_bs: float
        Estimated probability of conditional probability of red urn given a blue ball signal_array
    :param pr_ru: float
        Prior probability of a red urn
    :return: float
        expected logarithmic reward_array given red signal_array
    """
    return actual_pr_ru_bs * (np.log(estimated_pr_ru_bs) - np.log(pr_ru)) + (1 - actual_pr_ru_bs) * (
            np.log(1 - estimated_pr_ru_bs) - np.log(1 - pr_ru))


# TODO: How to compute the regret for second agent?
# def compute_regret(signal_array, pi, prior_red, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
#     if signal_array == 'red':
#         actual_pr_ru_S = analytical_best_report_ru_rs(pr_ru=prior_red, pr_rs_ru=pr_red_ball_red_bucket,
#                                                       pr_rs_bu=pr_red_ball_blue_bucket)
#         expected_log_reward = expected_log_reward_red_ball(actual_pr_ru_rs=actual_pr_ru_S, estimated_pr_ru_rs=pi,
#                                                            pr_ru=prior_red)
#         max_expected_log_reward = expected_log_reward_red_ball(actual_pr_ru_rs=actual_pr_ru_S,
#                                                                estimated_pr_ru_rs=actual_pr_ru_S, pr_ru=prior_red)
#     else:
#         actual_pr_ru_S = analytical_best_report_ru_bs(pr_ru=prior_red, pr_bs_ru=1 - pr_red_ball_red_bucket,
#                                                       pr_bs_bu=1 - pr_red_ball_blue_bucket)
#         expected_log_reward = expected_log_reward_blue_ball(actual_pr_ru_bs=actual_pr_ru_S, estimated_pr_ru_bs=pi,
#                                                             pr_ru=prior_red)
#         max_expected_log_reward = expected_log_reward_blue_ball(actual_pr_ru_bs=actual_pr_ru_S,
#                                                                 estimated_pr_ru_bs=actual_pr_ru_S, pr_ru=prior_red)
#     return max_expected_log_reward - expected_log_reward

def analytical_best_report(bucket_no, ball_colour, current_prediction, pr_red_ball_red_bucket, pr_red_ball_blue_bucket):
    if ball_colour == Ball.RED:
        report = analytical_best_report_ru_rs(current_prediction[bucket_no], pr_red_ball_red_bucket,
                                              pr_red_ball_blue_bucket)
    else:
        report = analytical_best_report_ru_bs(current_prediction[bucket_no], 1 - pr_red_ball_red_bucket,
                                              1 - pr_red_ball_blue_bucket)
    return report


def no_outlier_array(points, thresh=3):
    z = np.abs(stats.zscore(points))
    z = z.reshape(-1)
    return points[z < thresh]


def no_outlier_df(df, thresh=3):
    return df[(np.abs(stats.zscore(df)) < thresh).all(axis=1)]


def signal_encode(bucket_no, ball, action_num, current_prediction):
    encoded_signal = np.zeros(shape=(1, 3 * action_num))
    encoded_signal[0, bucket_no * 3 + ball.value] = 1
    prior_index = np.arange(start=2, stop=3 * action_num, step=3)
    encoded_signal[0, prior_index] = logit(current_prediction)
    return encoded_signal


def one_hot_encode(feature):
    if feature == 'red':
        return [1, 0]
    else:
        return [0, 1]


def one_hot_decode(one_hot_feature):
    if np.array_equal(one_hot_feature, [1, 0]):
        return 'red'
    else:
        return 'blue'


def gradients_box_plot(df, bins, col_name, color, ax):
    _df = df.copy()
    _df['bin'] = pd.cut(_df.index.to_series(), bins=bins, include_lowest=True)

    box_list = []
    for interval in _df['bin'].unique():
        box_list.append(_df.loc[_df['bin'] == interval, col_name].values)
    bplot = ax.boxplot(box_list, patch_artist=True, notch=True, vert=True, meanline=False, zorder=-99, showmeans=True)
    left, right = ax.get_xlim()
    ax.hlines(y=0, xmin=left, xmax=right, linestyles='dashdot', zorder=-98, color='black')
    for patch in bplot['boxes']:
        patch.set_facecolor(color)
    ax.yaxis.grid(True)
    ax.set_xticklabels(labels=_df['bin'].unique(), rotation=15)
    ax.set_title(col_name)


def gradients_box_subplot(df, column_list, colour_list, axs):
    for col_name, ax, colour in zip(column_list, axs, colour_list):
        gradients_box_plot(df, bins=10, col_name=col_name, color=colour, ax=ax)


def dm_expected_log_reward_red_ball(pr_ru1, pr_ru2, pr_rs_ru, pr_rs_bu):
    r1 = np.linspace(start=0.01, stop=0.99, num=50)
    r2 = np.linspace(start=0.01, stop=0.99, num=50)
    r1v, r2v = np.meshgrid(r1, r2)
    actual_pr_ru_rs1 = analytical_best_report_ru_rs(pr_ru=pr_ru1, pr_rs_ru=pr_rs_ru, pr_rs_bu=pr_rs_bu)
    actual_pr_ru_rs2 = analytical_best_report_ru_rs(pr_ru=pr_ru2, pr_rs_ru=pr_rs_ru, pr_rs_bu=pr_rs_bu)
    rv = r1v * (r1v > r2v) + r2v * (r1v <= r2v)
    actual_pr_ru_rsv = actual_pr_ru_rs1 * (r1v > r2v) + pr_ru2 * (r1v <= r2v)
    pr_ruv = pr_ru1 * (r1v > r2v) + pr_ru2 * (r1v <= r2v)
    return r1v, r2v, expected_log_reward_red_ball(actual_pr_ru_rs=actual_pr_ru_rsv, estimated_pr_ru_rs=rv, pr_ru=pr_ruv)


def dm_expected_log_reward_blue_ball(pr_ru1, pr_ru2, pr_bs_ru, pr_bs_bu):
    r1 = np.linspace(start=0.01, stop=0.99, num=50)
    r2 = np.linspace(start=0.01, stop=0.99, num=50)
    r1v, r2v = np.meshgrid(r1, r2)
    actual_pr_ru_bs1 = analytical_best_report_ru_bs(pr_ru=pr_ru1, pr_bs_ru=pr_bs_ru, pr_bs_bu=pr_bs_bu)
    actual_pr_ru_bs2 = analytical_best_report_ru_bs(pr_ru=pr_ru2, pr_bs_ru=pr_bs_ru, pr_bs_bu=pr_bs_bu)
    rv = r1v * (r1v > r2v) + r2v * (r1v <= r2v)
    actual_pr_ru_bsv = actual_pr_ru_bs1 * (r1v > r2v) + pr_ru2 * (r1v <= r2v)
    pr_ruv = pr_ru1 * (r1v > r2v) + pr_ru2 * (r1v <= r2v)
    return r1v, r2v, expected_log_reward_blue_ball(actual_pr_ru_bs=actual_pr_ru_bsv, estimated_pr_ru_bs=rv, pr_ru=pr_ruv)


bucket_colour_to_num = {'red_bucket': 0, 'blue_bucket': 1}
