import numpy as np
import copy
from const import const_value as const


class Rnn:
    def __init__(self, Num_input, Num_hidden, Num_output, Learning_rate, Discount_rate):
        self.num_input = Num_input
        self.num_hidden = Num_hidden
        self.num_output = Num_output
        self.learning_rate = Learning_rate
        self.discount_rate = Discount_rate

        # 閾値の部分を抜いたバージョン
        self.hidden_weight = -1.0 * np.random.rand(self.num_hidden,
                                                   self.num_input + self.num_hidden + self.num_output) + 0.5
        self.output_weight = -1.0 * np.random.rand(self.num_output, self.num_hidden) + 0.5

        # 行列として扱うために2次元にしている
        self.hidden_input = np.zeros((1, self.num_hidden + self.num_input))
        self.hidden_net = np.zeros((1, self.num_hidden))
        self.hidden_out = np.zeros((1, self.num_hidden))
        self.out_net = np.zeros((1, self.num_output))
        self.out_output = np.zeros((1, self.num_output))

    def forward(self, Input):
        input_data = Input
        self.hidden_input = input_data
        h_net = np.dot(input_data, self.hidden_weight.T)
        self.hidden_net = h_net
        h_out = np.tanh(h_net)
        self.hidden_out = h_out

        o_net = np.dot(h_out, self.output_weight.T)
        self.out_net = o_net
        o_out = np.tanh(o_net)
        self.out_output = o_out

        # 出力層の出力
        return o_out

    # バッチ学習のために中間層の出力のみ保存してる
    def hidden_save_forward(self, Input):
        input_data = Input
        self.hidden_input = input_data
        h_net = np.dot(input_data, self.hidden_weight.T)
        h_out = np.tanh(h_net)
        self.hidden_out = h_out

        # 閾値の部分を追加
        o_net = np.dot(h_out, self.output_weight.T)
        o_out = np.tanh(o_net)
        # 教師データを作るために使う
        self.out_output = o_out

        return o_out

    def no_save_forward(self, Input):
        input_data = Input
        self.hidden_input = input_data
        h_net = np.dot(input_data, self.hidden_weight.T)
        h_out = np.tanh(h_net)
        o_net = np.dot(h_out, self.output_weight.T)
        o_out = np.tanh(o_net)

        return o_out

    # 誤差逆伝搬法を使ってる
    # 活性化関数はtanh()
    def backward(self, Teach_data):
        tanh_o = (1. - self.out_output ** 2).T
        tanh_h = (1. - self.hidden_out ** 2).T

        error_output = (self.out_output - Teach_data).T
        error_hidden = np.dot(self.output_weight.T, error_output)

        self.output_weight -= self.learning_rate * np.dot((error_output * tanh_o), self.hidden_out)
        self.hidden_weight -= self.learning_rate * np.dot((error_hidden * tanh_h), self.hidden_input)

    def make_teach_data(self, Next_input, decide_act, reward):
        next_q_values = self.no_save_forward(Next_input)  # 次の環境状態を入力してQ値を取得する
        arg_next_q_values = np.where(next_q_values == next_q_values.max(axis=1))  # 次の環境状態の中の最大値のインデックスを全て取得する
        rand_arg_max_q = np.random.randint(0, len(arg_next_q_values[0]))  # 最大なQ値が複数ある場合，ランダムで選択する
        arg_max_q = arg_next_q_values[1][rand_arg_max_q]  # ランダムな値を元に最大のQ値のインデックスを一つ決める

        teach_data = copy.deepcopy(self.out_output)
        # 教師データのうち，実際にとった行動に対応する値のみQ学習の更新式で更新する(学習率aがないため，この式になる)
        # -1 <= tanh(x) <= 1であるため，それに収まるように正規化
        teach_data[0][decide_act] = (reward + self.discount_rate * next_q_values[0][arg_max_q]) / (
                    const.MAX_REWARD_ABS + self.discount_rate)

        # teach_data[0][decide_act] = reward + self.discount_rate * next_q_values[0][arg_max_q]

        # teach_data[0][decide_act] = reward + self.discount_rate * next_q_values[0][arg_max_q]
        # print("q_teach_data=", teach_data)
        # print(teach_data)
        return teach_data

    def make_input_data(self, Env_input):
        flatten_input = np.array(Env_input).reshape((1, -1))
        concat_input = np.concatenate([flatten_input, self.hidden_out], 1)
        concat_output = np.concatenate([concat_input, self.out_output], 1)

        return concat_output
