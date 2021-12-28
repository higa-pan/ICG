import numpy as np
import copy
from .softmax import softmax
# これらはmain.pyが実行される階層からの相対参照になっていることに注意
from const import const_value as const
from rnn import rnn


class Agent:
    def __init__(self, my_agent_number):
        self.my_number = my_agent_number

        # group_member_mapに所属グループごとのメンバーを記録する
        self.group_member_map = {}

        self.who_cooperated = [[0, 0]]
        # 現在,group_member_map[a]が協力したがかwho_cooperated[a]に入っている
        self.who_cooperate = [[0, 0]]
        # どのグループに協力するのかが入る
        self.cooperation_group = 0
        # 選択した行動を表す
        self.select_action = 0
        # ゲームで自身が得た報酬が入る
        self.get_reward = 0.0
        # ゲームで得た報酬の合計が入る
        self.payoff = 0.0
        # リカレントQ実装のための変数
        self.rec_q = rnn.Rnn(const.NUM_INPUT_UNIT, const.NUM_HIDDEN_UNIT, const.NUM_OUTPUT_UNIT,
                             const.LEARNING_RATE_ETA, const.DISCOUNT_RATE_G)
        self.past_input = []
        self.past_teach = []

    def choose_first_cooperate_group(self, rand0_or_1):
        self.cooperation_group = list(self.group_member_map.keys())[rand0_or_1]

    # 所属集団とそのメンバー以外の変数を初期化する
    ## 変更した
    def clear_vars(self, rand_0_or_1, rand_list):
        self.who_cooperated = [[rand_list[0], rand_list[1]]]
        # 現在,join_group[0]のメンバーの誰が協力したかどうかがwho_cooperated[0]に入っている
        self.who_cooperate = [[rand_list[2], rand_list[3]]]
        # cooperation_group_help = [a, b]なら グループaに協力しエージェントbに協力要請したとする
        self.get_reward = 0.0
        self.payoff = 0.0
        self.choose_first_cooperate_group(rand_0_or_1)
        self.rec_q = rnn.Rnn(const.NUM_INPUT_UNIT, const.NUM_HIDDEN_UNIT, const.NUM_OUTPUT_UNIT,
                             const.LEARNING_RATE_ETA, const.DISCOUNT_RATE_G)

    # 所属集団のどちらに協力するのかを決める
    def decide_action(self, rand0_to_1):
        self.past_input.append(copy.deepcopy(self.who_cooperated))  # 入力データの保存

        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1, const.TEMPERATURE, const.NUM_OF_ACTIONS)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group)

        if self.select_action == 1:
            self.cooperation_group = other_groups[0]

    ## 変更した
    def no_save_decide_action(self, rand0_to_1):
        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        # self.past_input.append(copy.deepcopy(rec_input))  # 入力データの保存
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1, const.TEMPERATURE, const.NUM_OF_ACTIONS)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group)

        if self.select_action == 1:
            self.cooperation_group = other_groups[0]

    # 他者が協力したかどうか、自分に協力要請したかどうかを観測する
    def view_other_action(self, other_agent_A, other_agent_B):
        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_A.cooperation_group in list(self.group_member_map):
            self.who_cooperate[0][0] = 1
        else:
            ## 変更した
            # 活性化関数をtanhにしているため-1にしている
            self.who_cooperate[0][0] = -1

        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_B.cooperation_group in list(self.group_member_map):
            self.who_cooperate[0][1] = 1
        else:
            ## 変更した
            # 活性化関数をtanhにしているため-1にしている
            self.who_cooperate[0][1] = -1

    # Qテーブルをview_other_actionをして得た情報を用いて更新する
    def update_decision_weight(self):
        input_data = self.rec_q.make_input_data(self.who_cooperate)
        teach_data = self.rec_q.make_teach_data(input_data, self.select_action, self.get_reward)
        self.rec_q.backward(teach_data)

        self.trans_env()

    def learn_batch(self):
        for i in range(const.MAX_BATCH_TIMES):  # バッチ学習で複数回学習させる
            error = np.zeros((1, const.NUM_OUTPUT_UNIT))
            for input_data, teach_data in zip(self.past_input, self.past_teach):
                # print(input_data, teach_data)
                feed_out = self.rec_q.forward(self.rec_q.make_input_data(input_data))
                self.rec_q.backward(teach_data)
                error += (teach_data - feed_out) ** 2
                # print(error)
            all_error = np.sum(error)

            if all_error <= const.MAX_ERROR:  # errorが小さければ複数回学習する必要はない
                print('breakできた')
                break
        
        # ここをpopにし、1ゲーム行うたびにlearn_batch()をすることで、リアルタイム学習を行うことができる
        self.past_input = []
        self.past_teach = []

    def save_past_data(self):
        # tステップ目の時の入力情報を保存する
        # input_data = self.rec_q.make_input_data(self.who_cooperated)
        # self.past_input.append(copy.deepcopy(input_data))

        # t+1ステップ目で入力する情報を使ってtステップ目の入力に対する重みを修正するため保存する
        next_step_input_data = self.rec_q.make_input_data(self.who_cooperate)
        teach_data = self.rec_q.make_teach_data(next_step_input_data, self.select_action, self.get_reward)
        self.past_teach.append(copy.deepcopy(teach_data))

        self.trans_env()

    def trans_env(self):
        self.who_cooperated = copy.deepcopy(self.who_cooperate)
        # get_rewardを初期化する
        self.get_reward = 0
