import numpy as np
import copy
import const_value as const


class Agent:
    def __init__(self, my_agent_number):
        self.my_number = my_agent_number

        # group_member_mapに所属グループごとのメンバーを記録する
        self.group_member_map = {}

        self.who_cooperated = [0, 0]
        # 現在,group_member_map[a]のメンバーの誰が協力したかどうかが,
        self.who_cooperate = [0, 0]
        # [どのグループに協力するのか, 誰に協力要請するのか]が入る
        self.cooperation_group = 0
        # ゲームで自身が得た報酬が入る
        self.get_reward = 0.0
        # ゲームで得た報酬の合計が入る
        self.payoff = 0.0
        # エージェント0は[協力した、してない],
        # エージェント1は[協力した、してない],
        # [自分が取るべき行動] という順序で参照するように作ってる
        self.Q_table = np.zeros(
            ((const.NUM_OF_MEMBER_IN_GROUPS - 1) * const.NUM_OF_ACTIONS,
             (const.NUM_OF_MEMBER_IN_GROUPS - 1) * const.NUM_OF_ACTIONS,
             const.NUM_OF_ACTIONS))

    def choose_first_cooperate_group(self, rand0_or_1):
        self.cooperation_group = list(self.group_member_map.keys())[rand0_or_1]

    # 所属集団とそのメンバー以外の変数を初期化する
    def clear_vars(self, rand0_or_1):
        self.who_cooperated = [0, 0]
        # 現在,join_group[0]のメンバーの誰が協力したかどうかがwho_cooperated[0]に入っている
        self.who_cooperate = [0, 0]
        # cooperation_group_help = [a, b]なら グループaに協力しエージェントbに協力要請したとする
        self.get_reward = 0.0
        self.payoff = 0.0
        self.Q_table = np.zeros(
            ((const.NUM_OF_MEMBER_IN_GROUPS - 1) * const.NUM_OF_ACTIONS,
             (const.NUM_OF_MEMBER_IN_GROUPS - 1) * const.NUM_OF_ACTIONS,
             const.NUM_OF_ACTIONS))
        self.choose_first_cooperate_group(rand0_or_1)

    # 所属集団のどちらに協力するのかを決める
    def decide_action(self, rand_0_to_1):
        # act_a_Q_value はjoin_group[0]に協力するとした時のQ値を表す
        act_a_q_value = self.Q_table[self.who_cooperated[0]][self.who_cooperated[1]][0]
        # act_b_Q_value はjoin_group[1]に協力するとした時のQ値を表す
        act_b_q_value = self.Q_table[self.who_cooperated[0]][self.who_cooperated[1]][1]

        act_q_value_exp = np.exp([act_a_q_value / const.TEMPERATURE, act_b_q_value / const.TEMPERATURE]).tolist()

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group)

        # ランダムな閾値を用意する
        thr_rand = rand_0_to_1

        softmax_value = 0.0
        # actは選ばれた行動を表す
        act = 0
        for action in range(const.NUM_OF_ACTIONS):
            softmax_value += act_q_value_exp[action] / sum(act_q_value_exp)
            if thr_rand <= softmax_value:
                act = action
                break
        # 現在協力していないグループのリストを作る
        if act == 1:
            self.cooperation_group = other_groups[0]

    # 他者が協力したかどうか、自分に協力要請したかどうかを観測する
    def view_other_action(self, other_agent_A, other_agent_B):
        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_A.cooperation_group in list(self.group_member_map):
            self.who_cooperate[0] = 1
        else:
            self.who_cooperate[0] = 0

        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_B.cooperation_group in list(self.group_member_map):
            self.who_cooperate[1] = 1
        else:
            self.who_cooperate[1] = 0

    # Qテーブルをview_other_actionをして得た情報を用いて更新する
    def update_q_table(self):
        # Q t+1 の値を取得
        next_Q_value = max(self.Q_table[self.who_cooperate[0]][self.who_cooperate[1]][0],
                           self.Q_table[self.who_cooperate[0]][self.who_cooperate[1]][1])

        # 協力する集団がgroup_member_map の何番目(添字)に対応しているかを求める
        cooperate_group_value = list(self.group_member_map.keys())

        group_index = cooperate_group_value.index(self.cooperation_group)

        # まとめて書くと長いため、Q t の値を変数に保存しとく
        now_Q_value = self.Q_table[self.who_cooperated[0]][self.who_cooperated[1]][group_index]

        # Q値の更新
        self.Q_table[self.who_cooperated[0]][self.who_cooperated[1]][group_index] = \
            now_Q_value + const.LEARNING_RATE_A * (self.get_reward + const.DISCOUNT_RATE_G * next_Q_value - now_Q_value)
        # 現在を過去にする
        self.who_cooperated = copy.deepcopy(self.who_cooperate)
        # get_rewardを初期化する
        self.get_reward = 0
