import numpy as np
import copy
from .softmax import softmax
import sys
# 別階層のフォルダのモジュールを読み込む時にはsysを読み込んでこんな感じに書く
# #sys.path.append('./')

# これらはmain*.pyが実行される階層からの相対参照になっていることに注意
from const import const_value as const
import ac2_agents


class Agent(ac2_agents.Agent):
    def __init__(self, my_agent_number):
        super(Agent, self).__init__(my_agent_number)
        self.who_cooperated = [[0, 0], [0, 0]]
        # 現在,group_member_map[a]が協力したがかwho_cooperated[a]に入っている
        self.who_cooperate = [[0, 0], [0, 0]]
        # どのグループに協力・助けを求めるのかが入る
        self.cooperation_group = [[0, 0]]

    def choose_first_cooperate_group(self, rand0_or_1):
        self.cooperation_group[0] = [list(self.group_member_map.keys())[rand0_or_1], const.NUM_OF_AGENTS]

    def clear_vars(self, rand0_or_1, rand_list):
        self.who_cooperated = [[rand_list[0], rand_list[1]], [rand_list[2], rand_list[3]]]
        # 現在,join_group[0]のメンバーの誰が協力したかどうかがwho_cooperated[0]に入っている
        self.who_cooperate = [[rand_list[4], rand_list[5]], [rand_list[6], rand_list[7]]]
        # cooperation_group_help = [a, b]なら グループaに協力しエージェントbに協力要請したとする

    # 所属集団のどちらに協力するのかを決める
    def decide_action(self, rand0_to_1):
        self.past_input.append(copy.deepcopy(self.who_cooperated))  # 入力データの保存

        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1, const.TEMPERATURE, const.NUM_OF_ACTIONS)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group[0][0])
        # other_groups[random値] にすることで所属集団が増えても対応できる

        # 留まる
        if self.select_action == 0:
            self.cooperation_group[0][1] = const.NUM_OF_AGENTS

        # 協力する集団を変えるだけ
        elif self.select_action == 1:
            # = [[other_groups[0]. const.NUM_OF_AGENTS]でいい気がする
            self.cooperation_group[0] = copy.deepcopy([other_groups[0], const.NUM_OF_AGENTS])

        # 現在協力している集団の誰かに助けを求め，協力先を変えない
        elif self.select_action == 2:
            self.cooperation_group[0][1] = self.group_member_map[self.cooperation_group[0][0]]

        # 現在協力していない集団の誰かに助けを求め，協力先を変えない
        elif self.select_action == 3:
            self.cooperation_group[0][1] = self.group_member_map[other_groups[0]]

        # 現在協力している集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 4:
            self.cooperation_group[0] = copy.deepcopy(
                [other_groups[0], self.group_member_map[self.cooperation_group[0][0]]])

        # 現在協力していない集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 5:
            self.cooperation_group[0] = copy.deepcopy([other_groups[0], self.group_member_map[other_groups[0]]])

    def no_save_decide_action(self, rand0_to_1):
        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        # self.past_input.append(copy.deepcopy(rec_input))  # 入力データの保存
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1, const.TEMPERATURE, const.NUM_OF_ACTIONS)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group[0][0])

        if self.select_action == 0:
            self.cooperation_group[0][1] = const.NUM_OF_AGENTS

        elif self.select_action == 1:
            # = [[other_groups[0]. const.NUM_OF_AGENTS]でいい気がする
            self.cooperation_group[0] = copy.deepcopy([other_groups[0], const.NUM_OF_AGENTS])

        elif self.select_action == 2:
            self.cooperation_group[0][1] = self.group_member_map[self.cooperation_group[0][0]]

        elif self.select_action == 3:
            self.cooperation_group[0][1] = self.group_member_map[other_groups[0]]

        # 現在協力している集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 4:
            self.cooperation_group[0] = copy.deepcopy(
                [other_groups[0], self.group_member_map[self.cooperation_group[0][0]]])

        # 現在協力していない集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 5:
            self.cooperation_group[0] = copy.deepcopy([other_groups[0], self.group_member_map[other_groups[0]]])

    # 他者が協力したかどうか、自分に協力要請したかどうかを観測する
    def view_other_action(self, other_agent_A, other_agent_B):
        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_A.cooperation_group[0][0] in list(self.group_member_map):
            self.who_cooperate[0][0] = 1
        else:
            self.who_cooperate[0][0] = -1

        # other_agent_Aが自分に協力要請しているかどうかをチェックする
        # していたら1, していなかったら0
        if other_agent_A.cooperation_group[0][1] == self.my_number:
            self.who_cooperate[0][1] = 1
        else:
            self.who_cooperate[0][1] = -1

        # other_agent_Aが自分側の集団に協力しているかどうかをチェックする
        if other_agent_B.cooperation_group[0][0] in list(self.group_member_map):
            self.who_cooperate[0][1] = 1
        else:
            self.who_cooperate[0][1] = -1

        # other_agent_Bが自分に協力要請しているかどうかをチェックする
        # していたら1, していなかったら0
        if other_agent_B.cooperation_group[0][1] == self.my_number:
            self.who_cooperate[1][1] = 1
        else:
            self.who_cooperate[1][1] = -1