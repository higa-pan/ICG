import copy

import ac6_agent
from softmax import softmax


class Agent(ac6_agent.Agent):
    def decide_action(self, rand0_to_1):
        self.past_input.append(copy.deepcopy(self.who_cooperated))  # 入力データの保存

        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group[0][0])
        # other_groups[random値] にすることで所属集団が増えても対応できる

        # 現在協力している集団の誰かに助けを求め，協力先を変えない
        if self.select_action == 0:
            self.cooperation_group[0][1] = self.group_member_map[self.cooperation_group[0][0]]

        # 現在協力していない集団の誰かに助けを求め，協力先を変えない
        elif self.select_action == 1:
            self.cooperation_group[0][1] = self.group_member_map[other_groups[0]]

        # 現在協力している集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 2:
            self.cooperation_group[0] = [other_groups[0], self.group_member_map[self.cooperation_group[0][0]]]

        # 現在協力していない集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 3:
            self.cooperation_group[0] = [other_groups[0], self.group_member_map[other_groups[0]]]

    def no_save_decide_action(self, rand0_to_1):
        rec_input = self.rec_q.make_input_data(self.who_cooperated)  # 入力用のデータを作る(リカレントだから中間層の入力も合わせる)
        out = self.rec_q.hidden_save_forward(rec_input)  # RNNに(推定のQ値を)出力させる

        self.select_action = softmax(out, rand0_to_1)  # ソフトマックス方式で行動を決定する

        other_groups = list(self.group_member_map.keys())
        other_groups.remove(self.cooperation_group[0][0])

        if self.select_action == 0:
            self.cooperation_group[0][1] = self.group_member_map[self.cooperation_group[0][0]]

        elif self.select_action == 1:
            self.cooperation_group[0][1] = self.group_member_map[other_groups[0]]

        # 現在協力している集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 2:
            self.cooperation_group[0] = [other_groups[0], self.group_member_map[self.cooperation_group[0][0]]]

        # 現在協力していない集団の誰かに助けを求め，協力先を変える
        elif self.select_action == 3:
            self.cooperation_group[0] = [other_groups[0], self.group_member_map[other_groups[0]]]
