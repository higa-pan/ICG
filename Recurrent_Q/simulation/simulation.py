from agent.ac4_agent import Agent
import random

import save_data as save
from icg import icg
from const import const_value as const


class Simulation:
    def __init__(self, Num_agent, Num_group):
        self.num_agent = Num_agent
        self.num_group = Num_group
        self.agent_list = [Agent(i) for i in range(self.num_agent)]
        self.group_member_map = {}
        self.make_groups()
        self.group_payoff = {i: 0.0 for i in range(self.num_group)}
        self.group_cooperate_times = {i: [] for i in range(self.num_group)}
        self.agent_cooperate_times = {i: [] for i in range(self.num_agent)}
        self.agent_select_actions = {i: [] for i in range(self.num_agent)}

    def reset_seed(self, Seed):
        random.seed(Seed)

    def make_groups(self):
        icg.make_groups(self.agent_list, self.group_member_map)

    def determine_act(self):
        for i in range(self.num_agent):
            self.agent_list[i].decide_action(random.random())
            self.agent_select_actions[i].append(self.agent_list[i].select_action)

    def no_save_determine_act(self):
        for i in range(self.num_agent):
            self.agent_list[i].no_save_decide_action(random.random())
            self.agent_select_actions[i].append(self.agent_list[i].select_action)

    def view_other_coop(self, agent_number):
        agent_members = list(self.agent_list[agent_number].group_member_map.values())
        self.agent_list[agent_number].view_other_action(self.agent_list[agent_members[0]],
                                                        self.agent_list[agent_members[1]])

    def update_weight(self):
        for i in range(self.num_agent):
            self.agent_list[i].learn_batch()

    def append_learning_data(self, agent_number):
        self.agent_list[agent_number].save_past_data()

    def make_batch_data(self):
        for i in range(self.num_agent):
            self.view_other_coop(i)
            self.append_learning_data(i)

    def trans_next_step(self):
        for i in range(self.num_agent):
            self.view_other_coop(i)
            self.agent_list[i].trans_env()

    def play_game(self, steps):
        for step in range(steps):
            self.determine_act()
            icg.icg(self.agent_list, self.group_member_map, self.group_payoff, self.group_cooperate_times,
                    self.agent_cooperate_times)
            self.make_batch_data()  # 他者の協力状態をみて，教師データを作成する

    def no_save_play_game(self, steps):
        for step in range(steps):
            self.no_save_determine_act()
            icg.icg(self.agent_list, self.group_member_map, self.group_payoff, self.group_cooperate_times,
                    self.agent_cooperate_times)
            self.trans_next_step()

    ## 変更した
    def clear_agent_state(self):
        coop_or_notcoop = [-1, 1]
        # 数はNUM_INPUT_UNIT * 2であっているんだけど，新たに定数作るの面倒だったのでそのまま使ってます
        # 正確には自分が所属する集団の他のメンバーの総数*2) (2 = 協力するかどうか + 助けを求めるかどうか)
        rand_list = [coop_or_notcoop[random.randint(0, 1)] for i in range(const.NUM_INPUT_UNIT * 2)]
        for i in range(self.num_agent):
            self.agent_list[i].clear_vars(random.randint(0, 1), rand_list)

    def reset_group_payoff(self):
        self.group_payoff = {i: 0.0 for i in range(self.num_group)}

    def reset_sequence_data(self):
        self.group_cooperate_times = {i: [] for i in range(self.num_group)}
        self.agent_cooperate_times = {i: [] for i in range(self.num_agent)}
        self.agent_select_actions = {i: [] for i in range(self.num_agent)}

    def save_sequence_data(self, Term, Save_path):
        # make_sequence_data(self.group_cooperate_times, self.agent_cooperate_times, Term, Save_path)
        save.make_coop_data(self.group_cooperate_times, self.agent_cooperate_times, self.agent_select_actions, Term,
                            Save_path)

    def save_payoff_data(self, Save_path):
        save.make_payoff_data(self.agent_list, self.num_agent, self.group_payoff, self.num_group, Save_path)

    def save_weight_data(self, Term, Save_path):
        for agent in self.agent_list:
            save.make_last_weight_data(agent.my_number, agent.rec_q.hidden_weight, agent.rec_q.output_weight, Term,
                                       Save_path)


