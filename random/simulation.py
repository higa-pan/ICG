from agents import Agent
import random
import pandas as pd
import const_value as const


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

    def reset_seed(self, Seed):
        random.seed(Seed)

    def make_groups(self):
        make_groups(self.agent_list, self.group_member_map)

    def determine_act(self):
        for i in range(self.num_agent):
            self.agent_list[i].decide_action(random.random())

    def play_game(self, steps):
        for step in range(steps):
            self.determine_act()
            ic_game(self.agent_list, self.group_member_map, self.group_payoff, self.group_cooperate_times,
                    self.agent_cooperate_times)

    def clear_agent_state(self):
        for i in range(self.num_agent):
            self.agent_list[i].clear_vars(random.randint(0, 1))

    def reset_group_payoff(self):
        self.group_payoff = {i: 0.0 for i in range(self.num_group)}

    def reset_sequence_data(self):
        self.group_cooperate_times = {i: [] for i in range(self.num_group)}
        self.agent_cooperate_times = {i: [] for i in range(self.num_agent)}

    def save_sequence_data(self, Term, Save_path):
        make_sequence_data(self.group_cooperate_times, self.agent_cooperate_times, Term, Save_path)

    def save_payoff_data(self, Save_path):
        make_payoff_data(self.agent_list, self.num_agent, self.group_payoff, self.num_group, self.group_member_map,
                         Save_path)


# エージェント・グループの協力回数、エージェント1人のQ値それぞれのstepごとのデータを保存する
def make_sequence_data(group_coop_list, agent_coop_list, Term, Save_path):
    data = pd.DataFrame.from_dict(group_coop_list, orient='index')
    path = Save_path + "/group_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)

    data = pd.DataFrame.from_dict(agent_coop_list, orient='index')
    path = Save_path + "/agent_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)


def make_payoff_data(Agent_list, Num_agent, Group_payoff_dict, Num_group, Group_member_dict, Save_path):
    agents_payoff = [Agent_list[i].payoff for i in range(Num_agent)]

    # データダイエット
    #agent_join_group_payoff1 = [Group_payoff_dict[list(Agent_list[i].group_member_map.keys())[0]] for i in range(Num_agent)]
    #agent_join_group_payoff2 = [Group_payoff_dict[list(Agent_list[i].group_member_map.keys())[1]] for i in range(Num_agent)]

    group_payoff_now = [Group_payoff_dict[i] for i in range(Num_group)]

    # データダイエット
    #agent_payoff_now1 = [Agent_list[Group_member_dict[i][0]].payoff for i in range(Num_group)]
    #agent_payoff_now2 = [Agent_list[Group_member_dict[i][1]].payoff for i in range(Num_group)]

    # データダイエット
    #csv_data = pd.DataFrame({"Group-payoff": group_payoff_now, "join-agent-payoff1": agent_payoff_now1, "join-agent-payoff2": agent_payoff_now2})
    csv_data = pd.DataFrame({"Group-payoff": group_payoff_now})
    csv_data_path = Save_path + "/group{0}_payoff.csv".format(const.NUM_OF_GROUPS)
    csv_data.to_csv(str(csv_data_path))

    # データダイエット
    #csv_data = pd.DataFrame({"Agent-payoff": agents_payoff, "join-group-payoff1": agent_join_group_payoff1, "join-group-payoff2": agent_join_group_payoff2})
    csv_data = pd.DataFrame({"Agent-payoff": agents_payoff})
    csv_data_path = Save_path + "/agent{0}_payoff.csv".format(const.NUM_OF_AGENTS)
    csv_data.to_csv(str(csv_data_path))


# make_groups はエージェントが所属する集団とメンバーの繋がり(ネットワーク)を作る
# 集団内のメンバーは2人であることに注意
def make_groups(Agents, Group_member_map):
    # ここでは2人1組の集団を作る
    agent_num = 0
    for group_num in range(const.NUM_OF_GROUPS):
        # 最後の番号のエージェントは最初の番号のエージェントと集団になる(円の構造になるため)
        # だから、最後の集団のときは端を繋げる処理に切り替える
        if group_num == const.NUM_OF_GROUPS - 1:
            Group_member_map[group_num] = [const.NUM_OF_AGENTS - 1, 0]
            Agents[const.NUM_OF_AGENTS - 1].group_member_map[group_num] = 0
            Agents[0].group_member_map[group_num] = const.NUM_OF_AGENTS - 1
        else:
            # ここでは、agentNum番目のエージェントはagentNum+1番目のエージェントと集団になるように処理している(円構造を意識)
            Group_member_map[group_num] = [agent_num, agent_num + 1]
            Agents[agent_num].group_member_map[group_num] = agent_num + 1
            agent_num = agent_num + 1
            Agents[agent_num].group_member_map[group_num] = agent_num - 1


# ICGame は全集団でICゲームを行わせる
def ic_game(Agents, Group_member_map, Group_payoff, Group_cooperate_times, Agent_cooperate_times):
    for group_index, members in Group_member_map.items():
        cooperation_times = 0
        # 集団内のエージェントの誰が協力したかを数える
        for member in members:
            if Agents[member].cooperation_group == group_index:
                Agent_cooperate_times[member].append(group_index)
                cooperation_times += 1

        # 協力した人数が2なら+1の報酬
        # 協力した人数が1なら+0の報酬 -> 得点に変化なし
        # 協力した人数が0なら-1の報酬

        if cooperation_times == 2:
            get_reward = 1
        elif cooperation_times == 1:
            get_reward = 0
        else:
            get_reward = -1

        Group_cooperate_times[group_index].append(cooperation_times)
        Group_payoff[group_index] += get_reward
        for member_index in range(const.NUM_OF_MEMBER_IN_GROUPS):
            Agents[members[member_index]].payoff += get_reward
            Agents[members[member_index]].get_reward += get_reward
