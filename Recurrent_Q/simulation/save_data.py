import pandas as pd
from const import const_value as const


def make_coop_data(Group_coop_list, Agent_coop_list, Agent_select_actions, Term, Save_path):
    data = pd.DataFrame.from_dict(Group_coop_list, orient='index')
    path = Save_path + "/group_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)

    data = pd.DataFrame.from_dict(Agent_coop_list, orient='index')
    path = Save_path + "/agent_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)

    data = pd.DataFrame.from_dict(Agent_select_actions, orient='index')
    path = Save_path + "/agent_select_actions_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)


def make_last_weight_data(agent_number, Hidden_weight, Out_weight, Term, Save_path):
    # 隠れ層の重みのデータを一次元にして保存
    data = pd.DataFrame(Hidden_weight.reshape((1, -1)))
    path = Save_path + "/Hidden_weight_{}step_agent{}.csv".format((Term + 1) * 10000, agent_number)
    data.to_csv(path)

    # 出力層の重みのデータを一次元にして保存
    data = pd.DataFrame(Out_weight.reshape((1, -1)))
    path = Save_path + "/Out_weight_{}step_agent{}.csv".format((Term + 1) * 10000, agent_number)
    data.to_csv(path)


# エージェント・グループの協力回数、エージェント1人のQ値それぞれのstepごとのデータを保存する
def make_sequence_data(group_coop_list, agent_coop_list, agent_q_table, Term, Save_path):
    data = pd.DataFrame.from_dict(group_coop_list, orient='index')
    path = Save_path + "/group_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)

    data = pd.DataFrame.from_dict(agent_coop_list, orient='index')
    path = Save_path + "/agent_coop_times_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)

    data = pd.DataFrame(agent_q_table)
    path = Save_path + "/agent_q_table_{0}-{1}step.csv".format(Term * 10000, (Term + 1) * 10000)
    data.to_csv(path)


def make_payoff_data(Agent_list, Num_agent, Group_payoff_dict, Num_group, Save_path):
    agents_payoff = [Agent_list[i].payoff for i in range(Num_agent)]

    group_payoff_now = [Group_payoff_dict[i] for i in range(Num_group)]

    csv_data = pd.DataFrame({"Group-payoff": group_payoff_now})
    csv_data_path = Save_path + "/group{0}_payoff.csv".format(const.NUM_OF_GROUPS)
    csv_data.to_csv(str(csv_data_path))

    csv_data = pd.DataFrame({"Agent-payoff": agents_payoff})
    csv_data_path = Save_path + "/agent{0}_payoff.csv".format(const.NUM_OF_AGENTS)
    csv_data.to_csv(str(csv_data_path))
