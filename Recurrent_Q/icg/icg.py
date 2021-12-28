from const import const_value as const


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
def icg(Agents, Group_member_map, Group_payoff, Group_cooperate_times, Agent_cooperate_times):
    for group_index, members in Group_member_map.items():
        cooperation_times = 0
        # 集団内のエージェントの誰が協力したかを数える
        for member in members:
            if Agents[member].cooperation_group[0][0] == group_index:
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
