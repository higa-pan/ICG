import plotly.graph_objs as go
import pathlib
import numpy as np
import pandas as pd
import const as const

root_path = pathlib.Path(__file__).resolve().parent.parent
data_seedfile_name = ["seed140891_result", "seed596853_result"] 

save_3d_path = root_path / "3d_plot"
#pathlib.Path.mkdir(save_3d_path)
end = 10001
fig = go.Figure()
#fig = go.Figure(data = [go.Scatter3d(x = hidden_out[:, 0], y = hidden_out[:, 1], z = output_out[:, 0])])

hidden_node = const.NUM_HIDDEN_UNIT
out_node = const.NUM_OUTPUT_UNIT

for seed_file in data_seedfile_name:
    data_path = root_path / "rnn_Help_result_data" / seed_file
    pathlib.Path.mkdir(save_3d_path / seed_file)
    for agent_number in range(3):
        save_path = save_3d_path / seed_file / "agent{}".format(agent_number)
        pathlib.Path.mkdir(save_path)
        for step in range(100):
            hidden_out = pd.read_csv(str(data_path) + "/Hidden_out_{}step_agent{}.csv".format((step + 1) * 10000, agent_number)).values[:, 1:]
            output_out = pd.read_csv(str(data_path) + "/Out_out_{}step_agent{}.csv".format((step + 1) * 10000, agent_number)).values[:, 1:]
            for o_node in range(out_node):
                for h_node1 in range(hidden_node):
                    for h_node2 in range(h_node1 + 1, hidden_node):
                        fig.add_trace(go.Scatter3d(x = hidden_out[:end, h_node1], y = hidden_out[:end,h_node2], z = output_out[:end, o_node], mode="markers"))
                        fig.update_traces(marker_size=1, selector=dict(type='scatter3d'))
                        fig.write_html(str(save_path) + "/{}-{}step_hidden{}_{}_out{}.html".format(step * 10000, (step+1)*1000, h_node1, h_node2, o_node))
                        # fig をclearする処理
                        fig.data = []

