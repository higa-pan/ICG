#!/bin/sh

python3 main.py
python3 graph/make_coop_graph.py &
sleep 1s
python3 graph/make_coop_graph_std.py &
