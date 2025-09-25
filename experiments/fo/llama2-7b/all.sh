#!/bin/sh

echo "Running boolq.sh"
sh experiments/fo/llama2-7b/boolq.sh

echo "Running copa.sh"
sh experiments/fo/llama2-7b/copa.sh

echo "Running multirc.sh"
sh experiments/fo/llama2-7b/multirc.sh

echo "Running rte.sh"
sh experiments/fo/llama2-7b/rte.sh

echo "Running sst2.sh"
sh experiments/fo/llama2-7b/sst2.sh

echo "Running wic.sh"
sh experiments/fo/llama2-7b/wic.sh

echo "Running winograde.sh"
sh experiments/fo/llama2-7b/winograde.sh

echo "Running wsc.sh"
sh experiments/fo/llama2-7b/wsc.sh

echo "Running arc_e.sh"
sh experiments/fo/llama2-7b/arc_e.sh

echo "Running arc_c.sh"
sh experiments/fo/llama2-7b/arc_c.sh
