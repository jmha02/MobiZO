#!/bin/sh

echo "Running mrpc.sh"
sh experiments/fo/tiny-llama/mrpc.sh

echo "Running qnli.sh"
sh experiments/fo/tiny-llama/qnli.sh

echo "Running qqp.sh"
sh experiments/fo/tiny-llama/qqp.sh

echo "Running rte.sh"
sh experiments/fo/tiny-llama/rte.sh

echo "Running sst2.sh"
sh experiments/fo/tiny-llama/sst2.sh

echo "Running wnli.sh"
sh experiments/fo/tiny-llama/wnli.sh