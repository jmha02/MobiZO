#!/bin/sh

echo "Running mrpc.sh"
sh experiments/zo/tiny-llama/mrpc.sh

echo "Running qnli.sh"
sh experiments/zo/tiny-llama/qnli.sh

echo "Running qqp.sh"
sh experiments/zo/tiny-llama/qqp.sh

echo "Running rte.sh"
sh experiments/zo/tiny-llama/rte.sh

echo "Running sst2.sh"
sh experiments/zo/tiny-llama/sst2.sh

echo "Running wnli.sh"
sh experiments/zo/tiny-llama/wnli.sh