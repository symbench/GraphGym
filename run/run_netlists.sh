#!/usr/bin/env bash

# Test for running a single experiment. --repeat means run how many different random seeds.
#python main.py --cfg configs/example.yaml --repeat 3
python main.py --cfg configs/omitted_classification.yaml --repeat 3
