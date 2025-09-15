#!/usr/bin/env bash

set -e
source "$PYENV/bin/activate"

counter=`ls -1p outputs | grep -v / | wc -l`
nohup python -u main.py >& outputs/out$counter.out &
