#!/bin/bash

# zip the autograder setup files
zip -r autograder.zip \
run_autograder \
setup.sh \
data/ \
evaluation.py \
evaluation_autograder.py \
util.py \
data.py \
main.py 
