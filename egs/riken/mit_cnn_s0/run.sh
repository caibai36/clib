#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -uo pipefail

# general configuration
stage=0  # start from 0 if you need to start from data preparation

# Parse the options. (eg. ./run.sh --stage 1)
# Note that the options should be defined as shell variable before parsing
. local/scripts/parse_options.sh || exit 1

echo $stage
