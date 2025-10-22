#!/bin/bash
set -euo pipefail

# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# source uv environment
source $HOME/.local/bin/env

# Install python 3.12
uv python install 3.12

echo "==== Setup complete ===="
