#!/bin/bash
find . -path './.venv' -prune -o -type f -name "*.py" -exec wc -l {} + | tail -n1
