#!/bin/bash

# Checking if a commit message was provided
if [ $# -eq 0 ]; then
    echo "No commit message provided. Exiting."
    exit 1
fi

# The commit message is the first argument to the script
COMMIT_MESSAGE="$1"

# Navigating to the git repository directory
# Replace '/path/to/your/repo' with the path to your git repository

# Adding all files except .slurm and .txt files
find . -type f ! -name '*.slurm' ! -name '*.txt' -exec git add {} +

# Committing the changes
git commit -m "$COMMIT_MESSAGE"

# Pushing to the main branch
git push origin main

# End of the script

