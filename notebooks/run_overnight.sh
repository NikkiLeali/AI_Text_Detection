#!/bin/bash

echo "Starting Overnight Generation"
echo "Current time: $(date)"

# keep Mac awake
caffeinate -d -i -m -u -s &

CAFFEINATE_PID=$!

echo "Mac will stay awake during generation"

# run first notebook (few-shot)
echo "Running 3_ai_generation_fewshot.py ..."
python 3_ai_generation_fewshot.py

echo "Few-shot generation completed at $(date)"
echo "Waiting 60 seconds before starting rewrites..."
sleep 60

# run second notebook (rewrites)
echo "Running 4_ai_generation_rewrites.py ..."
python 4_ai_generation_rewrites.py

echo "All generations completed at $(date)"

# stop caffeinate
kill $CAFFEINATE_PID

echo "Mac can now sleep"