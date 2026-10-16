#!/bin/sh
# (re)start the job runner if it is not running; clear stale locks first
cd "$(dirname "$0")"; mkdir -p logs results ckpt
if ! ps -eo args | grep -q "^python3 runner.py jobs.txt"; then
  rm -f results/*.lock
  nohup python3 runner.py jobs.txt 2 > logs/runner_main.out 2>&1 &
  echo restarted
else
  echo running
fi
tail -1 logs/runner.log
