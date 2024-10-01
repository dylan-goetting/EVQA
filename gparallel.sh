task=goat
parts=18
itr=8
il=10000
name=oct1
gpu_id=1
#split=val
split=val_unseen
port=5000
his=0
slp=200
CMD1="python scripts/dynamicBench_run.py --task  ${task} --his ${his} -itr ${itr} -il ${il} -msg 30  --name ${name} --catch --split ${split} --parts ${parts} --parallel -lf 100 --port ${port}"

mkdir -p "parallel/${task}_${name}"

SESSION_NAMES=()

aggregator_session="aggregator_${name}"
tmux new-session -d -s "$aggregator_session" "bash -i -c 'source activate habitat && python scripts/aggregator.py --name ${task}_${name} --sleep ${slp} --port ${port}'"
SESSION_NAMES+=("$aggregator_session")

cleanup() {
  echo ""
  echo "Caught interrupt signal. Cleaning up tmux sessions..."

  for session in "${SESSION_NAMES[@]}"; do
    if tmux has-session -t "$session" 2>/dev/null; then
      tmux kill-session -t "$session"
      echo "Killed session: $session"
    fi
  done

  # Optionally, kill the aggregator session if it's running
  if tmux has-session -t "$aggregator_session" 2>/dev/null; then
    tmux kill-session -t "$aggregator_session"
    echo "Killed session: $aggregator_session"
  fi

  exit 1
}

trap cleanup SIGINT
# Start the tmux sessions
for x in $(seq 0 $(($parts-1)))  # Start from 0, and loop up to MOD-1
do
  gpu_id=$((x % 2))
  # Create a new tmux session for CMD1, run the command, and log output
  session_name="${gpu_id}${task}_${name}_${x}/${parts}"

  tmux new-session -d -s "$session_name" "bash -i -c 'source activate habitat && CUDA_VISIBLE_DEVICES=$gpu_id $CMD1  --part $x'"
  SESSION_NAMES+=("$session_name")
done

# Monitor the tmux sessions
while true; do
  sleep 100

  all_done=true

  for x in $(seq 0 $(($parts-1)))
  do
    gpu_id=$((x % 2))
    session_name="${gpu_id}${task}_${name}_${x}/${parts}"

    # Check if the tmux session is still running
    tmux has-session -t "$session_name" 2>/dev/null
    if [ $? != 0 ]; then
      echo "$session_name finished"
    else
      all_done=false
    fi
  done

  # Break the loop if all tasks are finished
  if $all_done; then
    echo "DONE"
    echo "$(date): Sending termination signal to aggregator."
    curl -X POST http://localhost:5000/terminate
    if [ $? -eq 0 ]; then
      echo "$(date): Termination signal sent successfully."
    else
      echo "$(date): Failed to send termination signal."
    fi
    #tell the aggregator to terminate 
    sleep 10
    if tmux has-session -t "$aggregator_session" 2>/dev/null; then
      tmux kill-session -t "$aggregator_session"
      echo "Killed session: $aggregator_session"
    fi
    break
  fi

  # Sleep for 1 second before checking again
done
