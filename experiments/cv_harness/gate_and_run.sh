#!/usr/bin/env bash
# Wait for ANY GPU with no FOREIGN compute process + >=30GB free (never preempt yena/root),
# then launch the lean-ablation + label-efficiency runner pinned to that GPU. Gives up after 5h.
set -u
cd /home/kraum/Prototype
SCR=/tmp/claude-16174/-home-kraum-Prototype/dd7a2b0b-3d51-4195-87d9-1e117f8ee71b/scratchpad
GATELOG=/tmp/le_gate.log
: > "$GATELOG"
deadline=$(( $(date +%s) + 5*3600 ))

find_free () {  # echo a free GPU index, or nothing
  # map uuid->index and uuid->has-compute-proc
  local busy; busy=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
  nvidia-smi --query-gpu=index,uuid,memory.free --format=csv,noheader | while IFS=',' read -r idx uuid memf; do
    idx=$(echo "$idx"|tr -d ' '); uuid=$(echo "$uuid"|tr -d ' '); memf=$(echo "$memf"|grep -oE '[0-9]+')
    if ! echo "$busy" | grep -q "$uuid" && [ "${memf:-0}" -ge 30000 ]; then echo "$idx"; break; fi
  done
}

while :; do
  g=$(find_free)
  if [ -n "${g:-}" ]; then
    echo "$(date +%H:%M:%S) GPU $g is free -> launching runner" >> "$GATELOG"
    CUDA_VISIBLE_DEVICES="$g" bash "$SCR/run_leanabl_le.sh" >> "$GATELOG" 2>&1
    echo "$(date +%H:%M:%S) runner exited" >> "$GATELOG"; break
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then echo "$(date +%H:%M:%S) GAVE UP (5h, no free GPU)" >> "$GATELOG"; echo "GATE GAVEUP" > /tmp/le_DONE.marker; break; fi
  echo "$(date +%H:%M:%S) no free GPU (foreign jobs on both); waiting 120s" >> "$GATELOG"
  sleep 120
done
