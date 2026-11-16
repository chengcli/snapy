set -eu
export PATH=/home/chengcli/pyenv/bin:$PATH
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
export CUDA_VISIBLE_DEVICES=0
cd /tmp/snapy-hllc-2f321dc/src
root=/tmp/snapy-hllc-2f321dc/results/replay-benchmark
mkdir -p "$root"
for trial in 1 2 3; do
 order='eager-float64 cuda-graph-float64 cuda-graph-float32'
 if [ "$trial" = 2 ]; then order='cuda-graph-float32 cuda-graph-float64 eager-float64'; fi
 for mode in $order; do
  precision=${mode##*-}
  execution=${mode%-$precision}
  out="$root/$trial-$mode"
  python -u tests/run_fp32_straka.py --precision "$precision" --execution "$execution" --device cuda --output "$out" > "$out.log" 2>&1
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[1],r["elapsed_seconds_including_diagnostics"],r["graph_captures"],r["graph_replays"],flush=True)' "$out/report.json"
 done
done
python -u tests/run_fp32_straka.py --precision float32 --device cuda --output "$root/1-eager-float32" > "$root/1-eager-float32.log" 2>&1
