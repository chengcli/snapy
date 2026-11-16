set -eu
export PATH=/home/chengcli/pyenv/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
base=/tmp/snapy-hllc-2f321dc
root=$base/results/cast-cache
mkdir -p "$root"
for trial in 1 2 3; do
 order='before-float64 after-float64 before-float32 after-float32'
 if [ "$trial" = 2 ]; then order='after-float32 before-float32 after-float64 before-float64'; fi
 for mode in $order; do
  variant=${mode%-*}
  precision=${mode##*-}
  export PYTHONPATH=$base/package
  if [ "$variant" = before ]; then export PYTHONPATH=$base/package-before-cast-cache; fi
  out=$root/$variant/$trial-graph-$precision
  mkdir -p "$out"
  python "$base/profile-straka-checked.py" --precision "$precision" --execution cuda-graph --device cuda --tlim 900 --profile --output "$out" > "$out.log" 2>&1
  python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(sys.argv[1],r["elapsed_seconds_including_diagnostics"],r["graph_captures"],r["graph_replays"],flush=True)' "$out/report.json"
 done
done
