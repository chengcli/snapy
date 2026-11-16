set -eu
export PATH=/home/chengcli/pyenv/bin:$PATH
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
export CUDA_VISIBLE_DEVICES=0
cd /tmp/snapy-hllc-2f321dc/src
for pair in 1 2 3; do
  order='float64 float32'
  if [ "$pair" = 2 ]; then order='float32 float64'; fi
  for precision in $order; do
    out="/tmp/snapy-hllc-2f321dc/results/pair-$pair-$precision"
    nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw --format=csv > "$out.gpu-before.csv"
    python -u tests/run_fp32_straka.py --precision "$precision" --device cuda --output "$out" > "$out.log" 2>&1
    python -c 'import json,sys; r=json.load(open(sys.argv[1])); print(r["precision"], r["steps"], r["elapsed_seconds_including_diagnostics"], flush=True)' "$out/report.json"
  done
done
python tests/compare_fp32_straka.py /tmp/snapy-hllc-2f321dc/results/pair-1-float64 /tmp/snapy-hllc-2f321dc/results/pair-1-float32 --output /tmp/snapy-hllc-2f321dc/results/comparison
