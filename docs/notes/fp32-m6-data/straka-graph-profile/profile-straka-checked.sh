set -eu
export PATH=/home/chengcli/pyenv/bin:$PATH
export PYTHONPATH=/tmp/snapy-hllc-2f321dc/package
export CUDA_VISIBLE_DEVICES=0
root=/tmp/snapy-hllc-2f321dc/results/graph-profile
mkdir -p "$root"
for trial in 1 2 3; do
 order='float64 float32'
 if [ "$trial" = 2 ]; then order='float32 float64'; fi
 for precision in $order; do
  python /tmp/snapy-hllc-2f321dc/profile-straka-checked.py --precision "$precision" --execution cuda-graph --tlim 10 --profile --output "$root/$trial-graph-$precision" > "$root/$trial-graph-$precision.log" 2>&1
 done
done
for precision in float64 float32; do
 python /tmp/snapy-hllc-2f321dc/profile-straka-checked.py --precision "$precision" --execution eager --tlim 10 --profile --output "$root/eager-$precision" > "$root/eager-$precision.log" 2>&1
done
