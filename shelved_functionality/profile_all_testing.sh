for f in testbench_*.ipynb; do
    base=$(basename "$f" .ipynb)
    jupyter nbconvert --to python "$f" --output "${base}.py"
    python3 -m profila annotate -- "${base}.py" --arg1=200 >> "profiling/${base}_output.txt" 2>&1
done
