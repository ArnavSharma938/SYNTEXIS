set -e

uv pip sync uv.lock

echo "Starting full benchmark run"

SUPPORTED_MODELS=(
    "mistral-math"
    "deepseek-math"
)

for model_slug in "${SUPPORTED_MODELS[@]}"; do
    echo ""
    echo "Running evaluation for: $model_slug"
    python src/main.py --evaluate --model_name "$model_slug"
done

echo "Benchmark run complete.
