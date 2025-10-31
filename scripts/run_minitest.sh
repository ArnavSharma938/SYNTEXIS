set -e

uv pip sync uv.lock

MODEL_NAME="mistral-math" # or deepseek-math as it is the other base supported model
DATASET_SIZE=4 # You might want to change this to a larger number for a more representative mini-test but it was kept small for quick testing in development

echo "Evaluating model: $MODEL_NAME on $DATASET_SIZE problems"
python src/main.py --evaluate --model_name "$MODEL_NAME" --dataset_size "$DATASET_SIZE"
echo "Done"
