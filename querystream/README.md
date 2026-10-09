# QueryStream

This directory contains the original QueryStream. It supports TimeChat-Online-7B and Qwen2.5-VL-7B-Instruct with OpenCLIP ViT-L/14.

## Requirements

Install the QueryStream dependencies:

```bash
python -m pip install -r querystream/requirements.txt
```

## Model Weights

Download TimeChat-Online-7B and OpenCLIP:

```bash
bash querystream/tools/download_model_weights.sh \
  --output-root "$PWD/querystream/models" \
  --backbone timechat
```

To use Qwen2.5-VL-7B-Instruct instead:

```bash
bash querystream/tools/download_model_weights.sh \
  --output-root "$PWD/querystream/models" \
  --backbone qwen2.5-vl
```

## Inference

Run QueryStream with TimeChat-Online-7B:

```bash
export MODEL_PATH="$PWD/querystream/models/TimeChatOnline-7B"
export CLIP_PRETRAINED="$PWD/querystream/models/openclip/ViT-L-14-openai.pt"
export VIDEO="$PWD/examples/videos/example.mp4"
export QUERY="What happens in the video?"

bash querystream/scripts/infer.sh
```

For Qwen2.5-VL, change the model path:

```bash
export MODEL_PATH="$PWD/querystream/models/Qwen2.5-VL-7B-Instruct"
export CLIP_PRETRAINED="$PWD/querystream/models/openclip/ViT-L-14-openai.pt"
export VIDEO="$PWD/examples/videos/example.mp4"
export QUERY="What happens in the video?"

bash querystream/scripts/infer.sh
```

Results are saved to `querystream/outputs/querystream_result.json`.

For additional options:

```bash
python querystream/demo/infer.py --help
```
