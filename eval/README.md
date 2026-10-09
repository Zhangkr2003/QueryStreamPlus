# QueryStream++ Evaluation

QueryStream++ supports StreamingBench, OVO-Bench, Video-MME, LongVideoBench, and SVBench.

## Requirements
Install the evaluation dependencies:

```bash
python -m pip install -r eval/requirements.txt
```

## Datasets

You can download all supported datasets with the helper script:

```bash
HF_TOKEN=your_huggingface_token \
bash tools/download_evaluation_datasets.sh --accept-licenses
```

To download only selected datasets, pass one or more names. Available names are `streamingbench`, `ovobench`, `videomme`, `longvideobench`, and `svbench`.

```bash
HF_TOKEN=your_huggingface_token \
bash tools/download_evaluation_datasets.sh --accept-licenses \
  --datasets <dataset_name> [<dataset_name> ...]
```

You can also prepare the datasets yourself using the same layout:

```text
data/benchmarks/
├── streamingbench/
│   ├── StreamingBench/Real_Time_Visual_Understanding.csv
│   └── Real-Time Visual Understanding/sample_*/video.mp4
├── ovobench/
│   ├── ovo_bench_new.json
│   └── <source videos or pre-cut clips>
├── video_mme/
│   ├── test.parquet
│   └── videos/<videoID>.mp4
├── longvideobench/
│   ├── lvb_val.json
│   └── videos/
└── svbench/
    ├── Meta/Meta_EN/meta_test.csv
    ├── Dialogue/Dialogue_EN/
    ├── Streaming/Streaming_EN/
    ├── Path/
    ├── Con/Con_EN/
    └── Video/
```

## Run evaluation

Once the datasets are ready, run whichever benchmarks you need:

```bash
bash eval/scripts/run_streamingbench.sh
bash eval/scripts/run_ovobench.sh
bash eval/scripts/run_videomme.sh
bash eval/scripts/run_longvideobench.sh
bash eval/scripts/run_svbench_dialogue.sh
bash eval/scripts/run_svbench_streaming.sh
```

If the datasets are stored elsewhere, provide their paths when launching the evaluation:

```bash
TASK_CSV=<annotation.csv> VIDEO_DIR=<video_dir> bash eval/scripts/run_streamingbench.sh
TASK_JSON=<annotation.json> VIDEO_DIR=<video_dir> bash eval/scripts/run_ovobench.sh
TASK_PARQUET=<annotation.parquet> VIDEO_DIR=<video_dir> bash eval/scripts/run_videomme.sh
HF_HOME=<benchmark_root> bash eval/scripts/run_longvideobench.sh
SVBENCH_ANNOTATION_ROOT=<svbench_root> VIDEO_DIR=<svbench_root> bash eval/scripts/run_svbench_dialogue.sh
SVBENCH_ANNOTATION_ROOT=<svbench_root> VIDEO_DIR=<svbench_root> bash eval/scripts/run_svbench_streaming.sh
```

The launchers use the 4K visual budget by default and save results under `outputs/evaluation/`. StreamingBench, OVO-Bench, Video-MME, and SVBench resume incomplete outputs automatically.

To use another released configuration:

```bash
VISUAL_BUDGET=<2k|4k|6k> \
bash eval/scripts/run_<BENCHMARK>.sh
```

## LongVideoBench Setup

LongVideoBench requires `lmms-eval`. See [`third_party/lmms_eval_patch/README.md`](../third_party/lmms_eval_patch/README.md) for setup.

## SVBench Scoring

Use GPT-4o to score the SVBench predictions:

```bash
OPENAI_API_KEY=your_key \
SVBENCH_EVAL_MODE=<dialogue|streaming> \
bash eval/scripts/score_svbench.sh
```

`SVBENCH_EVAL_MODE` defaults to `dialogue`. Scores are saved under `outputs/evaluation/svbench/`.

## Visual Feature Cache

If you plan to run evaluations multiple times, you may want to precompute and cache the visual features to save time.

```bash
BENCHMARKS=<all|streamingbench,ovobench,videomme,longvideobench,svbench> \
bash eval/scripts/precompute_features.sh
```

The cache is written to `outputs/features/`. Enable it for evaluation with:

```bash
VISUAL_CACHE_DIR=outputs/features \
bash eval/scripts/run_<BENCHMARK>.sh
```
