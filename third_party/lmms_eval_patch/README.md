# lmms-eval Setup

QueryStream++ uses an `lmms-eval` adapter for LongVideoBench evaluation.

From the repository root, install the supported `lmms-eval` revision and apply the adapter:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
git -C lmms-eval checkout f7a6d6bf6b1622e08ebf5336a48f98e69d503ea6
python -m pip install -e ./lmms-eval
python third_party/lmms_eval_patch/install.py ./lmms-eval
```

Verify the installation:

```bash
python third_party/lmms_eval_patch/install.py --check ./lmms-eval
```

Then run LongVideoBench:

```bash
bash eval/scripts/run_longvideobench.sh
```
