# LLM-XTM

ACL 2026 Main: LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models.

Cross-lingual topic modeling framework with LLM-based topic refinement. The main pipeline trains a topic model, refines cross-lingual topic words with an LLM during late training, then evaluates topic quality and cross-lingual classification.

## Supported Models

- `XTRA`
- `NMTM`
- `InfoCTM`

Model configs are in:

```text
configs/model/
```

## Supported Datasets

- `Amazon_Review` (`en`/`cn`)
- `ECNews` (`en`/`cn`)
- `Rakuten_Amazon` (`en`/`ja`)

Dataset configs are in:

```text
configs/dataset/
```

Each dataset folder under `data/<DATASET>/` must contain the preprocessed BoW, vocabulary, labels, document embeddings, and any model-specific files. The existing datasets already include the required embedding files.

## Setup

Install the project dependencies in your Python environment. At minimum, the pipeline expects packages such as:

```bash
pip install torch numpy scipy scikit-learn pyyaml wandb openai sentence-transformers gensim
```

For Gemini-native refinement, also install:

```bash
pip install google-generativeai
```

## LLM API Configuration

The refinement API key is never hardcoded. Set it before running.

The main LLM setup used in the paper is Gemini 2.5 Flash:

```bash
export LLM_PROVIDER=gemini
export GEMINI_API_KEY="your_gemini_api_key"
export GEMINI_MODEL="gemini-2.5-flash"
```

OpenAI-compatible providers are also supported:

```bash
export LLM_PROVIDER=openai
export NVIDIA_API_KEY="your_api_key"
export LLM_BASE_URL="https://integrate.api.nvidia.com/v1"
export LLM_MODEL="qwen/qwen3-coder-480b-a35b-instruct"
```

Optional W&B logging:

```bash
export WANDB_API_KEY="your_wandb_key"
```

If `WANDB_API_KEY` is not set, the code will skip explicit `wandb.login(...)` and let W&B use its normal local configuration.

## Run One Experiment

Use `run.sh`:

```bash
bash run.sh <MODEL> <DATASET> [SEED] [DEVICE]
```

You can also call `main.py` directly:

```bash
python main.py \
  --model XTRA \
  --dataset Amazon_Review \
  --seed 0 \
  --device 0 \
  --llm_provider gemini \
  --gemini_api_key "$GEMINI_API_KEY" \
  --llm_model gemini-2.5-flash
```

## Run All Datasets

XTRA:

```bash
bash run_xtra.sh
```

NMTM:

```bash
bash run_nmtm.sh
```

InfoCTM:

```bash
bash run_infoctm.sh
```

## Output

Results are written under:

```text
output/<MODEL>/<DATASET>/<PARAMS>/<TIMESTAMP>/
```

Key files:

```text
T15_en.txt
T15_cn.txt
T15_ja.txt
rst.mat
```

The pipeline then computes:

- CNPMI
- Topic uniqueness
- Intra-language classification
- Cross-language classification

## Notes

- `--gemini_api_key` is kept for backward compatibility as the general refinement API key argument. With `--llm_provider gemini`, it should be a real Gemini key. With `--llm_provider openai`, it should be a key accepted by the configured OpenAI-compatible endpoint.

## Citation

If you use this repository, please cite:

ACL 2026 Main: LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models.

Paper: https://arxiv.org/abs/2605.03299

```bibtex
@misc{xuan2026llmxtmenhancingcrosslingualtopic,
      title={LLM-XTM: Enhancing Cross-Lingual Topic Models with Large Language Models}, 
      author={Minh Chu Xuan and Tien-Phat Nguyen and Linh Ngo Van and Dinh Viet Sang and Nguyen Thi Ngoc Diep and Trung Le},
      year={2026},
      eprint={2605.03299},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2605.03299}, 
}
```
