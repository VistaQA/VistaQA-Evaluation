# VistaQA Evaluation

<p align="center">
  
Official evaluation code for VistaQA, a benchmark for joint visual question answering and pixel-level visual evidence grounding.
</p>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2605.20676"><b>Paper</b></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  🤗 <a href="https://huggingface.co/datasets/mozhina/VistaQA"><b>Dataset</b></a>
  &nbsp;&nbsp;•&nbsp;&nbsp;
  🌐 <a href="https://vistaqa.github.io"><b>Project Page</b></a>
</p>



## Install

```bash
pip install -r requirements.txt
```

## Exact Input Format

Both `--gt` and `--pred` accept either:

- one `.json` file
- one `.jsonl` file
- one directory containing `.json` and/or `.jsonl` files

Records are matched strictly by `image.image_id`.

### Ground Truth Structure

Each GT record must be a JSON object with this structure:

```json
{
  "image": {
    "image_id": 1,
    "width": 812,
    "height": 564
  },
  "question": "what is ... ?",
  "answer": "the red panda",
  "task_type": "ocr",
  "task_domain": "science",
  "hallucination": 0,
  "annotations": [
    {
      "segmentation": {
        "size": [564, 812],
        "counts": "RLE_STRING"
      }
    }
  ]
}
```

Required fields and types:

- `image` object
- `image.image_id` integer
- `question` string
- `answer` string
- `task_type` string
- `task_domain` string
- `hallucination` bool-compatible (`0/1/true/false`)
- `annotations` list
- each `annotations[i].segmentation` must be RLE object with:
  - `size`: `[height, width]` (positive integers)
  - `counts`: string (or bytes)

### Prediction Structure

Each prediction record must be a JSON object with this exact structure:

```json
{
  "image": {
    "image_id": 1,
    "width": 812,
    "height": 564
  },
  "prediction": {
    "text": "model answer",
    "masks": [
      {
        "size": [564, 812],
        "counts": "RLE_STRING"
      }
    ]
  }
}
```

Required fields and types:

- `image` object
- `image.image_id` integer
- `prediction` object
- `prediction.text` string
- `prediction.masks` list
- each `prediction.masks[i]` must be RLE object with:
  - `size`: `[height, width]` (positive integers)
  - `counts`: string (or bytes)

## Alignment Rules

- Every GT `image.image_id` must appear in prediction.
- Prediction cannot contain unknown `image.image_id` values not present in GT.
- Duplicate `image.image_id` in GT or prediction is rejected.

## How To Run

Set OpenAI-compatible environment variables:

```bash
export OPENAI_API_KEY=YOUR_KEY_OR_EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
```

Run:

```bash
python main.py \
  --gt /path/to/gt \
  --pred /path/to/pred \
  --llm_model Qwen2.5-72B-Instruct \
  --epsilon 0.01 \
  --output_dir outputs
```

## Output

A single file is produced:

- `outputs/evaluation_results.json`

Example output:

```json
{
  "metadata": {
    "epsilon": 0.01,
    "judge_model": "Qwen2.5-72B-Instruct",
    "total_samples": 3
  },
  "overall": {
    "standard_average": 0.7214
  },
  "breakdown": {
    "task_type": [
      {"group": "ocr", "standard_average": 0.8012},
      {"group": "reasoning", "standard_average": 0.6120}
    ],
    "domain": [
      {"group": "science", "standard_average": 0.7451},
      {"group": "daily", "standard_average": 0.6898}
    ],
    "is_hallucination": [
      {"group": "false", "standard_average": 0.7602},
      {"group": "true", "standard_average": 0.5833}
    ],
    "instance_group": [
      {"group": "zero_instance", "standard_average": 0.7000},
      {"group": "single_instance", "standard_average": 0.7510},
      {"group": "multi_instance", "standard_average": 0.6642}
    ]
  }
}
```
