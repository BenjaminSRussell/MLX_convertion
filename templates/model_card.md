---
library_name: mlx
tags:
  - mlx
  - quantized
  - {{bits}}-bit
base_model: {{model_id}}
---

# {{repo_id}}

{{bits}}-bit MLX conversion of [`{{model_id}}`](https://huggingface.co/{{model_id}}).

| field | value |
|-------|-------|
| strategy | {{strategy}} |
| bits | {{bits}} |
| source commit | {{git_sha}} |
| converted | {{created_at}} |

## Quality gates

{{metrics_table}}

## Files

{{files_table}}

## Usage

```python
from mlx_lm import load  # or the matching mlx loader for this task
model, tokenizer = load("{{repo_id}}")
```

Converted with [MLX_convertion](https://github.com/BenjaminSRussell/MLX_convertion).
