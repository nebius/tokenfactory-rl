## Filtering openapi spec

```
python -m filter_openapi.__main__ openapi.json --config filter-rl-job.json --output openapi-rl-job.yaml
```

## Models

Models are generated from the OpenAPI specification and are located in the `models` pacakge.

To generate models from `openapi.json` use the following commands:

```
uv tool install datamodel-code-generator
uv run datamodel-codegen \
    --input openapi-fine-tuning.yaml \
    --output ../models/fine_tuning.py \
    --input-file-type openapi \
    --formatters ruff-check ruff-format \
    --openapi-scopes paths \
    --openapi-include-paths /v1\* \
    --target-python-version 3.10 \
    --output-model-type pydantic_v2.BaseModel \
    --use-non-positive-negative-number-constrained-types \
    --set-default-enum-member  \
    --extra-fields allow \
    --field-constraints
uv run datamodel-codegen \
    --input openapi-rl-job.yaml \
    --output ../models/rl_job.py \
    --input-file-type openapi \
    --formatters ruff-check ruff-format \
    --openapi-scopes paths \
    --openapi-include-paths /v1\* \
    --target-python-version 3.10 \
    --output-model-type pydantic_v2.BaseModel \
    --use-non-positive-negative-number-constrained-types \
    --set-default-enum-member  \
    --extra-fields allow \
    --field-constraints
```
