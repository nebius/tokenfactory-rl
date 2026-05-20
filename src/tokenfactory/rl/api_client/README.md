# Token Factory Fine-Tuning API Client

TODO usage example.

## Models

Models are generated from the OpenAPI specification and are located in the `models.py` module. To generate models from `openapi.json` use the following command:
```
datamodel-codegen  --input openapi.json --openapi-scopes paths --openapi-include-paths /v1alpha1/* --target-python-version 3.10 --output models.py --output-model-type pydantic_v2.BaseModel --http-ignore-tls --use-non-positive-negative-number-constrained-types --set-default-enum-member
```
