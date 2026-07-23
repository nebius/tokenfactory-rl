# Project

Python client for Nebius Token Factory API.

Implements client for services with OpenAPI specifications:
 - schemas/openapi-fine-tuning.yaml
 - schemas/openapi-rl-job.yaml

Supports Python 3.10+.

# Package structure

## Client module

Main module with client `_client.py`:
- class `TokenFactory`:
  - serves as the main entry point for interacting with the API;
  - initializes an `httpx` API client;
  - provides properties for nested resources:
    - `v1alpha1`;
    - `v1`;
  - passes the initialized API client to nested resources;
  - implements call retries with `tenacity`;
  - `httpx` client factory can be submitted via the constructor;
  - synchronous only (uses `httpx.Client`);
  - supports context manager protocol;

## API resources

API resources are organized in the package `resources` in a way to reflect the API's hierarchy.
API resources can provide access to nested resources.
Resource classes have methods corresponding to the API endpoints, which use the initialized API client to make requests.
Resource classes have method `endpoint` that returns the API endpoint URL for the resource.
Resources hierarchy:
- resource `v1alpha1`:
   - resource `fine_tuning`:
     - resource `jobs`:
       - method `get_runtime_status`
       - resource `batches`
            - method `create`
            - method `get`
            - method `submit_samples`
- resource `v1`:
   - resource `fine_tuning`:
     - resource `jobs`:
       - method `create`
       - method `list`
       - method `get`
       - method `cancel`
       - method `get_events`
       - resource `checkpoints`
         - method `list`
         - method `get`

API methods are resource class methods, they accept request parameters as method arguments, validate them using corresponding Pydantic models, and return response data as Pydantic models.
Resources package structure reflects the API hierarchy, e.g.: `resources/v1alpha1/fine_tuning/jobs/_batches.py`.
Resources are not put in the `__init__.py`; instead, a private module with the same name created. E.g.: `resources/v1alpha1/_v1alpha1.py`, `resources/v1alpha1/fine_tuning/jobs/_jobs.py`

### Included resources

Only resources described in the section 'API resources' are included in the client.

# Models

Models are generated from the OpenAPI specification and are located in the `models` package.

# Tests

Tests are written using `pytest`.
`tests/unit` — unit tests.
