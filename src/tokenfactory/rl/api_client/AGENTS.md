# Project

Python client for Nebius Token Factory API.
OpenAPI specification: [openapi.json](./openapi.json).
Supports Python 3.10+.

# Package structure

## Client module

Main module with client `client.py`:
- class `TokenFactory`:
  - serves as the main entry point for interacting with the API;
  - initializes an `httpx` API client;
  - provides properties for nested resources:
    - `v1alpha1`;
  - passes the initialized API client to nested namespaces;
  - implements call retries with `tenacity`;
  - `httpx` client can be submitted via the constructor;
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
       - resource `inference`:
         - resource `openai`:
           - resource `v1`:
             - method `models`
             - resource `chat`
               - resource `completions`
                  - method `create`
             - resource `completions`
                - method `create`

Methods accept parameters, validate them using corresponding Pydantic models, and return response data as Pydantic models.
Resources package structure reflects the API hierarchy, e.g.: `resources/v1alpha1/fine_tuning/jobs/batches.py`.
Resources are not put in the `__init__.py`; instead, a module with the same name created. E.g.: `resources/v1alpha1/v1alpha1.py`, `resources/v1alpha1/fine_tuning/jobs/jobs.py`

### Included resources

Only resources inside namespace `v1alpha1` are included in the client.

# Models

Models are generated from the OpenAPI specification and are located in the `models.py` module.

# Tests

Tests are written using `pytest`.
`tests/unit` — unit tests.
