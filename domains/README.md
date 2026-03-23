# Domains Directory

This directory contains planning domain implementations that can be executed via `run.py`.

## How it works

- Each domain lives in a Python module under `domains/`.
- Each domain module must expose a `DOMAIN` object (or a `get_domain()` function)
  that returns an instance of `domains.base.Domain`.
- `run.py` imports `domains.DOMAIN` entries listed in `domains/__init__.py`.

## Adding a new domain

1. Create a new module file, e.g. `domains/my_problem.py`.
2. Implement a `Domain` subclass.
3. Export a `DOMAIN` instance from that module.
4. Add an entry to `domains/__init__.py`:

```py
from .my_problem import DOMAIN as my_problem

DOMAINS["my-problem"] = my_problem
```

5. Run it:

```bash
python run.py --domain my-problem --compilation up --solving fast-downward
```

## Instance support

Domains may expose multiple named instances by overriding `list_instances()` in
their `Domain` implementation.

The runner provides a `--list-instances <domain>` flag to show available names.