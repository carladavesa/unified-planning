"""Domain problem implementations for `run.py`.

This package contains individual domain/problem implementations that expose a
common interface for creating `unified_planning.model.Problem` instances.

To add a new domain, implement a module (e.g. `my_domain.py`) that exposes a
`DOMAIN` object (or a `get_domain()` function) that returns a `Domain`.
Then add it to the `DOMAINS` mapping below.
"""
from .base import Domain

# ---------------------------------------------------------------------------
# Domain registrations
# ---------------------------------------------------------------------------

# Each entry should map a normalized domain name (dash-separated) to a Domain
# instance.
#
# Example:
#   from .my_domain import DOMAIN as my_domain
#   DOMAINS["my-domain"] = my_domain
#
# The runner (run.py) uses this mapping to locate and execute domains.

from .pancake_sorting import DOMAIN as pancake_sorting
from .puzznic import DOMAIN as puzznic
from .n_puzzle import DOMAIN as n_puzzle
from .counters import DOMAIN as counters
from .plotting import DOMAIN as plotting
from .labyrinth import DOMAIN as labyrinth
from .delivery import DOMAIN as delivery
from .rush_hour import DOMAIN as rush_hour
from .sokoban import DOMAIN as sokoban
from .watering import DOMAIN as watering

DOMAINS: dict[str, Domain] = {
    "delivery": delivery,
    "pancake_sorting": pancake_sorting,
    "puzznic": puzznic,
    "n_puzzle": n_puzzle,
    "counters": counters,
    "plotting": plotting,
    "labyrinth": labyrinth,
    "rush_hour": rush_hour,
    "sokoban": sokoban,
    "watering": watering,
}

__all__ = ["Domain", "DOMAINS"]