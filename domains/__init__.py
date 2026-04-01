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

from .block_grouping import DOMAIN as block_grouping
from .counters import DOMAIN as counters
from .delivery import DOMAIN as delivery
from .labyrinth import DOMAIN as labyrinth
from .n_puzzle import DOMAIN as n_puzzle
from .pancake_sorting import DOMAIN as pancake_sorting
from .plotting import DOMAIN as plotting
from .puzznic import DOMAIN as puzznic
from .rush_hour import DOMAIN as rush_hour
from .sokoban import DOMAIN as sokoban
from .watering import DOMAIN as watering

DOMAINS: dict[str, Domain] = {
    "block_grouping": block_grouping,
    "counters": counters,
    "delivery": delivery,
    "labyrinth": labyrinth,
    "n_puzzle": n_puzzle,
    "pancake_sorting": pancake_sorting,
    "plotting": plotting,
    "puzznic": puzznic,
    "rush_hour": rush_hour,
    "sokoban": sokoban,
    "watering": watering,
}

__all__ = ["Domain", "DOMAINS"]