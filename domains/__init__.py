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
from .expedition import DOMAIN as expedition
from .farmland import DOMAIN as farmland
from .hydro_power import DOMAIN as hydro_power
from .labyrinth import DOMAIN as labyrinth
from .market_trader import DOMAIN as market_trader
from .mystery_prime import DOMAIN as mystery_prime
from .n_puzzle import DOMAIN as n_puzzle
from .pancake_sorting import DOMAIN as pancake_sorting
from .pathways import DOMAIN as pathways
from .plotting import DOMAIN as plotting
from .puzznic import DOMAIN as puzznic
from .rovers import DOMAIN as rovers
from .rush_hour import DOMAIN as rush_hour
from .sailing import DOMAIN as sailing
from .settlers import DOMAIN as settlers
from .sokoban import DOMAIN as sokoban
from .sugar import DOMAIN as sugar
from .watering import DOMAIN as watering

DOMAINS: dict[str, Domain] = {
    "block_grouping": block_grouping,
    "counters": counters,
    "delivery": delivery,
    "expedition": expedition,
    "farmland": farmland,
    "hydro_power": hydro_power,
    "labyrinth": labyrinth,
    "market_trader": market_trader,
    "mystery_prime": mystery_prime,
    "n_puzzle": n_puzzle,
    "pancake_sorting": pancake_sorting,
    "pathways": pathways,
    "plotting": plotting,
    "puzznic": puzznic,
    "rovers": rovers,
    "rush_hour": rush_hour,
    "sailing": sailing,
    "settlers": settlers,
    "sokoban": sokoban,
    "sugar": sugar,
    "watering": watering,
}

__all__ = ["Domain", "DOMAINS"]