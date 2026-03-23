"""Common domain interface for planning problems.

This module defines an abstract base class that domain implementations should
inherit from. It also provides a small adapter helper for modules that expose a
simple `build_problem()` function instead of a full class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from unified_planning.model import Problem

class Domain(ABC):
    """Abstract domain interface."""

    @abstractmethod
    def build_problem(self, instance: Optional[str] = None) -> Problem:
        """Build a planning problem.

        Args:
            instance: Optional instance name (e.g., "default", "hard").

        Returns:
            A `unified_planning.model.Problem`.
        """

    @abstractmethod
    def list_instances(self):
        """Return a mapping of instance names to parameter dicts.

        Useful for `--list-instances` support in the runner.
        """


class FunctionDomain(Domain):
    """Adapter that wraps a simple module-level `build_problem()` function."""

    def __init__(self, build_problem_func, instances_func=None):
        self._build_problem = build_problem_func
        self._instances_func = instances_func

    def build_problem(self, instance: Optional[str] = None) -> Problem:
        return self._build_problem(instance=instance)

    def list_instances(self) -> Dict[str, Dict[str, Any]]:
        if self._instances_func:
            return self._instances_func()
        return super().list_instances()