# Unified Planning Extensions

> **Note:** This repository is a fork of the [Unified Planning (UP)](https://github.com/aiplan4eu/unified-planning) 
> framework, extended with high-level constructs: 
> arrays, sets, bounded integer parameters in actions, integer range variables, and Boolean counting expressions,
> along with automated compilation to PDDL.
> 
> For the original Unified Planning framework documentation, please see [README_UP_ORIGINAL.md](README_UP_ORIGINAL.md).

This directory contains the source code, compilers, and benchmark domains required to reproduce the experiments for 
our paper *"Compiling Expressive Planning with Data Types"* accepted at ICAPS2026. 
A link to the official publication will be added here soon.

## Table of Contents

- [Extensions](#extensions)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Compilation Pipelines](#compilation-pipelines)
- [Running the Experiments](#running-the-experiments)

---

## Extensions

Our extended Unified Planning framework provides advanced features for modelling complex planning problems naturally:

- **Arrays**: Define multi-dimensional fluents (e.g., grids, matrices).
- **Sets**: Fluents representing sets of objects with standard operations (e.g., union, intersection).
- **Integer Parameters in Actions**: Action parameters that take integer values within a specified range.
- **Range Variables**: Variables that allow iterating over ranges of integers in preconditions/effects.
- **Count**: Counting expressions to evaluate Boolean conditions.

These extensions allow you to write more expressive and concise planning problems compared to standard PDDL, 
which our built-in compilers then automatically translate for standard solvers.

---

## Installation

We highly recommend using a Python virtual environment to avoid conflicts.

### 1. Clone the repository
```bash
git clone https://github.com/carladavesa/unified-planning.git
cd unified-planning
```

### 2. Set up a virtual environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install the Framework
Install our extended version of the Unified Planning framework and its dependencies:
```bash
pip install -e .
pip install -r requirements.txt
```
*(Note: Installing with `-e .` automatically handles the Python path for the `unified_planning` module, so manual `PYTHONPATH` exports are not needed).*

### 4. Install Planners (Engines)
To run the examples, ensure you have the UP integrations for the planners used in the paper 
(e.g., Fast Downward, ENHSP, SymK) installed in your environment. 
Example for Fast Downward:
```bash
pip install up-fast-downward
```

---

## Folder Structure

```text
unified-planning/
├── README.md                           # This file
├── README_UP_ORIGINAL.md               # Original framework documentation
├── unified_planning/                   # Core framework (includes our new extensions and compilers)
└── docs/extensions/
    ├── simple_tutorial.py              # A minimal, didactic example demonstrating extensions
    └── domains/                        # Collection of paper benchmarks
        ├── compilation_solving.py      # Core script for applying compilation & solving strategies
        ├── handcrafted_reader.py       # Script to read/evaluate handcrafted PDDL models
        ├── 15-puzzle/                  # 15-Puzzle models and scripts
        ├── sokoban/                    # Sokoban models and scripts
        ├── dump-trucks/                # Dump-Trucks models and scripts
        ├── labyrinth/                  # Labyrinth models and scripts
        ├── pancake-sorting/            # Pancake-Sorting models and scripts
        └── ... (other domains)
```

---

## Compilation Pipelines

To make high-level models compatible with standard planners, we use **compilation pipelines** that transform 
advanced constructs into a representation compatible for a target planner.


### Available Compilers

| Abbreviation | Name | Feature Removed                                   | Transforms Into                                                            |
|---|---|---------------------------------------------------|----------------------------------------------------------------------------|
| `IPAR` | INT_PARAMETER_ACTIONS_REMOVING | Integer parameters in actions and Range Variables | Expanded actions for each parameter value                                  |
| `AR` | ARRAYS_REMOVING | (Multi)Array fluents of an *element_type*         | An *element_type* fluent + (multi)position objects (p0, p1, p2, ...) as parameters |
| `ALR` | ARRAYS_LOGARITHMIC_REMOVING | Bounded integer array fluents                     | Boolean fluents representing integer bits (logarithmic encoding)           |
| `IR` | INTEGERS_REMOVING | Bounded integer fluents                           | Boolean fluents + number objects (n0, n1, n2, ...) as parameters           |
| `CR` | COUNT_REMOVING | Count expressions                                 | Expanded boolean formulas (e.g., Count >= 2 becomes disjunctions)          |
| `CIR` | COUNT_INT_REMOVING | Count expressions                                 | Integer fluents + sum expressions (each condition becomes a 0/1 fluent)    |
| `SR` | SETS_REMOVING | Set fluents                                       | Boolean array fluents (membership represented as boolean)                  |

### Compilation Strategies
As described in the paper, we can chain these compilers to produce different PDDL models 
(e.g., `uti` = IPAR $\rightarrow$ AR $\rightarrow$ IR). 
The general rule of order is:
1. Remove **integer parameters** in actions (`IPAR`).
2. Remove **high-level structures** (arrays `AR`/`ALR`, sets `SR`).
3. Remove **remaining numeric features** (integers `IR`, counts `CR`/`CIR`).

---

## Running the Experiments

To test the compilation pipelines or reproduce the paper results, start by navigating to the `extensions/` folder:

```bash
cd docs/extensions
```

### Run a Quick Tutorial
You can run a minimal, didactic example to see how the extended syntax operates without invoking a solver:

```bash
python simple_tutorial.py
```

### Run Benchmarks

All benchmark domains evaluated in the paper are located in the `domains/` subdirectory. 
This folder contains both our high-level extended models and the scripts to compile and solve them, 
allowing for direct comparison against standard PDDL handcrafted models.

The domain scripts allow you to easily specify which compilation strategy and which solving engine to use via command-line arguments.

```bash
# Navigate to the domains folder (assuming you are in docs/extensions)
cd domains

# Run 15-puzzle with the 'uti' strategy using Fast Downward
python 15-puzzle/15Puzzle.py --compilation uti --solving fast-downward

# Run Pancake Sorting with the 'log' strategy using Symk
python pancake-sorting/PancakeSorting.py --compilation log --solving symk
```

### Command-Line Options
- `--compilation <strategy>`: The pipeline to apply (`up`, `log`, `uti`, `c`, `ci`, `int`, `sc`, `sci`, `scin`, etc.)
- `--solving <planner>`: The planning engine to use (`fast-downward`, `enhsp`, `symk`, etc.)
