# Unified Planning Extensions

This directory contains examples and resources for working with extended features of the Unified Planning framework.

## Table of Contents

- [Installation](#installation)
- [Extensions](#extensions)
- [Folder Structure](#folder-structure)
- [Getting Started with Examples](#getting-started-with-examples)
- [Understanding Compilation Strategies](#understanding-compilation-strategies)
- [Running Examples](#running-examples)

---

## Installation

Follow these steps to get started:

### Prerequisites

- **Python 3.10+**

- To ensure Python can find the package:

```bash
export PYTHONPATH=/path/to/unified-planning:$PYTHONPATH
```

Add this to your `.bashrc` or `.zshrc` to make it permanent.


### 1. Navigate to the Project root

```bash
cd unified-planning
```

### 2. Install the Framework

Install the Unified Planning framework and its dependencies:

```bash
pip install -e .
pip install -r requirements.txt
```

### 3. Run a simple tutorial

Try running a simple tutorial:

```bash
cd docs/extensions
python simple_tutorial.py
```

---

## Extensions

The Unified Planning framework provides advanced features for modelling complex planning problems:

- **Arrays**: Define multi-dimensional fluents (e.g., grids, matrices)
- **Sets**: Fluents representing sets of objects
- **Integer Parameters in Actions**: Parameters that take bounded integer values within a specified range
- **Range Variables**: Variables that allow iterating over ranges of integers in preconditions/effects
- **Count**: Using counting expressions to query set sizes

These extensions allow you to write more expressive and concise planning problems compared to standard PDDL.

---

## Folder Structure

```
extensions/
├── README.md                    # This file
├── tutorial.py                  # Demonstrating example (not solvable)
└── domains/                     # Collection of planning problems
    ├── compilation_solving.py   # Compilation & solving script
    ├── handcrafted_reader.py    # Handcrafted pddl reader script
    ├── 15-puzzle/               # 15-Puzzle domain
    ├── sokoban/                 # Sokoban domain
    ├── dump-trucks/             # Dump-Trucks domain
    ├── labyrinth/               # Labyrinth domain
    ├── pancake-sorting/         # Pancake-Sorting domain
    └── ... (other domains)
```

### File Descriptions

| File | Purpose                                                                                         |
|------|-------------------------------------------------------------------------------------------------|
| `tutorial.py` | A minimal, didactic example demonstrating extensions (not designed to be solved)                |
| `domains/` | Comprehensive collection of planning problems with multiple domains                             |
| `domains/compilation_solving.py` | Core script for applying compilation strategies and solving                                     |
| `domains/handcrafted_reader.py` | Core script for reading the handcrafted pddl files and calling `domains/compilation_solving.py` |

---

## Getting Started with Examples

## Compilation Pipelines

To make high-level models compatible with standard planners, we use **compilation pipelines** that transform advanced constructs into low-level, planner-compatible representations.

### What is Compilation?

Compilation transforms a planning problem with extensions into an equivalent problem that standard planners can solve. For example:
- Arrays might be transformed into multiple fluents
- Integer parameters might be unrolled into separate actions

### Available Compilation Pipelines

#### Compiler Abbreviations

| Abbreviation | Full Name | What it Removes                                   | What it Transforms Into                                                            |
|---|---|---------------------------------------------------|------------------------------------------------------------------------------------|
| `IPAR` | INT_PARAMETER_ACTIONS_REMOVING | Integer parameters in actions and Range Variables | Expanded actions for each parameter value                                          |
| `AR` | ARRAYS_REMOVING | (Multi)Array fluents of an *element_type*         | An *element_type* fluent + (multi)position objects (p0, p1, p2, ...) as parameters |
| `ALR` | ARRAYS_LOGARITHMIC_REMOVING | Bounded integer array fluents                     | Boolean fluents representing integer bits (logarithmic encoding)                   |
| `IR` | INTEGERS_REMOVING | Bounded integer fluents                           | Boolean fluents + number objects (n0, n1, n2, ...) as parameters                   |
| `CR` | COUNT_REMOVING | Count expressions                                 | Expanded boolean formulas (e.g., Count >= 2 becomes disjunctions)                  |
| `CIR` | COUNT_INT_REMOVING | Count expressions                                 | Integer fluents + sum expressions (each condition becomes a 0/1 fluent)            |
| `SR` | SETS_REMOVING | Set fluents                                       | Boolean array fluents (membership represented as boolean)                          |

#### Compilation Order Matters

When applying multiple compilers, **the order is crucial**. The compilation sequence depends on the features your problem uses:

**General guideline:**
1. First, remove **integer parameters** in actions (`IPAR`)
2. Then, remove **high-level structures** (arrays, sets, etc.)
3. Finally, remove **remaining numeric features** (integers, counts, etc.)

**How to choose?**
- Analyze your problem's features (arrays? integers? counts?)
- Apply compilers that target those features

---

## Running Examples

### Running Domain-Specific Models

Navigate to the `domains/` folder to run real solvable problems:

```bash
cd docs/extensions/domains

# Run 15-puzzle with different compilation strategies
python 15-puzzle/15puzzle.py --compilation uti --solving fast-downward
python 15-puzzle/15puzzle.py --compilation log --solving fast-downward

# Run other domains
python sokoban/sokoban.py --compilation up --solving fast-downward
python counters/counters.py --compilation c --solving fast-downward
```

### Command-Line Options

For domain scripts:

- `--compilation <strategy>`: Choose the compilation pipeline
  - Options: `up`, `log`, `uti`, `c`, `ci`, `int`, etc.
  
- `--solving <planner>`: Select the planner to use
  - Options: `fast-downward`, `enhsp`, etc.
