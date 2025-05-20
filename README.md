# Mini-Evolve: Code Evolution System

A genetic programming system that evolves code solutions to find faster implementations using LLM-based mutations.

## Overview

This system takes a baseline implementation of a problem and evolves it to find significantly faster solutions while maintaining correctness. It uses a combination of genetic programming principles and LLM-based code generation to explore the solution space.

## Core Features

- **LLM-Powered Evolution**: Uses language models to generate intelligent mutations
- **Performance Optimization**: Focuses on finding faster implementations
- **Correctness Preservation**: Ensures all evolved solutions maintain 100% correctness
- **Diversity Maintenance**: Balances between performance and solution diversity
- **Configurable Targets**: Set your desired speedup factor

## How It Works

### 1. Core Purpose
- Starts with a baseline solution
- Evolves it to find significantly faster versions
- Maintains 100% correctness throughout evolution
- Configurable target speedup (e.g., 10x faster)

### 2. Main Components
```python
def main(generations: int, top_k: int, problem: str, target_speedup: float):
```
- `generations`: Number of evolution cycles
- `top_k`: Number of solutions to keep between generations
- `problem`: Problem to solve (e.g., "prime_count")
- `target_speedup`: Desired speedup factor

### 3. Evolution Process

#### a. Initialization
- Loads problem file (e.g., `prime_count/input.py`)
- Finds the `<evolve>` block in the code
- Uses original code as baseline
- Creates output directory for solutions

#### b. Generation Loop
For each generation:
1. Sorts solutions by performance
2. Selects survivors in two ways:
   - Top performers (half of top_k)
   - Diverse solutions (other half of top_k)
3. For each survivor:
   - Generates mutations using LLM
   - Validates the new code
   - Evaluates performance
   - Adds to new population

### 4. Key Functions
- `parse_evolve_blocks`: Finds code marked for evolution
- `generate_mutations`: Uses LLM to create new variations
- `evaluate`: Tests correctness and measures speed
- `validate_code`: Checks for syntax errors
- `reconstruct_file`: Properly indents and formats the code

### 5. Selection Strategy
```python
# Take top performers
survivors = population[:top_k//2]

# Add diverse solutions
diverse_solutions = []
for candidate in remaining:
    if not any(is_similar(candidate['code'], s['code']) for s in survivors):
        diverse_solutions.append(candidate)
```
- Balances between performance and diversity
- Prevents getting stuck in local optima

### 6. Stopping Conditions
```python
if best['pass_rate'] == 1.0 and best['speed'] < baseline_speed:
    speedup = baseline_speed / best['speed']
    if speedup >= target_speedup:
        print(f"Reached perfect solution with {speedup:.2f}x speedup—stopping early.")
        break
```
- Stops when we find a solution that is:
  1. 100% correct (pass_rate = 1.0)
  2. Faster than baseline by target_speedup factor
- Or runs out of generations

### 7. Output and Progress
- Prints detailed progress for each generation
- Shows current best solution and speedup
- Saves each generation's best solution to files
- Provides feedback about syntax errors and performance

## Storage and Persistence

The system maintains a persistent record of all evolved solutions in `data.json`. This storage mechanism serves several important purposes:

### Storage Features
- **Solution Persistence**: All generated solutions are saved between runs
- **Resume Capability**: Can continue evolution from previous runs
- **History Tracking**: Maintains a record of all attempts and their performance
- **Error Recovery**: Preserves solutions even if the process is interrupted

### Storage Format
```json
[
  {
    "id": "unique_identifier",
    "code": "evolved_code_snippet",
    "full_code": "complete_file_content",
    "pass_rate": 1.0,
    "speed": 0.000123,
    "feedback": "Performance feedback"
  }
]
```

### Key Fields
- `id`: Unique identifier for each solution
- `code`: The evolved code snippet
- `full_code`: Complete file content with proper indentation
- `pass_rate`: Test pass rate (0.0 to 1.0)
- `speed`: Execution time in seconds
- `feedback`: Performance feedback and error messages

### Storage Operations
- **Loading**: Solutions are loaded at startup to continue evolution
- **Saving**: New solutions are saved after each generation
- **Validation**: Ensures all solutions maintain proper format
- **Error Handling**: Gracefully handles storage failures

### Benefits
1. **Continuous Evolution**: Can run multiple sessions to find better solutions
2. **Solution Recovery**: Never loses progress if process is interrupted
3. **Performance Analysis**: Can analyze evolution history
4. **Debugging**: Helps track how solutions evolved over time

## Usage

### Basic Usage
```bash
python src/evolve.py --problem prime_count --generations 50 --target_speedup 10.0
```

### Command Line Arguments
- `--problem`: Name of the problem directory in problems/ (required)
- `--generations`: Number of evolution generations (default: 20)
- `--top_k`: Number of top survivors each generation (default: 5)
- `--target_speedup`: Target speedup factor to achieve (default: 10.0)

### Example
```bash
# Find a solution 10x faster than baseline
python src/evolve.py --problem prime_count --generations 50 --target_speedup 10.0

# Try for a more aggressive speedup
python src/evolve.py --problem prime_count --generations 100 --target_speedup 20.0
```

## Project Structure
```
mini-evolve/
├── src/
│   ├── evolve.py          # Main evolution logic
│   ├── mutate_prompt.jinja # LLM prompt template
│   └── storage.py         # Solution storage
├── problems/              # Problem-specific implementations
│   ├── prime_count/      # Prime counting problem
│   │   ├── input.py     # Problem definition and baseline
│   │   └── evaluator.py # Problem-specific evaluation
│   └── ...              # Other problems
└── outputs/             # Generated solutions
```

## Adding New Problems

1. Create a new directory in `problems/` for your problem
2. Add `input.py` with:
   - Problem description
   - Function signature
   - Baseline implementation
   - `<evolve>` block around the code to optimize
3. Add `evaluator.py` with:
   - Test case generation
   - Performance measurement
   - Correctness verification
   - Return pass_rate and speed metrics

### Example Problem Structure
```
problems/your_problem/
├── input.py      # Problem definition and baseline implementation
└── evaluator.py  # Problem-specific evaluation logic
```

### Example input.py
```python
def your_function(input):
    """
    Problem description and examples here.
    """
    # <evolve>
    # Your baseline implementation here
    # </evolve>
```

### Example evaluator.py
```python
def evaluate(candidate_code: str, num_tests: int = 200) -> dict:
    """
    Evaluates a candidate solution.
    Returns dict with pass_rate and speed metrics.
    """
    # Your evaluation logic here
    return {
        'pass_rate': pass_rate,  # float between 0 and 1
        'speed': speed          # execution time in seconds
    }
```

## Requirements
- Python 3.8+
- OpenAI API key
- Required packages (see requirements.txt)

## License
[Your License Here] 