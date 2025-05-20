"""Main evolution module for the genetic coding agent."""
import os
import argparse
import uuid
import re
import importlib.util
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple, Dict, NamedTuple
import matplotlib.pyplot as plt
import numpy as np

from jinja2 import Environment, FileSystemLoader
from openai import OpenAI
from tqdm import tqdm

from src.storage import load_candidates, save_candidates

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Define possible mutation actions with more specific strategies
# MUTATIONS = [
#     "Convert to numpy vectorized operations",
#     "Implement Strassen's algorithm for 10x10 matrices",
#     "Use list comprehensions instead of loops",
#     "Implement block matrix multiplication",
#     "Use parallel processing with multiprocessing",
#     "Implement cache-friendly memory access patterns",
#     "Use SIMD operations through numpy",
#     "Implement divide-and-conquer approach",
#     "Use memory-mapped arrays for large matrices",
#     "Implement tiled matrix multiplication"
# ]

class EvolveBlock(NamedTuple):
    """Represents a block of code to be evolved."""
    start_line: int
    end_line: int
    content: str
    context_before: str
    context_after: str

def is_similar(code1: str, code2: str) -> bool:
    """Check if two code implementations are too similar."""
    # Remove whitespace and comments
    code1 = ''.join(c for c in code1 if not c.isspace())
    code2 = ''.join(c for c in code2 if not c.isspace())
    
    # Simple similarity check - can be made more sophisticated
    return code1 == code2

def validate_matrix_size(code: str) -> Tuple[bool, str]:
    """
    Validate that the code uses 10x10 matrices.
    Returns (is_valid, error_message)
    """
    # Check for range(3) or other sizes
    if re.search(r'range\(\s*[0-9]+\s*\)', code):
        if not re.search(r'range\(\s*10\s*\)', code):
            return False, "Code must use range(10) for matrix dimensions"
    
    # Check for matrix initialization
    if re.search(r'\[\s*\[\s*0\s*for\s*_\s*in\s*range\(\s*[0-9]+\s*\)\s*\]', code):
        if not re.search(r'\[\s*\[\s*0\s*for\s*_\s*in\s*range\(\s*10\s*\)\s*\]', code):
            return False, "Matrix initialization must use size 10"
    
    return True, ""

def parse_evolve_blocks(file_path: str) -> List[EvolveBlock]:
    """
    Parse a Python file to find evolve-blocks.
    Returns a list of EvolveBlock objects containing the block content and context.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    blocks = []
    in_block = False
    block_start = 0
    context_before = []
    
    for i, line in enumerate(lines):
        if '# <evolve>' in line:
            in_block = True
            block_start = i
            # Store context before the block
            context_before = ''.join(lines[:i])
        elif '# </evolve>' in line and in_block:
            in_block = False
            # Get block content (excluding the markers)
            block_content = ''.join(lines[block_start+1:i])
            # Get context after the block
            context_after = ''.join(lines[i+1:])
            
            blocks.append(EvolveBlock(
                start_line=block_start,
                end_line=i,
                content=block_content,
                context_before=context_before,
                context_after=context_after
            ))
    
    return blocks

def reconstruct_file(block: EvolveBlock, new_content: str) -> str:
    """
    Reconstruct the full file with the evolved block content.
    """
    # Get the indentation level from the evolve block marker
    indent = ''
    for line in block.context_before.split('\n'):
        if '# <evolve>' in line:
            indent = line.split('#')[0]
            break
    
    # Clean and indent the new content
    # First, remove any common leading whitespace from all lines
    lines = new_content.split('\n')
    if lines:
        # Find the minimum indentation
        min_indent = float('inf')
        for line in lines:
            if line.strip():  # Only consider non-empty lines
                leading_spaces = len(line) - len(line.lstrip())
                min_indent = min(min_indent, leading_spaces)
        
        # Remove the common indentation
        if min_indent != float('inf'):
            lines = [line[min_indent:] if line.strip() else line for line in lines]
    
    # Now add the function body indentation
    indented_content = '\n'.join(indent + '    ' + line for line in lines)
    
    # Reconstruct the file with evolve block markers
    return f"{block.context_before}{indent}# <evolve>\n{indented_content}\n{indent}# </evolve>\n{block.context_after}"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evolve code solutions")
    parser.add_argument("--generations", type=int, default=20,
                       help="Number of evolution generations")
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top survivors each generation")
    parser.add_argument("--problem", type=str, required=True,
                       help="Name of the problem directory in problems/")
    parser.add_argument("--target_speedup", type=float, default=10.0,
                       help="Target speedup factor to achieve before stopping (default: 10.0)")
    return parser.parse_args()

def generate_mutations(problem: str, context: str) -> List[str]:
    """
    Generate problem-specific mutations using LLM.
    
    Args:
        problem: Name of the problem
        context: The problem context including function signature and docstring
    
    Returns:
        List of possible mutations to apply
    """
    prompt = f"""Given the following problem context, generate a list of 10 specific mutations or optimizations that could be applied to improve the code's performance while maintaining correctness.

Problem: {problem}
Context:
{context}

For each mutation:
1. Be specific about the optimization technique
2. Focus on performance improvements
3. Consider both algorithmic and implementation-level optimizations
4. Ensure the mutation maintains correctness
5. Make it clear how it would improve the code

Return ONLY a list of 10 mutations, one per line, starting with a dash (-). Do not include any other text or explanations."""

    try:
        response = call_openai(prompt)  # Use small budget for mutation generation
        # Parse the response into a list of mutations
        mutations = [line.strip('- ').strip() for line in response.split('\n') if line.strip().startswith('-')]
        return mutations[:10]  # Ensure we get exactly 10 mutations
    except Exception as e:
        print(f"Warning: Failed to generate mutations: {str(e)}")
        # Fallback to generic mutations
        return [
            "Use list comprehensions instead of loops",
            "Implement divide-and-conquer approach",
            "Use parallel processing with multiprocessing",
            "Implement cache-friendly memory access patterns",
            "Use built-in functions and libraries",
            "Optimize memory usage",
            "Reduce time complexity",
            "Use more efficient data structures",
            "Implement early termination conditions",
            "Use in-place operations"
        ]

def get_mutations(problem: str, block: EvolveBlock) -> List[str]:
    """Get list of possible mutations to apply for the given problem."""
    # Extract context from the block
    context = f"{block.context_before}\n{block.context_after}"
    return generate_mutations(problem, context)

def get_baseline_code(block: EvolveBlock) -> str:
    """
    Get the baseline implementation from the original evolve block content.
    
    Args:
        block: The EvolveBlock containing the original code
        
    Returns:
        The original code from the evolve block
    """
    return block.content

def render_mutation_prompt(block: EvolveBlock, previous_code: str, feedback: str, problem: str) -> str:
    """Render the mutation prompt template with the given code and feedback."""
    env = Environment(loader=FileSystemLoader(os.path.dirname(__file__)))
    template = env.get_template('mutate_prompt.jinja')
    
    # Get the function signature from the context
    context_lines = block.context_before.split('\n')
    function_sig = None
    for line in context_lines:
        if line.strip().startswith('def '):
            function_sig = line.strip()
            break
    
    # Get problem-specific mutations
    mutations = get_mutations(problem, block)
    
    return template.render(
        previous_code=previous_code,
        feedback=feedback,
        mutations=mutations,
        context_before=block.context_before,
        context_after=block.context_after,
        function_sig=function_sig
    )

def call_openai(prompt: str) -> str:
    """
    Calls OpenAI API.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables or .env file")
    
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model='gpt-4.1',
        messages=[
            {'role': 'system', 'content': 'You are a code mutation assistant. Return ONLY raw Python code without any markdown formatting, comments, or explanations. The code must implement matrix multiplication for 10x10 matrices, not 3x3 or any other size.'},
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=1024,
        temperature=0.7,
    )
    return response.choices[0].message.content

def validate_code(code: str) -> Tuple[bool, str]:
    """
    Validate Python code syntax.
    Returns (is_valid, error_message)
    """
    try:
        compile(code, '<string>', 'exec')
        return True, ""
    except SyntaxError as e:
        error_msg = f"Syntax error: {str(e)}\nLine {e.lineno}: {e.text}"
        return False, error_msg
    except Exception as e:
        return False, str(e)

def load_evaluator(evaluator_path: str):
    """
    Load the custom evaluator function from the specified file.
    
    Args:
        evaluator_path: Path to the Python file containing the evaluate function
        
    Returns:
        The evaluate function
    """
    spec = importlib.util.spec_from_file_location("evaluator", evaluator_path)
    if spec is None:
        raise ValueError(f"Could not load evaluator from {evaluator_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if not hasattr(module, 'evaluate'):
        raise ValueError("No evaluate function found in the evaluator file")
    
    return module.evaluate

def plot_performance_curve(generations: List[int], speeds: List[float], pass_rates: List[float], 
                         baseline_speed: float, target_speedup: float, problem: str):
    """
    Plot performance curves showing evolution of speed and pass rate.
    
    Args:
        generations: List of generation numbers
        speeds: List of execution speeds for each generation
        pass_rates: List of pass rates for each generation
        baseline_speed: Original baseline speed
        target_speedup: Target speedup factor
        problem: Name of the problem
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot speed curve
    ax1.plot(generations, speeds, 'b-', label='Execution Speed')
    ax1.axhline(y=baseline_speed, color='r', linestyle='--', label='Baseline Speed')
    ax1.axhline(y=baseline_speed/target_speedup, color='g', linestyle='--', 
                label=f'Target Speed ({target_speedup}x speedup)')
    ax1.set_yscale('log')  # Use log scale for better visualization
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Execution Time (s)')
    ax1.set_title(f'Performance Evolution - {problem}')
    ax1.legend()
    ax1.grid(True)
    
    # Plot pass rate curve
    ax2.plot(generations, pass_rates, 'g-', label='Pass Rate')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Pass Rate')
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / 'outputs' / problem
    plt.savefig(outputs_dir / 'performance_curve.png')
    plt.close()

def main(generations: int, top_k: int = 5, problem: str = None, target_speedup: float = 10.0):
    """
    Main evolution function.
    
    Args:
        generations: Number of evolution generations
        top_k: Number of top survivors each generation
        problem: Name of the problem directory in problems/
        target_speedup: Target speedup factor to achieve before stopping
    """
    if not problem:
        raise ValueError("Problem name is required")
    
    # Construct paths
    base_dir = Path(__file__).parent.parent
    problem_dir = base_dir / 'problems' / problem
    input_file = problem_dir / 'input.py'
    evaluator_file = problem_dir / 'evaluator.py'
    
    if not problem_dir.exists():
        raise ValueError(f"Problem directory not found: {problem_dir}")
    if not input_file.exists():
        raise ValueError(f"Input file not found: {input_file}")
    if not evaluator_file.exists():
        raise ValueError(f"Evaluator file not found: {evaluator_file}")
    
    # Load the custom evaluator
    evaluate = load_evaluator(str(evaluator_file))
    
    # Parse the input file
    blocks = parse_evolve_blocks(str(input_file))
    if not blocks:
        raise ValueError(f"No evolve-blocks found in {input_file}")
    
    # For now, we'll only evolve the first block
    block = blocks[0]
    
    # Create outputs directory if it doesn't exist
    outputs_dir = base_dir / 'outputs' / problem
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Load or initialize population
    population = load_candidates()  # list of dicts
    if not population:
        # Get baseline from the original evolve block
        baseline_code = get_baseline_code(block)
        if not baseline_code:
            raise ValueError("No baseline implementation found in evolve block")
        
        print("\nUsing original code as baseline")
        
        # Reconstruct full file with the evolved block
        full_code = reconstruct_file(block, baseline_code)
        result = evaluate(full_code)
        population = [{
            'id': uuid.uuid4().hex,
            'code': baseline_code,
            'full_code': full_code,
            'pass_rate': result['pass_rate'],
            'speed': result['speed'],
            'feedback': ''
        }]
        save_candidates(population)
    else:
        # Ensure all existing candidates have full_code
        for candidate in population:
            if 'full_code' not in candidate:
                candidate['full_code'] = reconstruct_file(block, candidate['code'])
    
    # Find first valid solution for baseline speed
    baseline_speed = float('inf')
    for candidate in population:
        if candidate['pass_rate'] == 1.0 and candidate['speed'] != float('inf'):
            baseline_speed = candidate['speed']
            break
    
    if baseline_speed == float('inf'):
        print("Warning: No valid baseline solution found. Will use first valid solution as baseline.")
    else:
        print(f"Baseline speed: {baseline_speed:.6f}s")

    # Lists to store performance metrics for plotting
    gen_numbers = []
    speeds = []
    pass_rates = []

    for gen in range(1, generations + 1):
        print(f"\nGeneration {gen}")
        
        # Sort by fitness but ensure diversity
        population.sort(key=lambda c: (-c['pass_rate'], c['speed']))
        
        # Take top performers but also some diverse solutions
        survivors = population[:top_k//2]  # Take half from top performers
        
        # Add some diverse solutions based on code structure
        remaining = population[top_k//2:]
        diverse_solutions = []
        for candidate in remaining:
            if not any(is_similar(candidate['code'], s['code']) for s in survivors):
                diverse_solutions.append(candidate)
                if len(diverse_solutions) >= top_k//2:
                    break
        
        survivors.extend(diverse_solutions)

        new_population = survivors.copy()
        # Mutate survivors
        for parent in survivors:
            # Build feedback from previous attempts
            feedback = []
            if parent.get('feedback'):
                feedback.append(parent['feedback'])
            if parent.get('syntax_error'):
                feedback.append(f"Previous syntax error: {parent['syntax_error']}")
            if parent.get('size_error'):
                feedback.append(f"Previous size error: {parent['size_error']}")
            
            # Join feedback with newlines
            feedback_str = "\n".join(feedback) if feedback else ""
            
            # Get problem-specific mutations
            mutations = get_mutations(problem, block)
            prompt = render_mutation_prompt(block, parent['code'], feedback_str, problem)
            child_code = call_openai(prompt)
            
            # First reconstruct the full file
            full_code = reconstruct_file(block, child_code)
            
            # Validate the full reconstructed code
            is_valid, error_msg = validate_code(full_code)
            if not is_valid:
                # Store the syntax error for next generation
                new_population.append({
                    'id': uuid.uuid4().hex,
                    'code': child_code,
                    'full_code': full_code,
                    'pass_rate': 0.0,
                    'speed': float('inf'),
                    'syntax_error': error_msg,
                    'feedback': f"Previous attempt had syntax error: {error_msg}"
                })
                continue
                
            # Validate matrix size
            is_valid, error_msg = validate_matrix_size(child_code)
            if not is_valid:
                new_population.append({
                    'id': uuid.uuid4().hex,
                    'code': child_code,
                    'full_code': full_code,
                    'pass_rate': 0.0,
                    'speed': float('inf'),
                    'size_error': error_msg,
                    'feedback': f"Previous attempt had size error: {error_msg}"
                })
                continue
            
            # Evaluate the full code
            try:
                result = evaluate(full_code)
                feedback = f"Passed {result['pass_rate']*100:.1f}% tests; avg speed {result['speed']:.6f}s"
                new_population.append({
                    'id': uuid.uuid4().hex,
                    'code': child_code,
                    'full_code': full_code,
                    'pass_rate': result['pass_rate'],
                    'speed': result['speed'],
                    'feedback': feedback
                })
            except Exception as e:
                new_population.append({
                    'id': uuid.uuid4().hex,
                    'code': child_code,
                    'full_code': full_code,
                    'pass_rate': 0.0,
                    'speed': float('inf'),
                    'feedback': f"Evaluation error: {str(e)}"
                })
            
        population = new_population
        save_candidates(population)
        best = population[0]
        print(f"Best solution: pass_rate={best['pass_rate']:.2f}, speed={best['speed']:.6f}s")
        
        # Store metrics for plotting
        gen_numbers.append(gen)
        speeds.append(best['speed'])
        pass_rates.append(best['pass_rate'])
        
        # Write out the best code for later inspection
        output_file = outputs_dir / f"gen_{gen}.py"
        with open(output_file, "w") as f:
            f.write(best["full_code"])
        
        # Update baseline speed if we haven't found a valid one yet
        if baseline_speed == float('inf') and best['pass_rate'] == 1.0:
            baseline_speed = best['speed']
            print(f"Setting baseline speed to: {baseline_speed:.6f}s")
        
        # Early stopping: must have 100% pass rate AND meet target speedup
        if best['pass_rate'] == 1.0 and best['speed'] < baseline_speed and baseline_speed != float('inf'):
            speedup = baseline_speed / best['speed']
            print(f"Current best solution: {speedup:.2f}x speedup")
            if speedup >= target_speedup:
                print(f"Reached perfect solution with {speedup:.2f}x speedup—stopping early.")
                break
            else:
                print(f"Continuing evolution to find {target_speedup}x speedup...")
    
    # Plot performance curves
    plot_performance_curve(gen_numbers, speeds, pass_rates, baseline_speed, target_speedup, problem)

if __name__ == '__main__':
    args = parse_args()
    main(args.generations, args.top_k, args.problem, args.target_speedup) 