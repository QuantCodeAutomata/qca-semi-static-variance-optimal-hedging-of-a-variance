#!/usr/bin/env python
"""Run all four experiments."""
import subprocess, sys, os

experiments = ['exp1_unconstrained.py', 'exp2_long_only.py', 'exp3_dynamic_hedge.py', 'exp4_validation.py']
for exp in experiments:
    print(f'\n=== Running {exp} ===')
    result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), exp)], capture_output=False)
    if result.returncode != 0:
        print(f'WARNING: {exp} failed with return code {result.returncode}')
