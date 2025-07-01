#!/usr/bin/env python3
"""
Test Runner for Advanced Agentic Document Intelligence System

This script provides an easy way to run different categories of tests.
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle the result"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {description} - PASSED")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED (exit code: {e.returncode})")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - FAILED (command not found)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for the Document Intelligence System")
    parser.add_argument(
        'test_type', 
        choices=['config', 'system', 'all', 'quick'], 
        help='Type of tests to run'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true', 
        help='Run with verbose output'
    )
    parser.add_argument(
        '--no-capture', '-s',
        action='store_true',
        help='Disable output capture (show print statements)'
    )
    
    args = parser.parse_args()
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent
    
    # Base pytest command
    pytest_cmd = [sys.executable, '-m', 'pytest']
    
    if args.verbose:
        pytest_cmd.extend(['-v', '--tb=short'])
    
    if args.no_capture:
        pytest_cmd.append('-s')
    
    # Add common options
    pytest_cmd.extend(['--tb=short', '--strict-markers'])
    
    success_count = 0
    total_count = 0
    
    if args.test_type == 'config':
        print("🔧 Running Configuration and Environment Tests")
        total_count += 1
        cmd = pytest_cmd + ['tests/test_config.py']
        if run_command(cmd, "Configuration Tests"):
            success_count += 1
    
    elif args.test_type == 'system':
        print("🚀 Running System Functionality Tests")
        total_count += 1
        cmd = pytest_cmd + ['tests/test_system.py']
        if run_command(cmd, "System Functionality Tests"):
            success_count += 1
    
    elif args.test_type == 'quick':
        print("⚡ Running Quick Tests (Config only)")
        total_count += 1
        cmd = pytest_cmd + ['tests/test_config.py', '-k', 'not test_mongodb and not test_vector_store']
        if run_command(cmd, "Quick Configuration Tests"):
            success_count += 1
    
    elif args.test_type == 'all':
        print("🧪 Running All Tests")
        
        # Run config tests first
        total_count += 1
        cmd = pytest_cmd + ['tests/test_config.py']
        if run_command(cmd, "Configuration Tests"):
            success_count += 1
        
        # Run system tests
        total_count += 1
        cmd = pytest_cmd + ['tests/test_system.py']
        if run_command(cmd, "System Functionality Tests"):
            success_count += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total test suites: {total_count}")
    print(f"Passed: {success_count}")
    print(f"Failed: {total_count - success_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()