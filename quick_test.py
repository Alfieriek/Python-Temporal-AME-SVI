#!/usr/bin/env python
"""
Quick diagnostic script to check project setup.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists and print status."""
    path = Path(filepath)
    exists = path.exists()
    size = path.stat().st_size if exists else 0
    
    status = "✓" if exists and size > 0 else "✗"
    size_str = f"({size} bytes)" if exists else "(missing)"
    
    print(f"{status} {filepath:50s} {size_str}")
    return exists and size > 0

def main():
    print("="*70)
    print("Temporal AME Project Setup Check")
    print("="*70)
    print()
    
    # Check Python path
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Current directory: {os.getcwd()}")
    print()
    
    # Check required files
    required_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/base.py",
        "src/models/static_ame.py",
        "src/models/temporal_ame.py",
        "src/inference/__init__.py",
        "src/inference/base.py",
        "src/inference/naive_mf.py",
        "src/inference/structured_mf.py",
        "demo.py",
    ]
    
    print("Checking required files:")
    print("-" * 70)
    
    all_exist = True
    for filepath in required_files:
        exists = check_file_exists(filepath)
        all_exist = all_exist and exists
    
    print()
    
    if all_exist:
        print("✓ All required files present!")
        print()
        print("Testing imports...")
        print("-" * 70)
        
        # Test imports
        try:
            from src.models import StaticAMEModel
            print("✓ Can import StaticAMEModel")
        except ImportError as e:
            print(f"✗ Cannot import StaticAMEModel: {e}")
            all_exist = False
        
        try:
            from src.models import TemporalAMEModel
            print("✓ Can import TemporalAMEModel")
        except ImportError as e:
            print(f"✗ Cannot import TemporalAMEModel: {e}")
            all_exist = False
        
        try:
            from src.inference import TemporalAMENaiveMFVI
            print("✓ Can import TemporalAMENaiveMFVI")
        except ImportError as e:
            print(f"✗ Cannot import TemporalAMENaiveMFVI: {e}")
            all_exist = False
        
        print()
    else:
        print("✗ Some files are missing!")
        print()
        print("To fix this:")
        print("1. Go back through the conversation")
        print("2. Find artifacts with filenames like 'src/models/base.py'")
        print("3. Copy the code into the corresponding file")
        print()
        print("Or ask me to provide all the code again!")
    
    print("="*70)
    
    if all_exist:
        print("✓ Setup complete! You can now run: python demo.py")
    else:
        print("✗ Setup incomplete. Create missing files first.")
    
    print("="*70)

if __name__ == "__main__":
    main()