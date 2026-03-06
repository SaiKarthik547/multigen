import sys
import os
import pathlib

def test_print_paths():
    print(f"\nCWD: {os.getcwd()}")
    print(f"sys.path: {sys.path}")
    for p in sys.path:
        print(f"Path entry: {p}")
