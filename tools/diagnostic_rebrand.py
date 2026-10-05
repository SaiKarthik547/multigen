import pathlib
import re

def get_project_files():
    root = pathlib.Path("multigenai")
    return list(root.rglob("*.py"))

def diagnostic_audit():
    files = get_project_files()
    legacy_pattern = re.compile(r'\bovi\b', re.IGNORECASE)
    
    print("--- MGOS Rebranding Diagnostic ---")
    for f in files:
        if "tests" in str(f) or "system_check" in str(f) or "mmaudio_core\\ext" in str(f):
            continue
            
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for i, line in enumerate(fh):
                    if legacy_pattern.search(line):
                        if "multigenai" in line.lower():
                            continue
                        print(f"MATCH: {f}:L{i+1} -> {line.strip()}")
        except Exception as e:
            print(f"ERROR reading {f}: {e}")

if __name__ == "__main__":
    diagnostic_audit()
