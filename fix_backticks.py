import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List
from pathlib import Path

# Pre-compile regex pattern for better performance
BACKTICK_PATTERN = re.compile(r'(?<!\`)\`([^\`]+)\`(?!\`)')

def fix_file_backticks(file_path: str) -> Tuple[str, int, str]:
    """Fix single backticks in docstrings to double backticks.
    
    Args:
        file_path: Path to the Python file to process
        
    Returns:
        Tuple containing (file_path, number of replacements, status message)
    """
    try:
        # Use pathlib for better path handling
        path = Path(file_path)
        content = path.read_text(encoding='utf-8')
        
        # Early return if no matches found
        if '`' not in content:
            return file_path, 0, "No changes needed"
            
        # Check if there are any single backticks to replace
        matches = BACKTICK_PATTERN.findall(content)
        if not matches:
            return file_path, 0, "No changes needed"
        
        # Replace single backticks with double backticks
        new_content = BACKTICK_PATTERN.sub(r'``\1``', content)
        
        # Only write if content actually changed
        if new_content != content:
            path.write_text(new_content, encoding='utf-8')
            return file_path, len(matches), "Updated"
            
        return file_path, 0, "No changes needed"
    
    except Exception as e:
        return file_path, 0, f"Error: {str(e)}"

def find_python_files(directory: str) -> List[str]:
    """Find all Python files in directory recursively.
    
    Args:
        directory: Root directory to search
        
    Returns:
        List of Python file paths
    """
    return [
        str(p) for p in Path(directory).rglob('*.py')
        if not p.name.startswith('.')  # Skip hidden files
    ]

def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_backticks.py <directory>")
        return
    
    directory = sys.argv[1]
    python_files = find_python_files(directory)
    
    print(f"Found {len(python_files)} Python files to process")
    
    # Process files in parallel with optimal number of workers
    total_files_changed = 0
    total_backticks_replaced = 0
    
    # Use optimal number of workers (min of 32 and CPU count)
    max_workers = min(32, os.cpu_count() or 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks at once
        futures = {executor.submit(fix_file_backticks, file): file for file in python_files}
        
        # Process results as they complete
        for future in as_completed(futures):
            file_path, num_replaced, status = future.result()
            if num_replaced > 0:
                total_files_changed += 1
                total_backticks_replaced += num_replaced
                print(f"Updated {file_path}: {num_replaced} replacements")
    
    print(f"\nSummary:")
    print(f"Total files processed: {len(python_files)}")
    print(f"Files modified: {total_files_changed}")
    print(f"Total backtick replacements: {total_backticks_replaced}")

if __name__ == "__main__":
    main() 