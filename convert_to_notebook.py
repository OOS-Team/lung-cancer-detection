#!/usr/bin/env python3
"""
Converts Python script with cell markers to Jupyter notebook
Usage: python convert_to_notebook.py input.py output.ipynb
"""

import sys
import json
from nbformat import v4 as nbf

def convert_to_notebook(input_file, output_file):
    # Read the Python file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into cells
    cells = []
    current_cell = []
    current_cell_type = "code"
    
    for line in content.split('\n'):
        if line.startswith('# %% [markdown]'):
            # Save the previous cell if it exists
            if current_cell:
                cell_content = '\n'.join(current_cell)
                if current_cell_type == "code":
                    cells.append(nbf.new_code_cell(cell_content))
                else:
                    cells.append(nbf.new_markdown_cell(cell_content))
            # Start a new markdown cell
            current_cell = []
            current_cell_type = "markdown"
            continue
        elif line.startswith('# %%'):
            # Save the previous cell if it exists
            if current_cell:
                cell_content = '\n'.join(current_cell)
                if current_cell_type == "code":
                    cells.append(nbf.new_code_cell(cell_content))
                else:
                    cells.append(nbf.new_markdown_cell(cell_content))
            # Start a new code cell
            current_cell = []
            current_cell_type = "code"
            continue
            
        # Skip the cell marker line
        if not line.startswith('# %%'):
            # For markdown cells, remove the leading '# ' if it exists
            if current_cell_type == "markdown" and line.startswith('# '):
                current_cell.append(line[2:])
            else:
                current_cell.append(line)
    
    # Append the last cell
    if current_cell:
        cell_content = '\n'.join(current_cell)
        if current_cell_type == "code":
            cells.append(nbf.new_code_cell(cell_content))
        else:
            cells.append(nbf.new_markdown_cell(cell_content))
    
    # Create the notebook
    nb = nbf.new_notebook()
    nb.cells = cells
    
    # Write the notebook to a file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_notebook.py input.py output.ipynb")
        sys.exit(1)
    
    convert_to_notebook(sys.argv[1], sys.argv[2])