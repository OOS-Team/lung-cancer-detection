#!/bin/bash

# Install nbconvert if not already installed
pip install nbformat

# Convert the script to notebook
python convert_to_notebook.py lung_cancer_detection.py lung_cancer_detection.ipynb

echo "Conversion complete! The notebook has been created as lung_cancer_detection.ipynb"