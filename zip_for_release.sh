#!/bin/bash

# Create a temporary directory for the hw2 structure
TEMP_DIR=$(mktemp -d)
HW2_DIR="$TEMP_DIR/hw2"

# Create the hw2 directory
mkdir -p "$HW2_DIR"

# Copy files to hw2 directory
cp -r data "$HW2_DIR/"
cp evaluation.py "$HW2_DIR/"
cp main.py "$HW2_DIR/"
cp model.py "$HW2_DIR/"
cp train.py "$HW2_DIR/"
cp util.py "$HW2_DIR/"
cp requirements.txt "$HW2_DIR/"
cp download_best_model.py "$HW2_DIR/"
cp data.py "$HW2_DIR/"
cp zip_for_submission.sh "$HW2_DIR/"

# Create the zip file from the temporary directory
cd "$TEMP_DIR"
zip -r hw2.zip hw2/

# Move the zip file to the original directory
mv hw2.zip "$OLDPWD/"

# Clean up temporary directory
cd "$OLDPWD"
rm -rf "$TEMP_DIR"

echo "Created hw2.zip successfully!"
