

# python3 src/testscript.py --zip_name "archive(3).zip --filepath "../../images" --train_sample_size 1000 --val_sample_size 200 --test_sample_size 200 --epochs 5

#!/usr/bin/env bash

# Create virtual enviroment 
python3 -m venv assignment_3

# Activate virtual enviroment 
source ./assignment_3/bin/activate 

# requirements
pip install --upgrade pip
python3 -m pip install -r requirements.txt


# Deactivate the virtual environment.
deactivate