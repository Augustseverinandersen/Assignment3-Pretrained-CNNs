

# python3 src/testscript.py --filepath "../../images" --train_sample_size 10000 --val_sample_size 2000 --test_sample_size 2000 --epochs 10

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