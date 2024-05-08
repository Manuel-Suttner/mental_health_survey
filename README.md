# Mental Health Survey Analysis

## Overview

This repository contains code for analyzing a mental health survey dataset using Python. The dataset used in this analysis is from the Open Source Mental Illness (OSMI) survey conducted in 2016. The analysis includes preprocessing the text data, performing principal component analysis (PCA), and applying hierarchical clustering to identify patterns in the data.

## Getting Started

To run the code in this repository, follow these steps:

1. **Clone the Repository**:

   git clone https://github.com/Manuel-Suttner/mental_health_survey.git

2. **Install Dependencies**: 
Ensure you have Python installed on your system. Install the required libraries by running:

pip install -r requirements.txt

3. **Download the Dataset**: 
Download the OSMI survey dataset (`osmi-survey-2016_1479139902.json`) from [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-2016) and place it in the `data` directory.

4. **Run the Code**: 
Run the main script:

python mental_health_analysis.py

## Configuring the Dataset Path

The dataset path is specified in the `mental_health_analysis.py` script using an absolute path:
```python
json_file_path = r"C:\Users\yourusername\path\to\mental_health_survey\osmi-survey-2016_1479139902.json"

To avoid file path issues, you can modify this line to make the file path configurable, for example:

import sys

# Check if a custom file path is provided as a command-line argument
if len(sys.argv) > 1:
    json_file_path = sys.argv[1]
else:
    # Default file path
    json_file_path = "data/osmi-survey-2016_1479139902.json"
