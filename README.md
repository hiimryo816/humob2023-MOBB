# HuMob Challenge 2023 - n-th Place Solution - Team MOBB
This repository contains the code for the 3rd solution of the [HuMob Challenge 2023](https://connection.mit.edu/humob-challenge-2023) held at SIGSPATIAL in Hamburg, Germany,
developed by Team MOBB.

## About Team MOBB
Team MOBB consists of the following members: 
- [Ryo Koyama](https://www.linkedin.com/in/ryo-koyama-b06a55187/)
- Meisaku Suzuki
- Yusuke Nakamura
- Tomohiro Mimura
- Shin Ishiguro

## Structure
```
│
├── notebooks/
│   ├── task1_make_data_train_BA.ipynb    # Generating training data for Task 1
│   ├── task1_make_data_valid_BA.ipynb    # Generating validation data for Task 1
│   ├── task1_make_data_test_BA.ipynb     # Generating test data for Task 1
│   ├── task1_ensemble002.ipynb           # Ensemble model for Task 1
│   ├── task1_sub_test_BA.ipynb           # Submitting test data for Task 1
│   ├── task2_make_data_train_CB.ipynb    # Generating training data for Task 2
│   ├── task2_make_data_valid_CB.ipynb    # Generating validation data for Task 2
│   ├── task2_make_data_test_CB.ipynb     # Generating test data for Task 2
│   ├── task2_ensemble104.ipynb           # Ensemble model for Task 2
│   └── task2_sub_test_CB.ipynb           # Submitting test data for Task 2
│
├── src/
│   ├── util.py                           # Utility functions
│   └── calc_metrics.py                   # Calculating metrics
│
├── README.md                             # Project description and usage
└── LICENSE  
```


### Notebooks Execution Order

1. `task1_make_data_*_BA.ipynb`
    - Purpose: Prepare the training, validation, and test data for Task 1.
    - Outputs: Processed datasets.

2. `task1_ensemble002.ipynb`
    - Purpose: Train the model for Task 1 using the prepared data.
    - Outputs: Trained model.

3. `task1_sub_test_BA.ipynb`
    - Purpose: Inference using the trained model, and generate submission file.
    - Outputs: Submission File.

4. `task2_make_data_*_CB.ipynb`
    - Purpose: Prepare the training, validation, and test data for Task 2.
    - Outputs: Processed datasets.

5. `task2_ensemble104.ipynb`
    - Purpose: Train the model for Task 2 using the prepared data.
    - Outputs: Trained model.

6. `task2_sub_test_CB.ipynb`
    - Purpose: Inference using the trained model, and generate submission file.
    - Outputs: Submission File.
