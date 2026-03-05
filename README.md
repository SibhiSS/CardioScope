# CardioScope

CardioScope is a Python-based project for analyzing and predicting cardiac data using machine learning and signal processing techniques. It leverages several scientific and machine learning libraries for data processing, visualization, and model training/prediction.

## Project Structure

- `main.py`: Main script for data processing, training, and evaluation.
- `predict.py`: Script for making predictions on new cardiac data.
- `data/`: Contains subfolders for `abnormal` and `normal` data samples.
- `training/`: Contains training data divided into multiple sets (e.g., `training-a/`, `training-b/`, etc.), each with `.hea` files.

## Requirements

Install the following Python libraries (preferably in a virtual environment or Conda environment):

- numpy
- pandas
- librosa
- scipy
- scikit-learn
- matplotlib
- seaborn
- joblib

You can install all dependencies using pip:

```bash
pip install numpy pandas librosa scipy scikit-learn matplotlib seaborn joblib
```

Or with conda:

```bash
conda install numpy pandas librosa scipy scikit-learn matplotlib seaborn joblib
```

## Usage

### 1. Environment Setup
- Ensure Python 3.x is installed.
- Install the required libraries as shown above.
- (Optional) Create and activate a virtual environment or Conda environment for isolation.

### 2. Training
- Use `main.py` to process data, train, and evaluate models.
- Example:
   ```bash
   python main.py
   ```

### 3. Prediction
- Use `predict.py` to make predictions on new data.
- Example:
   ```bash
   python predict.py
   ```

## Data
- The `data/` folder contains categorized data for model input.
- The `training/` folder contains multiple sets of `.hea` files for model training.

