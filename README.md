
---

# Stock Price Prediction Model

This repository contains a stock price prediction model developed using time series analysis and LSTM (Long Short-Term Memory) neural networks. The model is deployed using Streamlit for interactive visualization.

## Files

- **LSTM.ipynb**: Jupyter notebook containing the LSTM model implementation.
- **app.py**: Streamlit application file for deploying the model.
- **keras_model.h5**: Pre-trained LSTM model saved in HDF5 format.

## Dependencies

Make sure you have the following Python libraries installed:

- numpy
- pandas
- matplotlib
- scikit-learn
- keras
- streamlit

You can install them using pip:

```bash
pip install numpy pandas matplotlib scikit-learn keras streamlit
```

## Usage

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open your browser and go to `http://localhost:8501` to view the app.

## Model Training

If you want to train the LSTM model yourself, refer to `LSTM.ipynb`. It includes the data preprocessing steps, model architecture, training, and evaluation.

## Credits

- **Author**: Lohit Pattnaik
- **Email**: p.lohit@iitg.ac.in

---
