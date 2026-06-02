# moviesent  

A lightweight sentiment‑analysis web app that predicts the sentiment of movie reviews. The model is a pre‑trained logistic regression classifier stored as a pickle file and served through a simple Flask interface.

---

## Overview  

`moviesent` demonstrates how to combine data exploration in Jupyter Notebook with a production‑ready Flask API. Users can input a movie review on the web page and receive a sentiment prediction (positive / negative) in real time.

---

## Features  

- **Pre‑trained Logistic Regression model** (`models/logistic_regression_model.pkl`)  
- **Flask web server** (`app.py`) with a clean HTML front‑end (`templates/index.html`)  
- **Jupyter Notebook** for data cleaning, feature engineering, and model training (not included in the repo but referenced for reproducibility)  
- Easy to extend with additional models or APIs  

---

## Tech Stack  

| Layer | Technology |
|-------|------------|
| **Language** | Python |
| **Notebook** | Jupyter |
| **Web Framework** | Flask |
| **Model** | Scikit‑learn (Logistic Regression) |
| **Serialization** | joblib / pickle |
| **Front‑end** | HTML + Bootstrap (via `templates/index.html`) |
| **Environment** | virtualenv / conda |

---

## Installation  

1. **Clone the repository**  

   ```bash
   git clone https://github.com/yourusername/moviesent.git
   cd moviesent
   ```

2. **Create a virtual environment** (optional but recommended)  

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**  

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is missing, install the core packages manually:*

   ```bash
   pip install flask scikit-learn pandas numpy
   ```

4. **(Optional) Set environment variables**  

   If you need an external API (e.g., for additional text preprocessing), create a `.env` file:

   ```dotenv
   EXTERNAL_API_KEY=YOUR_OWN_API_KEY
   ```

   The app reads this variable via `python-dotenv`.

---

## Usage  

### Run the web app  

```bash
python app.py
```

The server starts on `http://127.0.0.1:5000`. Open this URL in a browser, type a movie review into the text box, and click **Predict** to see the sentiment result.

### Explore the notebook  

If you wish to retrain the model or explore the dataset, open the accompanying Jupyter notebook (not listed in the file tree but part of the project) with:

```bash
jupyter notebook path/to/your_notebook.ipynb
```

Make any changes, re‑export the model to `models/logistic_regression_model.pkl`, and restart the Flask app to use the updated model.

---

## License  

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.