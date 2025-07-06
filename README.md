# IMDb-Movie-Review-Sentiment-Analysis-using-SimpleRNN

```
# 🧠 IMDb Movie Review Sentiment Analysis using SimpleRNN

This project implements a one-to-one Recurrent Neural Network (RNN) model using Keras to classify the sentiment (positive or negative) of IMDb movie reviews.

---

## 📌 Project Type
**One-to-One RNN**  
Each input review (text) is mapped to a single output (positive or negative sentiment).

---

## 🧰 Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib
- IMDb Dataset (50K movie reviews)

---

## 📂 Dataset

- **Name:** IMDb Movie Review Dataset  
- **Size:** 50,000 labeled reviews  
- **Source:** [Kaggle - IMDb Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 📊 Model Architecture

- **Embedding Layer**: Turns words into dense vectors
- **SimpleRNN Layer**: Processes the sequence data
- **Dropout Layer**: Reduces overfitting
- **Dense Layer**: Outputs sigmoid probability

---

## 🛠️ Model Summary

```plaintext
Embedding(input_dim=10000, output_dim=64, input_length=200)
SimpleRNN(units=64)
Dropout(rate=0.5)
Dense(1, activation='sigmoid')
````

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:
   `pip install tensorflow pandas matplotlib scikit-learn`
3. Run the notebook:
   `python sentiment_rnn.py` or open in Google Colab
4. The model will train, validate, and visualize accuracy/loss

---

## 📈 Results

* **Training Accuracy**: \~91%
* **Validation Accuracy**: \~84%
* **No overfitting**: Thanks to Dropout and EarlyStopping

---

## 📉 Loss and Accuracy Graphs

<img src="your_image_path_here.png" width="700"/>

---

## 🔮 Future Improvements

* Use **LSTM** or **GRU** for deeper context
* Try **Bidirectional RNN**
* Experiment with **pre-trained embeddings** (e.g., GloVe)

---

## 📄 License

This project is licensed under the MIT License.

---

## 💡 Author

Made with ❤️ by \[Speranza Deejoe]
[GitHub Profile](https://github.com/yourusername)

```

