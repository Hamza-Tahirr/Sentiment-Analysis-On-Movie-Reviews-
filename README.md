# Sentiment Analysis on Movie Reviews

> This is a fork of [alihssan/Sentiment-Analysis-On-Movie-Reviews-](https://github.com/alihssan/Sentiment-Analysis-On-Movie-Reviews-). The original author wrote the notebook as a solution to the sentiment RNN exercise in Udacity's [deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch/tree/master/sentiment-rnn) course repository. This fork only adds small fixes and setup notes (see [Changes in this fork](#changes-in-this-fork)).

This Jupyter notebook trains an LSTM network in PyTorch to classify movie reviews as positive or negative. You can then pass any review text to the trained model and it prints the predicted sentiment.

## What the notebook does

- Loads 25,000 labelled movie reviews from `data/reviews.txt` and `data/labels.txt`
- Lowercases the text, removes punctuation and encodes every word as an integer
- Drops empty reviews and pads or truncates each review to 200 tokens
- Splits the data 80/10/10 into training, validation and test sets and batches it with PyTorch `DataLoader`s
- Defines a `SentimentRNN` model: an embedding layer (400 dimensions), a 2-layer LSTM (256 hidden units), dropout, a linear layer and a sigmoid output
- Trains for 4 epochs with Adam, binary cross-entropy loss and gradient clipping, printing the validation loss every 100 steps
- Reports loss and accuracy on the test set
- Provides a `predict()` function that takes a plain text review and prints whether it is positive or negative

It uses a GPU when CUDA is available and falls back to the CPU otherwise. In the run saved in the notebook, the model reached a test accuracy of 0.795.

## Tech stack

- Python
- PyTorch
- NumPy
- Jupyter Notebook

## Project structure

```
Sentiment_RNN_Exercise.ipynb   data preparation, model, training, testing and inference
requirements.txt               Python dependencies
data/                          reviews.txt and labels.txt (not included, see below)
```

## Dataset

The notebook expects two text files in a `data/` folder next to it:

- `reviews.txt`: one review per line
- `labels.txt`: `positive` or `negative` for each review, one per line

Both files come from the `sentiment-rnn/data` folder of Udacity's repository. From the project folder you can download them with:

```bash
mkdir data
curl -L -o data/reviews.txt https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/sentiment-rnn/data/reviews.txt
curl -L -o data/labels.txt https://raw.githubusercontent.com/udacity/deep-learning-v2-pytorch/master/sentiment-rnn/data/labels.txt
```

The diagrams the notebook links to (`assets/*.png`) aren't in this repository either. If you want them to show up, copy the `assets` folder from the same Udacity directory.

## Setup and usage

```bash
git clone https://github.com/Hamza-Tahirr/Sentiment-Analysis-On-Movie-Reviews-.git
cd Sentiment-Analysis-On-Movie-Reviews-
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Download the dataset as shown above, then start Jupyter and run all cells in order:

```bash
jupyter notebook Sentiment_RNN_Exercise.ipynb
```

Training is much faster on a GPU. Once the model is trained, you can try your own text in the last cells:

```python
predict(net, "The acting was great and I enjoyed every minute.", 200)
```

## Changes in this fork

- The labels are now 1-D arrays, so `BCELoss` gets an input and target of the same shape. Newer PyTorch versions raise an error when the shapes differ.
- `dataiter.next()` is now `next(dataiter)`, since newer DataLoader iterators no longer have a `.next()` method.
- `tokenize_review` maps words that aren't in the vocabulary to 0 instead of raising a `KeyError`.
- Added `requirements.txt`, `.gitignore` and this README.

The outputs saved in the notebook are from the original author's run.

## License

The original repository doesn't include a license file. The notebook is based on material from Udacity's [deep-learning-v2-pytorch](https://github.com/udacity/deep-learning-v2-pytorch) repository, which is released under the MIT License.
