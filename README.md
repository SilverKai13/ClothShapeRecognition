# ClothShapeRecognition

A walk through seven progressively bigger fully-connected networks on
Fashion-MNIST (60,000 training images, 10,000 test images, 28x28
grayscale clothing items across 10 classes), each change motivated by
the previous model's result: add a hidden layer, add dropout, add
neurons, switch optimizer, add a second hidden layer, shrink the batch
size.

## Results

| model | test accuracy |
|---|---|
| model_0: single layer, SGD | 83.9% |
| model_1: +hidden layer (relu), SGD | 86.5% |
| model_2: +dropout 0.3, SGD | 86.5% |
| model_3: 256-unit hidden, SGD | 86.8% |
| model_4: 256-unit hidden, ADAM | 89.2% |
| **model_5: 2x256-unit hidden, ADAM** | **89.3%** |
| model_6: 256-unit hidden, SGD, batch=32 | 88.5% |

Switching the optimizer from SGD to ADAM (model_3 -> model_4) is the
single biggest jump here, +2.3 points, bigger than adding layers,
dropout, or extra neurons managed on their own. That's a bigger effect
than any of the architecture changes, which is worth remembering before
reaching for "add another layer" as the first move.

Worth calling out directly: model_5 (two hidden layers) only just edges
out model_4 (one hidden layer) — 89.3% vs 89.2%. That gap is small
enough to be noise rather than a real effect of the second layer; a
single run isn't enough to say confidently whether the extra layer
helps, hurts, or does nothing here.

## What I'd do differently

- Average over a few seeds before claiming any model beats another by
  less than half a point — a gap this size on a single run isn't a
  result you can hang a conclusion on.
- A small CNN would beat every model in this comparison by a wide
  margin on Fashion-MNIST; this notebook's real value is in comparing
  fully-connected variants against each other, not in reaching for a
  competitive accuracy number.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook FashionMNIST.ipynb
```

No GPU required. CPU training all seven models takes several minutes.
Fashion-MNIST downloads automatically via `keras.datasets.fashion_mnist`.
