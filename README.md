# ClothShapeRecognition

A walk through seven versions of a small neural network, each one tweaked
based on how the last one performed. The task is Fashion-MNIST: 70,000
small grayscale images of clothing (shirts, shoes, bags, and so on) sorted
into 10 categories, 60,000 for training and 10,000 held back for testing.
Each version changes one thing — adding a layer, adding some randomness to
prevent overfitting, adding more neurons, switching the training method, and
so on — so you can see what actually moved the needle.

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

("SGD" and "ADAM" are two different methods for how the network updates
itself during training.)

Switching from SGD to ADAM (model_3 to model_4) is the single biggest jump
here — 2.3 percentage points, bigger than adding layers, adding randomness,
or adding more neurons managed on their own. That's worth remembering
before reaching for "add another layer" as the default first move.

Worth calling out directly: model_5, with two hidden layers, only barely
beats model_4, which has one — 89.3% versus 89.2%. That gap is small enough
to just be noise rather than a real effect of the extra layer. A single
training run isn't enough to say with any confidence whether the second
layer actually helps, hurts, or does nothing at all.

## What I'd do differently

- Run each model several times with different random starting points
  before claiming one beats another by less than half a point — a gap
  this small from a single run isn't something you can draw a real
  conclusion from.
- A small convolutional network (a model type built specifically for
  images) would beat every model in this comparison by a wide margin on
  Fashion-MNIST. This notebook's real value is comparing simpler
  fully-connected models against each other, not chasing the best
  possible accuracy.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook FashionMNIST.ipynb
```

No GPU required. Training all seven models on a normal computer takes a
few minutes. The Fashion-MNIST dataset downloads automatically.
