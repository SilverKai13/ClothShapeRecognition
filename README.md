# ClothShapeRecognition

A walk through seven progressively bigger fully-connected networks on
Fashion-MNIST (60,000 training images, 10,000 test images, 28x28
grayscale clothing items across 10 classes), each change motivated by
the previous model's result: add a hidden layer, add dropout, add
neurons, switch optimizer, add a second hidden layer, shrink the batch
size. This one didn't need a rewrite so much as a fragility fix.

## The bug that was here

Every model in the notebook prints its own test accuracy, but the
notebook never stored those numbers anywhere. So when it wanted to say
"model_4 is 2.4% better than model_3," the only way to get that number
was to copy the two accuracy values by hand out of the printed output
and hardcode them into a `(0.868... - 0.892...) * 100` cell. That's
fine right up until you rerun the notebook: the hardcoded numbers stay
frozen at whatever the first run produced, silently disconnected from
whatever the cells above them just computed. Three cells did this.

Fixed by giving each model's test accuracy its own variable
(`test_acc_0` through `test_acc_6`) and computing every diff from those
variables instead of from typed-in literals. Also added a final
summary comparing all seven models at once — the original notebook
just stopped after the seventh model's raw accuracy number, with no
markdown wrap-up telling you which one actually won.

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

Worth calling out directly: the original notebook's own conclusion
about model_5 doesn't hold up on rerun. It had claimed adding a second
hidden layer *reduced* accuracy by 0.3% and used that to argue against
the extra layer. Retrained fresh, model_5 comes out very slightly
*ahead* of model_4 (89.3% vs 89.2%), not behind. The honest reading of
both runs together: two hidden layers vs. one makes essentially no
difference for this architecture on this dataset, and whether it reads
as a tiny win or a tiny loss depends on which run you happened to catch
it on. Neither the original claim nor this run's number is "the" right
answer — a 0.1-0.3 point gap on a single run, no repeated trials, isn't
a result you can hang a conclusion on either way.

## What I'd do differently

- Average over a few seeds before claiming any model beats another by
  less than half a point. Both this run and the original's single run
  are exactly the kind of single-sample comparison that produces
  exactly this "which one actually won" confusion.
- Log accuracy to a list or dict as you go instead of relying on
  reading printed output back into hardcoded literals — the fix here,
  but worth having in mind from the start next time.
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
