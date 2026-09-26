# makemore

Character-level language models built from scratch while following
Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series.
Trained on `names.txt`, 32 033 first names.

One directory per video, each containing the code written along with the
video, a rewrite done from memory and comprehension the next morning, and the exercises from
the video description.

| part | topic | status |
|---|---|---|
| [`part_1/`](part_1) | bigram and trigram models, counting vs gradient descent | done |
| `part_2/` | MLP with character embeddings (Bengio et al., 2003) | — |

## Part 1 — bigram and trigram models

| notebook | content |
|---|---|
| `with_video.ipynb` | bigram model by counting, then as a one-layer neural network |
| `from_scratch.ipynb` | the same model rewritten from memory, without the video |
| `E01.ipynb` | trigram model, by counting and by gradient descent |
| `E02.ipynb` | train/dev/test split, generalization of the bigram vs the trigram |
| `E03.ipynb` | tuning the smoothing constant on the dev split |
| `E04.ipynb` | replacing the one-hot product with direct row indexing |
| `E05.ipynb` | `F.cross_entropy` and the numerical stability it provides |

Average negative log-likelihood, in nats per character.

| model | smoothing | dev |
|---|---|---|
| uniform | — | 3.2958 |
| bigram | 1 | 2.4533 |
| trigram | 1 | 2.2365 |
| trigram | 0.127 (tuned) | 2.2222 |

Test loss of the final model: 2.2236, evaluated once.

The counting model and the neural network converge to the same distribution:
after training, `W.exp()` normalized row-wise reproduces the count table with
a mean absolute error of 5e-4, and both generate identical names from the
same seed.

### What the exercises show

The trigram gains 0.22 nats over the bigram on unseen words, so the extra
context is worth far more than any hyperparameter tuning, which adds 0.014.

Only the trigram overfits. With 19 683 parameters for 182 000 training
examples it reacts to smoothing: lowering it from 1 to 0.001 improves the
training loss by 0.033 and degrades the dev loss by 0.008. The bigram, with
729 parameters, shows no train/dev gap at all.

Multiplying a one-hot vector by `W` is an indexing operation in disguise.
Replacing it with `W[xs]` makes the forward pass 65 times faster and drops
the allocation from 665 MB to 25 MB per iteration, with identical gradients.

Computing the softmax and the log by hand overflows in float32 once logits
reach ~100, silently producing `inf` or `nan`. `F.cross_entropy` subtracts
the maximum logit first, which leaves the result unchanged mathematically
and defined numerically. It is also 1.45x faster over a full
forward-backward step, most of the gain coming from its fused backward.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch matplotlib
jupyter lab
```
