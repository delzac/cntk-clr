# cntk-clr
Cyclical Learning rate implementation for Microsoft Cognitive Toolkit CNTK

This repository provides a class that can be used in training that allows the implementation of cyclical learning rate policies, as detailed in Leslie Smith's paper [Cyclical Learning Rates for Training Neural Networks
arXiv:1506.01186v4](https://arxiv.org/abs/1506.01186 "Title")

A cyclical learning rate is a policy of learning rate adjustment that increases the learning rate off a base value in a cyclical nature. Typically the frequency of the cycle is constant, but the amplitude is often scaled dynamically at either each cycle or each training iteration/update (i.e. every minibatch update).

Cyclical learning rate can easily be incorporated into your CNTK training script with 2 addtional lines of code:

```python
sgd_momentum = C.momentum_sgd(...)
clr = CyclicalLeaningRate(sgd_momentum, minibatch_size=32)  # instantiate the class
trainer = C.Trainer(...)

for epoch in range(10):
    for batch in range(100):
        trainer.train_minibatch(...)
        clr.batch_step()  # add this line of code after every trainer.train_minibatch call
```
