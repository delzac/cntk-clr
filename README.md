# cntk-clr
Cyclical Learning rate implementation for Microsoft Cognitive Toolkit CNTK

Cyclical learning rate is an implementation that  practically eliminates the need to experimentally find the best values and schedule  for  the global  learning  rates. Instead  of  monotonically decreasing the learning rate, this method lets the learning  rate  cyclically  vary  between  reasonable  boundary  values. Training  with  cyclical  learning  rates  instead of  fixed  values  achieves improved  classification  accuracy without a need to tune and often in fewer iterations.

This repository provides a class that can be used in training that allows the implementation of cyclical learning rate policies, as detailed in Leslie Smith's paper [Cyclical Learning Rates for Training Neural Networks
arXiv:1506.01186v4](https://arxiv.org/abs/1506.01186 "Title")

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
