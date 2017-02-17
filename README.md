# AdditionOperatorNNs

Poor generalizability is a common problem of all artificial neural networks: they tend to make bad predictions for inputs, which are considerably different from the training ones. As an example, one can consider the addition of two integers: after training on small integers neural networks still struggle to predict the sum of two large integers. 

Recently it has been [proposed](https://openreview.net/forum?id=BkbY4psgg) that this limitation is due to absence of recursion in the architectures of neural networks. I am going to investigate this interesting idea here.
