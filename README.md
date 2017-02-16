# AdditionOperatorNNs

It is well known that artificial nueral networks suffer from lack of generalizability, which means that they tend to make bad predictions for inputs, which are considerably different from the training inputs. As an example, one can consider addition of two integers: after training on small integers neural networks still struggle to predict the sum of two large integers. 

Recently it has been [suggested](https://openreview.net/forum?id=BkbY4psgg) that this limitation is due to absense of recursion in neural networks' architectures. I am going to investigate this claim in this repository.
