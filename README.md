# AdditionOperatorNNs

Poor generalizability is a common problem of all artificial neural networks: they tend to make bad predictions for inputs, which are considerably different from the training ones. As an example, one can consider the addition of two integers: after training on small integers neural networks still struggle to predict the sum of two large integers. 

Recently it has been [proposed](https://openreview.net/forum?id=BkbY4psgg) that this limitation is due to absence of recursion in the architectures of neural networks. I am going to investigate this interesting idea here.

## Naive addition

Let's start with a simple addition, it should work. The very first idea is to use a linear perceptron and compare it to a non-linear multilayer net. As expected, linear perceptron learns faster. On the other hand, generalizability is pretty bad for both: ~20 and 5% recpectively. Smth is wrong because obviously linear perceptron knows how to add numbers.

Ok, it seeme like the source of inability to model the addition operator (even with the linear perceptron, whose fucntion is a simple addition) is inability of a neural net to assign the weights to 1, 1 and bias to 0 (which is equivalent to a simple addition for the linear perceptron). It is possible to come as close as we want to (1, 1, 0), but never reach exactly (1, 1, 0). For large values of inputs it results in predicting the wrong answer with a pretty big error. It is expected because when we ask the perceptron to add two random large numbers we basically want it to perform classification with infinite number of classes.

To conclude it is impossible to teach a neural net to perform a simple addition, because the output is infinite dimensional space. A solution to this problem is to teach the nueral net to perform the grade school addition, and that is exactly what authors of the [paper](https://openreview.net/forum?id=BkbY4psgg) do.


![](https://github.com/g3n1uss/AdditionOperatorNNs/blob/master/plots/LinearPerceptronVsMultilayerRelu.png)

## The grade school addition

