# Inception Score
Papers
1. Improved techniques for training GANs (https://arxiv.org/pdf/1606.03498.pdf)
2. A Note on the Inception Score (https://arxiv.org/pdf/1801.01973.pdf)

Generative Adversarial Networks lack an objective function, which makes it difficult to compare performance of different models. The **Inception Score** is a metric for automatically evalutating the quality of image generative models ([Salimans et al.2016](https://arxiv.org/pdf/1606.03498.pdf))

The Inception Score uses an Inception v3 Network pre-trained on ImageNet and calculates a statistic of the network's outputs when applied to generated images.

- The generated image should be sharp rather than blurry
- p(y|x) should be low entropy. (the inception network should be highly confident there is a single object in the image)
- p(y) should be high entropy. (the generative algorithm should output a high diversity of images)

## Evaluating (Black-Box) Generative Models
Why is it so hard?
- The real data distribution p(x) is unknown
- The explicit generative distribution q(x) is unknown. (ex. GANs uses random noise vectors for the latent variable)

Some metrics for the evaluation of generative models
- To approximate density function over generated samples and then calculate the likelihood of held-out samples
- To apply a pre-trained neural network to generated images and calculate statistics of its output or at a particular hidden layer.(Inception Score approach)



## Issues With the Inception Score
1. Suboptimalities of the Inception Score itself
2. Problems with the popular usage of the Inception Score

It is extremely important when reporting the Inception Score of an algorithm to include some alternative scroe demonstrating that the model is not overfitting to training data, validating that the high score achieved is not simply replaying the training data.
