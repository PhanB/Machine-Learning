
# Image Classification with a Multilayer Perceptron



## Goal
To classify handwritten digits with a neural network based on a multilayer perceptron.

![MNIST dataset Example](/images/logo.png)

## Results

Due to the large training datasets and similarities between the training and testing datasets, our results are quite accurate. A factor that played into the results is the number of nodes in the hidden layer. Below is a table with various number of nodes in the hidden layer and the resulting accuracy.
The first test set consisted of images containing digits 0 and 1, while the second test set consisted of images containing digits 0, 1, 2, 3, and 4.


| Nodes in Hidden Layer | Test Set 1 Accuracy| Test Set 2 Accuracy |
| --- | --- | --- |
| 10 | 0.5366 | 0.1965 |
| 20 | 0.5366 | 0.7819 |
| 30 | 0.9995 | 0.9621 |
| 40 | 0.9995 | 0.9634 |
| 100 | 0.9995 | 0.9741 |
| 200 | 0.9995 | 0.9724 |
| 500 | 0.9995 | 0.9595 |

As we can observe, more complex data requires more a complex hidden layer. However, if the hidden layer has too many nodes it results in overfitting, thus decreasing our overall accuracy.

## Libraries

- Pandas – Used to parse the training and testing sets (tab delimited data)
- Sys – Taking in arguments (files for training/test data)
- Os.path – Verifying files exist
- Numpy – For matrix operations
- Random – To generate pseudo random numbers used for initial weights

## Design Choices

- A multilayer perceptron comprised of an input layer, hidden layer, and output layer is used to classify the handwritten digits
  - Tested on a subset of the MNIST dataset
    - Digits in the set are 28x28 pixels
    - The digits are represented by 748 pixel values (0-255) along with a classification of the digit
    - The layers are all modular and the dimensions can be easily edited to work for another dataset
  - Weights are initialized to a random small decimal using python's random library
  - Learning rate is set to 0.05
  - The activation function used is the sigmoid function: 1/(1−e^(−x))
  - The pixel values are normalized before computations to prevent overflow
  - Predictions are calculated by using the feed forward algorithm