# RNN (Recurrent Neural Network)

Lectures 
- CS231n
- [Recurrent Neural Networks | MIT 6.S191](https://www.youtube.com/watch?v=SEnXr6v2ifU)
- https://cs224d.stanford.edu/lectures/CS224d-Lecture6.pdf

# Ideas before RNN and their limitations
1. Fixed Window : limited history and fixed length dependency 
2. Use Entire Sequence as Set of Counts : Counts do not preserve **Order**
3. Use a Really Big Fixed Window : No parameter Sharing

# Sequence Modeling : Design Criteria
1. Handle **Variable-length** sequences (in contrast to a fixed-vector length)
2. Track **Long-term** dependencies
3. Maintain information about **Order**
4. **Share Parameters** across the sequence

# RNN (Recurrent Neural Network)
What are RNNs?
- Standard neural networks go from input to output in **one direction** which can not maintain the information about a previous events in a sequence of events.
- RNNs perform the same task for every element of a sequence, with the output being depended on the previous computations.
- RNNs have a "memory" which captures information about what has been calculated so far.

Deep Learning for Sequence Modeling
- RNNs are well suited for **sequence modeling** tasks
- Model sequences via a **recurrence relation**
- Training RNNs with **backpropagation through time**
- Gated cells like **LSTMs** let us model **long-term dependencies**
- Models for music generation, machine translation , classification , ... 

Tasks for RNNs
- One to One 
- Many to One (Classification)
- Many to Many (Music Generation)
- Machine Translation
- Enviromental Modeling
- Autonomous vehicle
- ...

RNNs can also work for Non-Sequence Data (One to One problem)
- It worked in Digit classification through taking a series of “glimpses”
    - [“Multiple Object Recognition with Visual Attention”](https://arxiv.org/abs/1412.7755), ICLR 2015.
- It worked on generating images one piece at a time
    - i.e generating a [captcha](https://ieeexplore.ieee.org/document/7966808)

Apply a recurrence relation at every time step to precess a sequence:
```python
h = tf.math.tanh( self.W_hh * self.h + self.W_xh * x)
# x : input
# h : internal state or hidden state
```

```python
class MyRNN(tf.keras.layers.Layer):
    def __init__(self, rnn_units, input_dim, output_dim):
        super(MyRNN,self).__init__()
        
        # initialize weight matrices
        self.W_xh = self.add_weight([rnn_units, input_dim])
        self.W_hh = self.add_weight([rnn_units, rnn_units])
        self.W_hy = self.add_weight([output_dim,rnn_units])
        
        
        # initialize hideen state to zeros
        self.h = tf.zeros([rnn_units , 1])
        
    def call(self.x):
        # update the hidden state
        self.h = tf.math.tanh( self.W_hh * self.h + self.W_xh * x)
        
        # Compute the output
        output = self.W_hy * self.h
        
        # Return the current output and hidden state
        return output, self.h
    
my_rnn = MyRNN()
hidden_state = [0,0,0,0]

sentence = ["I", "love", "recurrent","neural"]

for word in sentence:
    prediction,hidden_state = my_rnn(word, hidden_state)
    
next_word_prediction = prediction
```
**Note, The Parameter Sharing in RNN where The same function and set of parameters are used at every time step**

#### Initialization trick for RNNs
- Initialize weight matrix to be the identity matrix
- Change activation function to ReLU
- Initialization	idea	first	introduced	in	Parsing	with Compositional Vector Grammars, Socher et al.2013


# Backpropagation
- BackPropagation Through Time (BPTT), : choose the whole sequence
     - Backpropagation through time forward through entire sequence to compute loss, then backward through entire sequence to compute gradient.
     - If we choose the whole sequence it will be so slow and take so much memory and will never converge!

## Truncated Backpropagation through time
- Run forward and backward  thorough chunks of the sequence instead of whole sequence
- Carry hidden states forward in time forever, but only backpropagate for some smaller number of steps
- 느린 속도를 극복하기 위해 배치별로 나누어서 진행
- [karpathy's code](https://gist.github.com/karpathy/d4dee566867f8291f086)
```python
seq_length = 25 # number of steps to unroll the RNN for

inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]
```
## Standard Feed Forward Network
- Take the derivative (gradient) of the loss with respect to each parameter
- Shift parameters in order to minimize loss

## Two widely known issues with training RNNs : Vanishing and Exploding gradient.
the vanishing gradient and exploding gradient problems described in Bengio et al. (1994).
#### Exploding gradients 
- Solution : Gradient clipping to scale big gradients 
- [On the difficulty of training recurrent neural networks](http://proceedings.mlr.press/v28/pascanu13.pdf)
```python
"""gradient clipping"""
grad_norm = np.sum(grad * grad)
if grad_norm > threshold:
    g *= (threshold / grad_norm)
```

#### Vanishing gradients 
#### Why are vanishing gradients a problem?
1. Multiply many **small numbers** together
2. Errors due to further back time steps have smaller and smaller gradients
3. Bias parameters to capture short-term dependencies

Solutions
1. Activation functions : Using **ReLU** prevents derivative from shrinking the gradients when x > 0.
2. Weight initialization : Initialize **weights** to identity matrix and initialize **bias** to zero.
3. Network Architecture : Use a more **Complex recurrent unit with gates** to control what information is passed through

Gated cell : LSTM, GRU, etc ..

# LSTM (Long Short Terms Memory) networks
In a standard RNN, repeating modules contain a simple computation node whereas LSTM networks rely on a gated cell to track information throughout many time steps.

LSTM can solve the **Vanishing Gradient** problem in RNNs.

[LSTM은 왜 tanh를 쓰는 가?](https://www.facebook.com/groups/TensorFlowKR/permalink/478174102523653/)

LSTM modules contain **computational blocks** that **control information flow**
```python
import tensorflow as tf
tf.keras.layers.LSTM(num_units)
```
## Gate
- Information is added or removed through structures called gates
- Gates optionally let information through, for example via a sigmoid neural net layer and pointwise multiplication

## How do LSTMs work?
1. Forget : Forget irrelevant parts of the previous state
2. Store : Store relevent informations new information into the cell state
3. Update : Selectively update cell state values
4. Output : The output gate controls what information is sent to the next time step

## Key Concepts
1. Maintain a **separate cell state** from what is outputted
2. Use **Gates** to control the **flow of information**. (Forget/Store/Update/Output)
3. Backpropagation through time with **uninterrupted gradient flow**


# Attention Mechanisms
Attention mechanisms in neural networks provide **learnable memory access**.






