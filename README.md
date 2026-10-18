# Neural Networks from First Principles Using Pytorch

This repo is to help understand basic building blocks of a neural network from the first principles through a series of labs and discussions on different topics: 

```text
data 
  → model
    → forward pass
      → loss
        → backpropagation (chain rules)
          → optimizer (weight updating)
            → evaluation
              → visualization
```

## Lab Progression

### Sine Function Aproximation

According to "universal approximation theorem", specifically inspired by this paper "Neural Network Approximation: Three Hidden Layers Are Enough", we decided to start with Sine function approximation in the range [-100, 100], which can give us an opportunity to decipher the basic building blocks of a nerual network using PyTorch.

**Task**: train a neural network to approximate Sine Wave in [-100, 100].

Our approach:

- Start with training on 1,000 random points from [-100, 100].
- Hold out 20% as a validation set.
- Test on 1,000 points across [-1000, 1000].

  Evaluate separately:
    - interpolation: [-100,100]
    - extrapolation: [-1000,-100] ∪ [100,1000]
- Build "SineNet", a small multilayer perceptron (`MLP`) with two hidden layers and `Tanh` activations.
- Use MSE loss + Adam.
- Normalize the input because [-100,100] is relatively large for a neural network.
- Plot how training loss and validation loss change with Epochs
- Plot the learned function against the true sine curve.

Training on $x \in [-100, 100]$ but tested on $x \in [-1000, 1000]$ is intentional to address three increasingly important concepts:

```text
1. Function approximation
       │
       ▼
"Can a neural network represent this function?"

2. Generalization
       │
       ▼
"Does it work on unseen data
from the same distribution?"

3. Extrapolation
       │
       ▼
"Does it work outside the distribution
 represented by the training data?"
```

This also hands us a chance to understand "universal approximation theorem". We will discuss this empirically rather than rigorously later.

One more subtle point: Python/PyTorch's `sin()` expects radians, so we'll use `y = sin(x)` directly.

**Note**: `sin(x)` completes about 31.8 periods over [-100, 100]. A small `tanh` `MLP` with two hidden layers of a small width usually has difficulty representing so many oscillations. `Tanh` networks also exhibit spectral bias: they learn smooth, low-frequency behavior first. So, to effectively approximate Sine function, we need special neural network architecture such as Fourier feature layer and sinusoidal activation to capture such high-frequency oscillations. However, our main focus in this repo is to understand basic building blocks of a neural network using PyTorch. That's why we still use a small `MLP` for illustration.

### Overview of Learning Progression

```text
Lab 01
nn.Sequential
    ↓
Understand basic training loop


Lab 02
Explicit nn.Linear layers
    ↓
Understand forward()


Lab 03
Manually implement Linear with nn.Parameter
    ↓
Understand weights, biases, matrix multiplication


Lab 04
Inspect model architecture & parameters
    ↓
Understand autograd

Lab 05
One pass through a small network, SmallNet:
forward
    ↓
loss
    ↓
backward
    ↓
weight updating

Lab 06
Manually work through one backpropagation 
using another smaller network `TinyNet`
    ↓
Understand chain rule


Lab 07
Manually perform an optimizer.step()
    ↓
update weights


Lab 08
Return to standard PyTorch style
    ↓
understand what PyTorch is doing for us

Lab 09
Runtime device detection and
moving the model and its data to that device.
```

### Lab 01: nn.Sequential->understand basic training loop

See [`Lab01_SineFunctionApproximation_nn_Sequential.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab01_SineFunctionApproximation_nn_Sequential.ipynb).

The `SineNet` architecture:

```text
             32 neurons
          ┌───────────────┐
          │   Tanh        │
          └───────────────┘
                 │
                 ▼
x ──► Linear(1,32) ──► Linear(32,32) ──► Linear(32,1) ──► ŷ
                 │             │
                Tanh          Tanh
```

Or more compactly: `1 → 32 → 32 → 1`

This is actually a 3-layer neural network if we're counting _trainable_ Linear layers.

We chose `Tanh` rather than `ReLU` deliberately. Our target function is smooth and oscillatory:

$$ y=\sin(x) $$

`Tanh` gives the network a smooth nonlinear basis, whereas `ReLU` produces piecewise-linear functions. `ReLU` can certainly approximate sine, but `Tanh` is a particularly intuitive choice for this first curve-fitting experiment.

This neural network has only:

$$ (1\times32+32) + (32\times32+32) + (32\times1+1) =1153 $$

trainable parameters.

For this "simple" problem, we would not add dropout or weight decay initially. Regularization is useful, but introducing it before understanding the basic training dynamics would obscure what's happening.

Note, we start with 32 nodes in the hidden layer for illustration. Actually, the width of the hidden layer is a property of the network class and its effect on the capacity of the network is an interesting thing to inspect.

With the following set-up:

```python
# Number of samples
N = 5_000

# Width of hidden layer
hidden_layer_width = 128

Epochs = 20_000
```

Here's is the Training/Validation Loss:

![Training/Validation Loss](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_loss.png)

And here's the plot of learned function vs true Sine curve in [-100, 100] (Interpolation):

![Interpolation](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_learned-func-vs-true-curve_interpolation.png)


And lastly, learned function vs true Sine curve in [-1000, 1000] (Extrapolation)

![Extrapolation](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_learned-func-vs-true-curve_extrapolation.png)

#### Understand the Code

See [`Lab01_SineFunctionApproximation_nn_Sequential.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab01_SineFunctionApproximation_nn_Sequential.ipynb).

##### Training Loop

```python
predictions = model(x_train_norm)

loss = loss_fn(predictions, y_train)

optimizer.zero_grad()

loss.backward()

optimizer.step()
```

This is essentially the heart of PyTorch training.

**1. Forward pass**
```python
predictions = model(x_train_norm)
```

Our input goes through:

```text
x
 ↓
Linear
 ↓
Tanh
 ↓
Linear
 ↓
Tanh
 ↓
Linear
 ↓
ŷ
```

The model produces predictions:

$$ \hat y = f_\theta(x) $$

where $\theta$ represents all of the model's weights and biases.

**2. Calculate error**

```python
loss = loss_fn(predictions, y_train)
```

We're using mean squared error:

$$ MSE = \frac{1}{N}\sum_i(y_i-\hat y_i)^2 $$

For example:

```text
true y       predicted y
 0.80           0.72
-0.40          -0.31
 0.95           0.91
```

The loss summarizes the errors.

**3. Clear old gradients**

```python
optimizer.zero_grad()
```

PyTorch accumulates gradients by default.

So before computing the next gradient, we normally clear the previous ones.

**4. Backpropagation**

```python
loss.backward()
```

This is where autograd calculates:

$$ \frac{\partial L}{\partial \theta} $$

for every trainable parameter.

Conceptually:

```text
loss
 │
 ├── ∂loss/∂W3
 ├── ∂loss/∂b3
 ├── ∂loss/∂W2
 ├── ∂loss/∂b2
 ├── ∂loss/∂W1
 └── ∂loss/∂b1
```

PyTorch builds the computational graph during the forward pass and automatically performs the chain rule during `.backward()`.

**5. Update parameters**

```python
optimizer.step()
```

An optimizer such as `Adam` then changes the parameters using the gradients.

Conceptually:

$$ \theta \leftarrow \theta-\eta\nabla_\theta L $$

`Adam` is more sophisticated than vanilla gradient descent, but that's the basic idea.

And then we repeat this thousands of times.


##### PyTorch Mental Model

```text
PyTorch model
      │
      ├── Parameters
      │      W1, b1, W2, b2, ...
      │
      ├── Forward pass
      │      x → model(x) → prediction
      │
      ├── Loss
      │      prediction vs target
      │
      ├── Autograd
      │      loss.backward()
      │
      └── Optimizer
             optimizer.step()
```

The key thing to internalize is that `nn.Module`, `autograd`, `loss functions`, and `optimizers` are separate pieces.

If we compare this to a Transformer, we will see conceptually the same training machinery:

```text
Transformer
     ↓
forward()
     ↓
logits
     ↓
CrossEntropyLoss
     ↓
loss.backward()
     ↓
optimizer.step()
```

The thing that changes is the complexity of `forward()` and the shape/meaning of the tensors—not the fundamental training loop.

We'd take this exact sine example and implement it from scratch without `nn.Sequential`, explicitly writing the `forward()` equations in our next lab and later inspecting `model.parameters()`, tensor shapes, `requires_grad`, `.grad`, and the computational graph in the progression. That will give us a very solid foundation.

##### `model.train()` vs `model.eval()`

**1. What do model.train() and model.eval() actually do?**

These two calls switch the model between training mode and evaluation/inference mode.

They do not themselves train the model, calculate gradients, or update weights.

`model.train()` means:

     "Set all modules inside this model to training behavior."

And:

`model.eval()` means:

     "Set all modules inside this model to evaluation behavior."

This matters because some neural-network layers behave differently during training and inference.

We will use the following two classic examples to help understand what this means.

**Dropout**

Suppose we have:

```python
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(20, 1)
)
```

During training, `model.train()`:

Dropout randomly shuts off approximately 50% of the neurons.
```text
      ●   ●   ●   ●   ●
      ↓   ↓   ↓   ↓   ↓
      ●   X   ●   X   ●
          ↑       ↑
       dropped  dropped
```

This is intentional regularization.

During evaluation, `model.eval()`:

Dropout is disabled:
```text
      ●   ●   ●   ●   ●
      ↓   ↓   ↓   ↓   ↓
      ●   ●   ●   ●   ●
```

So we don't get random predictions every time we evaluate the model.

**BatchNorm**

Batch normalization is another important example.

During training, BatchNorm uses statistics from the current mini-batch and maintains running statistics.

During evaluation, it uses those accumulated running statistics instead.

Therefore, `model.train()` and model.eval() can materially change the behavior of a model containing these layers.

But our `SineNet` has neither `Dropout` nor `BatchNorm`.

Our network is:
```Python
nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)
```

It contains only `Linear` and `Tanh`. Neither behaves differently between training and evaluation.

Therefore, in this particular example, `model.train()`
and `model.eval()` essentially have no observable effect.

We included them because it's good PyTorch practice to establish the pattern we'll use with real models.

**2. eval() does NOT mean "turn off gradients"**

This is a very important distinction.

We often see:

```python
model.eval()

with torch.no_grad():
    predictions = model(x)
```

These are doing **two different things**.

`model.eval()` changes the behavior of layers such as Dropout and BatchNorm.

`torch.no_grad()` tells autograd:

     "Don't construct a gradient computation graph for these operations."

So, `model.eval()` does not mean: "_don't calculate gradients_" and `torch.no_grad()` does not mean: "_switch the model to evaluation mode_". They are independent concepts. 

For inference, we commonly use both:

```python
model.eval()

with torch.no_grad():
    predictions = model(x)
```

#### Testing on [-1000, 1000]: Extrapolation vs Interpolation

There's something particularly interesting about this experiment:

```text
training: (-100,100)
testing: (-1000,1000)
```

That sounds like a normal train/test experiment, but mathematically it's not really the usual definition of generalization.

This is our training data:

```text
-100 ---------------------- 100
              ↑
         training region
```

and we test:

```text
-1000 ------------ -100 -------- 100 ----------------- 1000
       |---------|                   |---------|
       extrapolation    training     extrapolation
```

The model has never seen values such as:
```text
200
500
800
-400
-900
```

So we're asking it to extrapolate the function.

That's a much harder problem.

And there's an even deeper issue.

A finite neural network doesn't inherently know:

    "The underlying function is periodic and therefore this pattern should continue forever."

It only sees examples and learns an approximation.

Consequently, we may see something like:

```text
              neural network
             /¯¯¯¯¯¯¯¯¯¯¯¯¯\
 true sin   /\/\/\/\/\/\/\/\/\
           -100               100
```

inside the training region, but outside it the neural network can go wild.

This is an extremely useful experiment because it demonstrates an important ML distinction:

interpolation ≠ extrapolation.

Then, more generally, we may want to ask "_how about the ability of the model we trained to perform well on any unseen input?_". To answer this, we have to distinguish between `same-distribution generalization` and `out-of-distribution (OOD) generalization`.

In ML, "generalization" typically means performance on unseen data drawn from the same distribution as training data. In our case, training data is sampled from [-100, 100], then testing on held-out points from [-100, 100] is _generalization_ in the ML sense — we're testing whether the model generalizes to new samples from the same distribution. Mathematically, this means we're interpolating — the test points lie inside the convex hull / range of the training inputs.

Thus, testing on [-1000, -100] ∪ [100, 1000] is OOD. From this pespective, in ML sense, extrapolation ≠ generalization.

That is actually a nice first lesson in neural networks:
     
"A low test loss on samples drawn from the same distribution as the training data does not imply reliable extrapolation outside that distribution."


But what does "universal approximation theorem (UAT)" go about this? Loosely speaking, doesn't it say that there exists a neural network that can approximate any function? This is on the right track but not quite accurate.

A better intuition of UAT can be described as:
    Loosely speaking, for any continuous function on a closed and bounded domain, there exists a neural network (with enough neurons) that can approximate it arbitrarily well on that domain. It guarantees the network has the capacity to represent the function, but it says nothing about whether training will find it, or how it behaves outside the domain.

Tie-back to our Sine funtion approximation:

- Our training region `[-100, 100]` is both closed and bounded (thus a compact set as stated in UAT). UAT guarantees a network exists that approximates $sin⁡(x)$ well on this interval.
- Testing on `[−1000,−100] ∪ [100,1000]`, this is outside the compact domain where UAT gives guarantees. The theorem is silent about behavior there. Extrapolation is unaddressed.
- Testing on [−100,100] at unseen points, still within the compact domain; this is covered by UAT.

We will stop discussion on this topic here. For rigorous theoretical treatment of it, please Google "universal approximation theorem" or "function approximation".

Here again are the two plots that show learned curves for interpolation and extrapolation from above.

![Interpolation](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_learned-func-vs-true-curve_interpolation.png)

![Extrapolation](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_learned-func-vs-true-curve_extrapolation.png)

Our model is optimizing:

$$ \min_\theta \frac{1}{N}\sum_{x_i\in[-100,100]} \left(f_\theta(x_i)-\sin(x_i)\right)^2 $$

It is not optimizing:

$$ \min_\theta \frac{1}{N}\sum_{x_i\in[-1000,1000]} \left(f_\theta(x_i)-\sin(x_i)\right)^2 $$

So the optimizer has absolutely no reason to care about what happens outside [-100,100], which is exactly where UAT does not cover.


Our network ultimately represents something like:

$$ f(x)=W_3 \tanh(W_2\tanh(W_1x+b_1)+b_2)+b_3 $$

That's a composition of `tanh` functions.

But `Sine` is:

$$ \sin(x) $$

These are fundamentally different functional forms.

The neural network can approximate Sine extremely well over a finite interval. But nothing in the architecture says: _"I should be periodic."_

That's why we can get:

```text
true:       ~~~~~\/\/\/\/\/\/~~~~~
network:    ~~~~~\/\/\/\/\/\/~~~~~
```

inside training domain; while

```text
true:       /\/\/\/\/\/\/\/\/\/\/\
network:    ────────╮
                    ╰──────────────
```
for the outside.


The network has learned a good local approximation, rather than the true global mathematical structure.

#### Another Subtle Issue with Normalization Mismatch between Training and Testing

Our test range is:

$$ [-1000,1000] $$

while training is:

$$ [-100,100] $$

That's a `10×` larger range in each direction.

And remember that we normalized `x_norm = x / 100` during training. So the network actually sees:

During training
```math
x_{\text{norm}}\in[-1,1]
```

During extrapolation
```math
x_{\text{norm}}\in[-10,10]
```

Now think about Tanh.

Its behavior is approximately:
```text
                 1 ─────────────────
                  /
                 /
                /
───────────────
              /
             /
            /
          -1 ─────────────────
```

For large positive or negative values:

$$\tanh(x)\rightarrow1$$

and

$$\tanh(x)\rightarrow-1$$

`tanh` units may saturate, so the result for input that is OOD can look especially misleading. This makes the extrapolation behavior particularly poor and interesting.

#### Things to Inspect

From this experiment and above discussion, we should have a basic grasp of how to build a neural network, train, test, and assess our observation from experiment results. Hopefully, we will have answers to the three questions we brought up at the beginning:

```text
1. Function approximation
       │
       ▼
"Can a neural network represent this function?"

2. Generalization
       │
       ▼
"Does it work on unseen data
 from the same distribution?"

3. Extrapolation
       │
       ▼
"Does it work outside the distribution
 represented by the training data?"
```

And there's a particularly interesting experiment we can do next: keep exactly the same network but progressively increase the hidden-layer width (4 → 8 → 16 → 32 → 64 → 128) and watch training loss, validation loss, parameter count, and the learned curve. That would give us an intuitive, empirical feel for what "universal approximation" actually means rather than treating it as just a theorem.

#### Approximate sin(x) on [-100, 100] with Fourier Features and an MLP

To make our first lab complete with more fun, we include an implementation of fitting Sine wave in [-100, 100] using Fourier features and an `MLP` with an extra hidden layer than our existing MLP only architecture.

See [`Lab01_SineFunctionApproximation_FourierFeatures_MLP.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab01_SineFunctionApproximation_FourierFeatures_MLP.ipynb).

Here's the new neural network:

```text
SineNet(
  (network): Sequential(
    (0): FourierFeatures()
    (1): Linear(in_features=17, out_features=128, bias=True)
    (2): Tanh()
    (3): Linear(in_features=128, out_features=128, bias=True)
    (4): Tanh()
    (5): Linear(in_features=128, out_features=128, bias=True)
    (6): Tanh()
    (7): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

With only 1000 samples (800 for training and 200 for validation) and training for 3000 epochs, we already got fair results:

Loss:

```text
Epoch   500 | Train Loss: 0.000035 | Val Loss: 0.000047
Epoch  1000 | Train Loss: 0.000011 | Val Loss: 0.000014
Epoch  1500 | Train Loss: 0.000005 | Val Loss: 0.000006
Epoch  2000 | Train Loss: 0.000002 | Val Loss: 0.000003
Epoch  2500 | Train Loss: 0.000013 | Val Loss: 0.000005
Epoch  3000 | Train Loss: 0.000001 | Val Loss: 0.000001
```

For fun, we tested the model both for `interpolation ([-100, 100])` and `extrapolation ([-1000, 1000])`. As can be seen below, no surprise, MSE from outside training domain is much higher compared to inside training domain.

```text
Evaluation on [-100, 100]
--------------------------
MSE: 0.00000126

Evaluation on [-1000, 1000]
--------------------------
MSE: 1.00146711
```

The interpolation curve and a series of extrapoloation curves against the true Sine wave are depicted in the following plots respectively, (a): [-100, 100], (b): [-150, 150], (c):[-200, 200], (d): [-300, 300], (e): [-500, 500], and (f): [-1000, 1000]). 

We see a nearly perfect match for the training domain but OOD generalization is getting poorer further away from the training domain. 

That said, we do see that this implementation seem to capture to some degree the oscillation patterns in the periodic Sine wave outside the training domain. 

As a comparison, we also include an extrapolation for [-150, 150] from the MLP only neural network in the last plot (g). It is clear it abruptly loses the osillation nature of Sine wave once it leaves the training domain.


![Interpolation [-100, 100]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-interpolation_N1K-Epoch3K-Width128.png)

```math
(a)
```

![Extrapolation [-150, 150]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-extrapolation-150_N1K-Epoch3K-Width128.png)

```math
(b)
```


![Extrapolation [-200, 200]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-extrapolation-200_N1K-Epoch3K-Width128.png)

```math
(c)
```


![Extrapolation [-300, 300]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-extrapolation-300_N1K-Epoch3K-Width128.png)

```math
(d)
```

![Extrapolation [-500, 500]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-extrapolation-500_N1K-Epoch3K-Width128.png)

```math
(e)
```

![Extrapolation [-1000, 1000]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Fourier-feature_neural_approximation_of_sine-function_learned-curve-extrapolation_N1K-Epoch3K-Width128.png)

```math
(f)
```


![MLP only Extrapolation [-150, 150]](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/N5K_3TrainableLayer_Width128_Epoch20K_learned-func-vs-true-curve_extrapolation-150.png)

```math
(g)
```

Before we wrap up our first lab, here is a reading list for properly implementing Sine Funciton Approximation:

Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains
— Matthew Tancik et al., NeurIPS 2020.
https://papers.neurips.cc/paper_files/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html

Implicit Neural Representations with Periodic Activation Functions
— Vincent Sitzmann et al., NeurIPS 2020.
https://papers.neurips.cc/paper_files/paper/2020/hash/53c04118df112c13a8c34b38343b9c10-Abstract.html
  
Neural Networks Fail to Learn Periodic Functions and How to Fix It
— Liu Ziyin et al., NeurIPS 2020
https://proceedings.neurips.cc/paper/2020/hash/1160453108d3e537255e9f7b931f4e90-Abstract.html

### Lab 02: replace nn.Sequential with manual construction of the network from nn.Linear and tanh -> understand what nn.Sequential and forward() is hiding

In this lab, we will keep exactly the same neural network mathematically, but stop letting  `nn.Sequential` hide the individual operations from us. This is a very good next exercise because we're learning PyTorch, not just learning how to call a high-level API.

Our current model is: 

```python
class SineNet(nn.Module):

    def __init__(self, hidden_layer_width=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, hidden_layer_width),
            nn.Tanh(),

            nn.Linear(hidden_layer_width, hidden_layer_width),
            nn.Tanh(),

            nn.Linear(hidden_layer_width, 1)
        )

    def forward(self, x):
        return self.network(x)

```

Here, `nn.Sequential` is doing quite a lot behind the scenes:

```text
x
 │
 ▼
Linear(1,128)
 │
 ▼
Tanh
 │
 ▼
Linear(128,128)
 │
 ▼
Tanh
 │
 ▼
Linear(128,1)
 │
 ▼
ŷ
```

We'd like to expose that explicitly.

So, we remove `nn.Sequential` and manually write the `forward` pass:

```python
class SineNet(nn.Module):

    def __init__(self, hidden_layer_width=128):
        super().__init__()

        self.hidden_layer_1 = nn.Linear(1, hidden_layer_width)
        self.activation_layer_1 = nn.Tanh()
        self.hidden_layer_2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.activation_layer_2 = nn.Tanh()
        self.output_layer = nn.Linear(hidden_layer_width, 1)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = self.activation_layer_1(x)
        x = self.hidden_layer_2(x)
        x = self.activation_layer_2(x)
        x = self.output_layer(x)
        return x

```

Notice that the architecture has not changed at all. We're simply making the computation explicit.

But there is yet another way to rewrite this as follows:


```python
class SineNet(nn.Module):

    def __init__(self, hidden_layer_width=128):
        super().__init__()

        self.hidden_layer_1 = nn.Linear(1, hidden_layer_width)
        self.hidden_layer_2 = nn.Linear(hidden_layer_width, hidden_layer_width)
        self.output_layer = nn.Linear(hidden_layer_width, 1)

    def forward(self, x):
        x = self.hidden_layer_1(x)
        x = torch.tanh(x)
        x = self.hidden_layer_2(x)
        x = torch.tanh(x)
        x = self.output_layer(x)
        return x

```

Functionally, there is no difference between these two rewrites — the two networks are mathematically identical. They produce the same output for the same input and have the same parameters. But there are subtle implementation-level differences worth knowing.

**1. Parameter and computation equivalence**

Both have the same three `nn.Linear` layers with identical shapes.

Both apply `tanh` after each of the first two linear layers.

The forward pass computes exactly the same function: $ŷ=W_3 \tanh(W_2\tanh(W_1x+b_1)+b_2)+b_3$

Given the same initialization and inputs, the outputs and gradients will be bit-for-bit identical (assuming the same seed and device/dtype).

**2. `nn.Tanh()` vs `torch.tanh`**

`nn.Tanh()` is a module (a subclass of `nn.Module`). It's stateless (no parameters), but it gets registered in the module tree.

`torch.tanh` is a function — a direct call to the `tanh` operation.

Numerically identical. Both compute the same thing.

**3. Differences in practice**
```text
Aspect	                    SineNet (nn.Tanh)                             SineNet (torch.tanh)
Appears in model.modules()	Yes — one extra module per activation	      No
Appears in state_dict()	    No (no parameters)                            No
Shows in print(model)       Yes (as Tanh())                    	          No
Overhead                    Negligible (module call)	                  Slightly less (direct function)
Flexibility                 Can swap/replace as a module	              Harder to swap programmatically
Style                       More "PyTorch idiomatic" for reusable blocks  More concise for simple cases
```

**4. Why we might prefer one over the other**

- `nn.Tanh()`:

  - More consistent with the module-based style of PyTorch.

  - Easier to replace programmatically (e.g., loop over `model.modules()` and swap activations, or use `nn.Sequential)`.

  - Participates in the module hierarchy, so it appears in `print(model)`, which is nice for debugging/inspection.

  - Standard choice when building reusable blocks (e.g., in `nn.Sequential`).

- `torch.tanh`:

  - Slightly more concise.

  - Marginally faster (avoids module call overhead — usually negligible).

  - Fine for quick scripts or when the activation is used once.

**5. A subtle gotcha**

If we ever do something like:

```python
for m in model.modules():
    if isinstance(m, nn.Tanh):
        # replace with something else
```

…this works for `SineNet (nn.Tanh)` but not `SineNet (torch.tanh)`, because `torch.tanh` isn't a module. So for introspection, swapping, or serialization-based tooling, `nn.Tanh` is more convenient.

**6. Training behavior**

- Identical. Same gradients, same optimization dynamics, same results (given the same seed, device/dtype).

- `nn.Tanh` and `torch.tanh` have the same forward and backward implementations under the hood.

**Bottom line:**

The two implementations are **mathematically and numerically equivalent**. The only real differences are **structural/introspective** — `SineNet (nn.Tanh)` registers tanh as modules (visible in `model.modules()`, `print(model)`), while `SineNet (torch.tanh)` uses the raw function (invisible to those tools). For training, they behave identically. Choose based on style and whether we need to inspect/swap activations programmatically.



### Lab 03: manually implement Linear layers with nn.Parameter -> understand tensor, parameter, weights, biases, matrix multiplication

In this lab, we go one step deeper and implement the Linear layers ourselves.

Remember what a linear layer does:

$$ y = xW^T+b $$

So instead of:

```python
self.layer1 = nn.Linear(1, 32)
```

we could explicitly create:

```python
self.W1 = nn.Parameter(torch.randn(32, 1))
self.b1 = nn.Parameter(torch.zeros(32))
```

and then `z1 = x @ self.W1.T + self.b1`

That's the actual mathematical operation occurring inside `nn.Linear`.

Then the entire network becomes:

```python
class SineNet(nn.Module):

    def __init__(self, hidden_layer_width=128):
        super().__init__()

        self.W1 = nn.Parameter(torch.randn(hidden_layer_width, 1))
        self.b1 = nn.Parameter(torch.zeros(hidden_layer_width))

        self.W2 = nn.Parameter(torch.randn(hidden_layer_width, hidden_layer_width))
        self.b2 = nn.Parameter(torch.zeros(hidden_layer_width))

        self.W3 = nn.Parameter(torch.randn(1, hidden_layer_width))
        self.b3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        # Layer 1
        z1 = x @ self.W1.T + self.b1
        a1 = torch.tanh(z1)

        # Layer 2
        z2 = a1 @ self.W2.T + self.b2
        a2 = torch.tanh(z2)

        # Output layer
        y = a2 @ self.W3.T + self.b3

        return y
```

So, this is a manually parameterized version of SineNet (torch.tanh). Now we can literally see the neural network mathematics. See [`Lab03_SineFunctionApproximation_nn_parameter.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab03_SineFunctionApproximation_nn_parameter.ipynb).

1. And this connects directly to implmenting advanced architecture such as Transformers

This is actually why we recommended the exercise. When we study the Transformer equations, we'll encounter things like:

$$ Q=XW_Q $$ $$ K=XW_K $$ $$ V=XW_V $$ $$ Attention(Q,K,V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Rather than thinking "PyTorch has an `Attention` function that magically does something.", we want to develop the ability to think, "what tensors are being multiplied? what are their shapes? what parameters are being used? what operations are recorded by autograd?"

Our manually parameterized MLP version is a perfect environment for developing that intuition.

**2. Initialization Matters**

With 5000 samples (8:2 training validation split ratio), 20000 epochs and a width of 128 for the hidden layer, we see the following evaluation result:

```text
Interpolation MSE: 0.326193
Extrapolation MSE: 37.045727
```

This initialization may be suboptimal. `torch.randn(128, 1)` can push `tanh` units toward saturation — especially with inputs in [-100, 100] (normalizing inputs by `1/100` only solves the issue partially since this doesn't touch weights; so, large weights remain large).

So, we make the following improvement to our weight initialization (See [`Lab03_SineFunctionApproximation_nn_parameter_xavier.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab03_SineFunctionApproximation_nn_parameter_xavier.ipynb)):

- Tanh layers: Xavier uniform initialization with the appropriate tanh gain.
- Output layer: Xavier uniform initialization with gain 1.0.
- All biases: initialized to zero.

```python
tanh_gain = nn.init.calculate_gain("tanh")

nn.init.xavier_uniform_(self.W1, gain=tanh_gain)
nn.init.xavier_uniform_(self.W2, gain=tanh_gain)
nn.init.xavier_uniform_(self.W3, gain=1.0)

nn.init.zeros_(self.b1)
nn.init.zeros_(self.b2)
nn.init.zeros_(self.b3)
```

Xavier uniform initialization uses

  $$ W_{ij} \sim U(-a,a) $$

where
```math
a =
  \text{gain} \times
  \sqrt{\frac{6}{\text{fan\_in}+\text{fan\_out}}}  
```

For `tanh`, PyTorch recommends

  $$ \text{gain} = \frac{5}{3} $$

This is intended to preserve signal and gradient scales across layers with both incoming and outgoing widths ([PyTorch xavier_uniform_ documentation](https://docs.pytorch.org/docs/stable/nn.init#torch.nn.init.xavier_uniform_))

**Comparison**

```text
   Initialization           Weight scale depends on       Activation assumption
  ━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  
   nn.Linear default        fan_in only                   Generic default
  ───────────────────────  ────────────────────────────  ─────────────────────────────────
   Xavier, gain 1           fan_in + fan_out              Linear-like activation
  ───────────────────────  ────────────────────────────  ────────────────────────────
   Xavier, tanh gain 5/3    fan_in + fan_out              Tanh
  ───────────────────────  ────────────────────────────  ─────────────────────────────
   torch.randn(...)         Fixed standard deviation 1    No scale adaptation
```

For our network (when hidden_layer_width=32), the approximate uniform bounds are:
```text
   Layer      nn.Linear default    Xavier with tanh gain
  ━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━
   1 → 32                ±1.000                   ±0.711
  ─────────  ───────────────────  ───────────────────────
   32 → 32               ±0.177                   ±0.510
  ─────────  ───────────────────  ───────────────────────
   32 → 1                ±0.177       ±0.426 with gain 1
```

The exact values are random samples, but these are the distribution limits.

The important point is that `Xavier` with `tanh gain` is not necessarily numerically smaller than `nn.Linear` initialization for every layer. Its purpose is to account for both `fan_in` and `fan_out`, and the `tanh gain` deliberately increases the scale to compensate for `tanh`’s compressive behavior.

This improves the evaluation significantly:

```text
Interpolation MSE: 0.106765
Extrapolation MSE: 0.784248
```

To understand why this improvement, let's take a look at `W1` (shape (128, 1)):

- fan_in = 1, fan_out = 128
- Xavier $std = \sqrt{2/(1+128)} ≈ 0.1245$
- Our earlier `torch.randn(128, 1)` gives `std = 1` — about `8×` larger than Xavier recommends.
  
Combined with inputs in [-100, 100], that's a recipe for immediate `tanh` saturation. This is exactly the kind of problem proper fan-in-based init is designed to prevent.


As a comparison, our first version `Lab01_SineFunctionApproximation_nn_Sequential.ipynb`, which is built with `nn.Linear`, gives us the following evaluation result:

```text
Interpolation MSE: 0.001071
Extrapolation MSE: 1.035617
```

PyTorch's `nn.Linear` uses `Kaiming uniform` for its weight initialization: `nn.init.kaiming_uniform_(weight, a=math.sqrt(5))`, which may behave generally well (a sure win over the Xavier setup used in this experiment). (PyTorch initialization documentation, https://docs.pytorch.org/docs/stable/nn.init)


### Lab 04: Model Inspection - Shapes, Parameters, and ternsor operations

**1. View Model**

After creating `model = SineNet()`, we can normally call `print(model)` to view the network of the model such as the following:

```text
SineNet(
  (network): Sequential(
    (0): Linear(in_features=1, out_features=128, bias=True)
    (1): Tanh()
    (2): Linear(in_features=128, out_features=128, bias=True)
    (3): Tanh()
    (4): Linear(in_features=128, out_features=1, bias=True)
  )
)
```

But for manually parameterized model using `nn.Parameter` in our last lab, `print(model)` will show something like `SineNet()`.

That's it — no layer details, because `print(model)` only displays registered child modules while we defined our model as `nn.Paramter` objects, not `nn.Module` objects. So, Pytorch has no layer hierarchy to print.

If we do want a readable architecture description for our manually parameterized MLP model, we can add a custom __repr__ method that looks like:

```python
def __repr__(self):
    return (
        "SineNet(\n"
        "  (hidden_layer_1): Linear(1, 32)\n"
        "  (activation_1): Tanh()\n"
        "  (hidden_layer_2): Linear(32, 32)\n"
        "  (activation_2): Tanh()\n"
        "  (output_layer): Linear(32, 1)\n"
        ")"
    )
```
Then:

```python
model = SineNet()
print(model)
```
will display that text.

Afterall, the only registered things in the current version of `SineNet` are Parameters, we should inspect them.

**2. View Parameters**

We can inspect the parameters:

```python
print(model)
print(list(model.named_parameters()))

for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

A clearer version is:

```python
for name, parameter in model.named_parameters():
    print(f"{name}: shape={tuple(parameter.shape)}, "
          f"trainable={parameter.requires_grad}")
```

This will show:

```text
W1: shape=(32, 1), trainable=True
b1: shape=(32,), trainable=True
W2: shape=(32, 32), trainable=True
b2: shape=(32,), trainable=True
W3: shape=(1, 32), trainable=True
b3: shape=(1,), trainable=True
```
And suddenly the parameter count we discussed earlier becomes very concrete: 32+32+32*32+32+32+1=1153, which will match what we will find out in the following.

We can also print the total number of trainable parameters:
```python
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print(f"Trainable parameters: {total_parameters}")
```

Alternatively, we can call `model.state_dict()` to view an OrderedDict (an ordered dictionary) that maps parameter and buffer names to their tensors. It's essentially a snapshot of the model's learnable parameters (and non-learnable buffers) at the current moment.

E.g.

```python
model = SineNet()
print(model)
print(model.state_dict())
```
will print out something as follows:

```text
OrderedDict([('W1',
              tensor([[ 0.6320],
                        ...
                      [-0.1337]])),
             ('b1',
              tensor([0., ..., 0.])),
             ('W2',
              tensor([[-0.4023, ...,  0.4723]])),
             ('b2',
              tensor([0., ..., 0.])),
             ('W3',
              tensor([[ 0.0802, ..., -0.3107]])),
             ('b3', tensor([0.]))])

```


`model.state_dict()` gives us a flat, ordered dictionary of {name: tensor} for all parameters and buffers. It's the canonical way to inspect, save, and load the learned values of a model — but it says nothing about the model's architecture, which lives in the class definition itself.

**3. Inspect the tensors during forward()**

We'd also temporarily put `print(x.shape)` inside the `forward` pass.

For example:

```python
def forward(self, x):

    print("input:", x.shape)

    z1 = x @ self.W1.T + self.b1
    print("z1:", z1.shape)

    a1 = torch.tanh(z1)
    print("a1:", a1.shape)

    z2 = a1 @ self.W2.T + self.b2
    print("z2:", z2.shape)

    a2 = torch.tanh(z2)
    print("a2:", a2.shape)

    y = a2 @ self.W3.T + self.b3
    print("output:", y.shape)

    return y
```

For a batch of 800 examples:

```text
input: torch.Size([800, 1])
z1: torch.Size([800, 32])
a1: torch.Size([800, 32])
z2: torch.Size([800, 32])
a2: torch.Size([800, 32])
output: torch.Size([800, 1])
```

This is extremely valuable PyTorch knowledge.

We start seeing a neural network not as a mysterious object, but as a sequence of tensor transformations.

**4. Division of responsibility**

We implement:
```text
    forward()
       │
       ▼
tensor operations
       │
       ▼
      loss
```       
PyTorch autograd implements:
```text
      loss.backward()
             │
             ▼
       chain rule
             │
             ▼
       gradients
             │
             ▼
     optimizer.step()
             │
             ▼
       new parameters
```
So we're manually implementing the model, while PyTorch handles the automatic differentiation and optimization.

That's exactly the abstraction boundary we need to understand.

**5. `loss.backward()`, one level deeper**

After we're comfortable with the above, we could take one training step and manually calculate the gradient for this tiny network.

For example, start with:

$$ z_1=xW_1^T+b_1 $$ $$ a_1=\tanh(z_1) $$ $$ z_2=a_1W_2^T+b_2 $$ $$ a_2=\tanh(z_2) $$ $$ \hat y=a_2W_3^T+b_3 $$ $$ L=(\hat y-y)^2 $$

Then trace:

$$ \frac{\partial L}{\partial W_3} $$

back through:

$$ \hat y \rightarrow a_2 \rightarrow z_2 \rightarrow a_1 \rightarrow z_1 \rightarrow W_1. $$

We'd see the chain rule physically operating through the neural network.

We don't need to do this for practical PyTorch programming, but doing it once with a tiny network is enormously useful for understanding what `loss.backward()` actually means.

So, that's the idea. We will get our hands dirty on this in our next lab, `Lab 05`: we will build a small neural network `SmallNet`, run a single example through the network, inspect $\hat y$, loss, every parameter, every .grad, and their shapes. 

Then in `Lab 06`, we will create a even smaller network `TinyNet` and use it to work through the same process but only with paper and pencil this time. 

Once we've done that, the next very revealing exercise is to manually perform one `optimizer.step()` ourself using the gradients we just calculated, and then let PyTorch do it and verify that the new parameters match. That's what we will cover in `Lab 07`. Stay tuned.

### Lab 05: One pass through `SmallNet`: forward →  backward → weight updating

We are going to continue our inspection journey from Lab 04.

First, We'll start with our manually defined network, but make it small.

Next, using this neural network, we let PyTorch perform one forward + backward pass, then inspect everything it calculated.

Then, we take the same tiny example, put the numbers on paper, and manually reproduce the gradients using the chain rule.

Lastly, compare our hand calculations with PyTorch's .grad.

See [`Lab05_inspection_one_example_forward_loss_backward_optimizer.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab05_inspection_one_example_forward_loss_backward_optimizer.ipynb).

Let's roll our sleeves and get started.

We will first create our small neural network `SmallNet`: `1 → 2 → 2 → 1`, i.e.:

```text
          Hidden 1       Hidden 2
          2 neurons      2 neurons

x ──────► [ 2 ] ───────► [ 2 ] ───────► y
             Tanh           Tanh
```

**1. `SmallNet`**

```python
import torch
import torch.nn as nn

torch.manual_seed(42)


class SmallNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.W1 = nn.Parameter(torch.randn(2, 1))
        self.b1 = nn.Parameter(torch.zeros(2))

        self.W2 = nn.Parameter(torch.randn(2, 2))
        self.b2 = nn.Parameter(torch.zeros(2))

        self.W3 = nn.Parameter(torch.randn(1, 2))
        self.b3 = nn.Parameter(torch.zeros(1))

    def forward(self, x):

        z1 = x @ self.W1.T + self.b1
        a1 = torch.tanh(z1)

        z2 = a1 @ self.W2.T + self.b2
        a2 = torch.tanh(z2)

        y_hat = a2 @ self.W3.T + self.b3

        return y_hat
```


Notice that there are only six parameter tensors:
```text
W1, b1
W2, b2
W3, b3
```

**2. Create _ONE_ training example**

For learning backpropagation, let's forget about 1,000 examples.

Just use:

```python
x = torch.tensor([[0.5]])
y = torch.tensor([[0.8]])
```

So we're asking:

  Given x = 0.5, the correct answer is y = 0.8. What should the network's weights do to reduce the error?

**3. Create the model and inspect its parameters**

```python
model = SmallNet()

for name, parameter in model.named_parameters():
    print(name)
    print(parameter)
    print()
```

We'll see something like:

```text
W1
tensor([[ 0.3367],
        [ 0.1288]])

b1
tensor([0., 0.])

W2
tensor([[ 0.2345,  0.2303],
        [-1.1229, -0.1863]])

b2
tensor([0., 0.])

W3
tensor([[2.2082, -0.6380]])

b3
tensor([0.])
```

The exact numbers will depend on PyTorch's random initialization/version, so don't worry if yours differ.

**4. Do ONE forward pass**

```python
y_hat = model(x)

print("prediction:", y_hat)
```

Conceptually:
```text
x
 ↓
z1 = xW1ᵀ + b1
 ↓
a1 = tanh(z1)
 ↓
z2 = a1W2ᵀ + b2
 ↓
a2 = tanh(z2)
 ↓
ŷ = a2W3ᵀ + b3
```

**5. Calculate the loss**

Let's use MSE, but with only one example:

```python
loss_fn = nn.MSELoss()

loss = loss_fn(y_hat, y)

print("prediction:", y_hat.item())
print("target:    ", y.item())
print("loss:      ", loss.item())
```

For a single scalar prediction:

$$ L=(\hat y-y)^2 $$

So we can actually manually calculate the loss with a calculator.

For example, if:
```python
ŷ = 0.42
y = 0.80
```
then:

$$ L=(0.42-0.80)^2 $$ $$ L=(-0.38)^2=0.1444 $$

**6. inspect `.grad` before `backward()`**

Try `print(model.W1.grad)` and we'll get: `None`

This is important. Why? Because we haven't calculated the gradients yet.

**7. Call `loss.backward()`**

Now, call `loss.backward()`, which tells PyTorch:

  Starting from this loss, calculate the derivative of the loss with respect to every trainable parameter that contributed to it.

Now do:

```python
for name, parameter in model.named_parameters():
    print(name)
    print("value:")
    print(parameter)
    print("gradient:")
    print(parameter.grad)
    print()
```

We'll see something like:
```text
W1
value:
Parameter containing:
tensor([[-1.0438],
        [ 0.9254]], requires_grad=True)
gradient:
tensor([[-1.7848],
        [-1.5849]])

b1
value:
Parameter containing:
tensor([0., 0.], requires_grad=True)
gradient:
tensor([-3.5695, -3.1697])

W2
value:
Parameter containing:
tensor([[1.2633, 1.0577],
        [0.4512, 0.9771]], requires_grad=True)
gradient:
tensor([[ 1.7520, -1.5805],
        [ 0.0150, -0.0135]])

b2
value:
Parameter containing:
tensor([0., 0.], requires_grad=True)
gradient:
tensor([-3.6563, -0.0313])

W3
value:
Parameter containing:
tensor([[1.7681, 0.0155]], requires_grad=True)
gradient:
tensor([[ 0.3108, -0.4297]])

b3
value:
Parameter containing:
tensor([0.], requires_grad=True)
gradient:
tensor([-2.1137])
```

We have now directly observed what backpropagation produces.

**8. Look at the gradient shapes**

Try:
```python
for name, parameter in model.named_parameters():
    print(
        name,
        "parameter shape =", parameter.shape,
        "gradient shape  =", parameter.grad.shape
    )
```

We'll see:
```text
W1 parameter shape = torch.Size([2, 1]) gradient shape  = torch.Size([2, 1])
b1 parameter shape = torch.Size([2]) gradient shape  = torch.Size([2])
W2 parameter shape = torch.Size([2, 2]) gradient shape  = torch.Size([2, 2])
b2 parameter shape = torch.Size([2]) gradient shape  = torch.Size([2])
W3 parameter shape = torch.Size([1, 2]) gradient shape  = torch.Size([1, 2])
b3 parameter shape = torch.Size([1]) gradient shape  = torch.Size([1])
```

This is a very important principle:

  **Every trainable parameter gets a gradient of the same shape as itself.**

**9. What does one of these numbers mean?**

Suppose PyTorch tells us:
```text
W3 gradient:

tensor([[0.1234, -0.4567]])
```

That means:

$$ \frac{\partial L}{\partial W_{3,1}}=0.1234 $$

and

$$ \frac{\partial L}{\partial W_{3,2}}=-0.4567 $$

Interpretation:

  If I make a tiny positive change to this particular weight, how does the loss change?

Mathematically:

$$ \frac{\partial L}{\partial W} $$

is the local slope of the loss with respect to that parameter.

**10. Connect this to `optimizer.step()`**

Now we're ready to see why the optimizer exists.

Suppose:

$$ W=2.0 $$

and:

$$ \frac{\partial L}{\partial W}=0.5 $$

With simple gradient descent:

$$ W_{\text{new}} = W-\eta\frac{\partial L}{\partial W} $$

If:

$$ \eta=0.1 $$

then:

$$ W_{\text{new}} = 2.0-0.1(0.5) =1.95 $$

That's essentially what the optimizer is doing, although `Adam` uses a more sophisticated update rule.

So the complete process is:
```text
forward()
   │
   ▼
prediction
   │
   ▼
loss
   │
   ▼
loss.backward()
   │
   ▼
gradients
   │
   ▼
optimizer.step()
   │
   ▼
new weights
```

### Lab 06: manually work through one backpropagation -> understand chain rule using TinyNet

Now let's take a moment, pretend we are sitting in a middle school classroom, get our paper and pencil ready:

```text
Run one forward pass
  → print the numbers
    → take paper and pencil
      → calculate all derivatives
        → compare against PyTorch.
```

But we would make one change: **Don't use random weights**.

For the manual exercise, we'll deliberately choose very simple numbers. That makes the arithmetic manageable.

See [`Lab06_paper_pencil_through_one_backpropagation.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab06_paper_pencil_through_one_backpropagation.ipynb)

#### 1. Use an even smaller network

Let's temporarily use: `1 → 2 → 1`, which is enough to demonstrate the entire chain rule.
```text
             hidden
           2 neurons

x ───────► [ a₁   a₂ ] ───────► ŷ
              Tanh
```

The equations are:

$$ z_1=xw_1+b_1 $$ $$ a_1=\tanh(z_1) $$ $$ z_2=xw_2+b_2 $$ $$ a_2=\tanh(z_2) $$

Then:

$$ \hat y=a_1v_1+a_2v_2+b_3 $$

And we'll use:

$$ L=(\hat y-y)^2 $$

This is small enough that we can calculate every derivative by hand.

#### 2. Pick simple numbers

Let's use:

$$ x=1 $$

and target:

$$ y=1 $$

Parameters:

$$ w_1=0.5 $$ $$ b_1=0 $$ $$ w_2=-0.5 $$ $$ b_2=0 $$ $$ v_1=1 $$ $$ v_2=1 $$ $$ b_3=0 $$

So our network is:
```text
              tanh
          ┌───────┐
x ───────►│       │──── a₁ ───┐
          └───────┘           │
                              ├──► ŷ
          ┌───────┐           │
x ───────►│       │──── a₂ ───┘
          └───────┘
              tanh
```

#### 3. Forward pass on paper

First hidden neuron:

$$ z_1=xw_1+b_1 $$ $$ z_1=1(0.5)+0=0.5 $$

Therefore:

$$ a_1=\tanh(0.5)\approx0.4621 $$

Second neuron:

$$ z_2=xw_2+b_2 $$ $$ z_2=1(-0.5)+0=-0.5 $$

Therefore:

$$ a_2=\tanh(-0.5)\approx-0.4621 $$

Output:

$$ \hat y=a_1v_1+a_2v_2+b_3 $$ $$ \hat y=0.4621(1)+(-0.4621)(1)+0 $$

So:

$$ \boxed{\hat y=0} $$

Our target is:

$$ y=1 $$

Therefore:

$$ L=(\hat y-y)^2 $$ $$ L=(0-1)^2 $$ $$ \boxed{L=1} $$

Now comes the interesting part.

#### 4. Start backpropagation from the loss

Our computation graph is:
```text
w₁ ──► z₁ ──► a₁ ───┐
                    │
                    ├──► ŷ ──► L
                    │
w₂ ──► z₂ ──► a₂ ───┘
```

We want:

$$ \frac{\partial L}{\partial w_1} $$

The chain rule tells us:

$$ \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial\hat y} \frac{\partial\hat y}{\partial a_1} \frac{\partial a_1}{\partial z_1} \frac{\partial z_1}{\partial w_1} $$

This is the essence of backpropagation. We work backward.

#### 5. First derivative

We have:

$$ L=(\hat y-y)^2 $$

Therefore:

$$ \frac{\partial L}{\partial\hat y} = 2(\hat y-y) $$

We know:

$$ \hat y=0 $$

and:

$$ y=1 $$

Therefore:

$$ \frac{\partial L}{\partial\hat y} = 2(0-1) = \boxed{-2} $$

This is our starting gradient.

#### 6. Move one step backward

We have:

$$ \hat y=a_1v_1+a_2v_2+b_3 $$

Therefore:

$$ \frac{\partial\hat y}{\partial a_1}=v_1 $$

Since:

$$ v_1=1 $$

we have:

$$ \boxed{\frac{\partial\hat y}{\partial a_1}=1} $$

So:

$$ \frac{\partial L}{\partial a_1} = \frac{\partial L}{\partial\hat y} \frac{\partial\hat y}{\partial a_1} $$ $$ =(-2)(1) $$ $$ \boxed{\frac{\partial L}{\partial a_1}=-2} $$

#### 7. Go through Tanh

Now:

$$ a_1=\tanh(z_1) $$

The derivative of tanh is:

$$ \frac{d}{dz}\tanh(z) = 1-\tanh^2(z) $$

Since:

$$ a_1=\tanh(z_1)=0.4621 $$

we have:

$$ \frac{\partial a_1}{\partial z_1} = 1-a_1^2 $$ $$ =1-(0.4621)^2 $$ $$ \approx0.7864 $$

Therefore:

$$ \frac{\partial L}{\partial z_1} = \frac{\partial L}{\partial a_1} \frac{\partial a_1}{\partial z_1} $$ $$ =(-2)(0.7864) $$ $$ \boxed{\frac{\partial L}{\partial z_1}\approx-1.5728} $$

#### 8. Go through the linear operation

We have:

$$ z_1=xw_1+b_1 $$

Therefore:

$$ \frac{\partial z_1}{\partial w_1}=x $$

Since:

$$ x=1 $$

we get:

$$ \boxed{\frac{\partial z_1}{\partial w_1}=1} $$

Therefore:

$$ \frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial z_1} \frac{\partial z_1}{\partial w_1} $$ $$ =(-1.5728)(1) $$

So:

$$ \boxed{\frac{\partial L}{\partial w_1}\approx-1.5728} $$

Hooray! we've just manually backpropagated all the way from the loss to $w_1$.

#### 9. Do the bias

Same process.

Since:

$$ z_1=xw_1+b_1 $$

we have:

$$ \frac{\partial z_1}{\partial b_1}=1 $$

Therefore:

$$ \frac{\partial L}{\partial b_1} = \frac{\partial L}{\partial z_1} \cdot1 $$

So:

$$ \boxed{\frac{\partial L}{\partial b_1}\approx-1.5728} $$

Notice something interesting:

$$ \frac{\partial L}{\partial w_1} = -1.5728 $$

and:

$$ \frac{\partial L}{\partial b_1} = -1.5728 $$

They happen to be equal because $x=1$.

If we had chosen $x=2$, they would differ.

#### 10. Now calculate the output weights

We have:

$$ \hat y=a_1v_1+a_2v_2+b_3 $$

Therefore:

$$ \frac{\partial\hat y}{\partial v_1}=a_1 $$

So:

$$ \frac{\partial L}{\partial v_1} = \frac{\partial L}{\partial\hat y} \frac{\partial\hat y}{\partial v_1} $$ $$ =(-2)(0.4621) $$ $$ \boxed{\frac{\partial L}{\partial v_1}\approx-0.9242} $$

Similarly:

$$ \frac{\partial L}{\partial v_2} = (-2)(-0.4621) $$ $$ \boxed{\frac{\partial L}{\partial v_2}\approx0.9242} $$

And:

$$ \frac{\partial\hat y}{\partial b_3}=1 $$

so:

$$ \boxed{\frac{\partial L}{\partial b_3}=-2} $$

#### 11. Now calculate the second hidden neuron

It's exactly the same process.

We already know:

$$ \frac{\partial L}{\partial\hat y}=-2 $$

And:

$$ \frac{\partial\hat y}{\partial a_2}=v_2=1 $$

Therefore:

$$ \frac{\partial L}{\partial a_2}=-2 $$

Tanh derivative:

$$ \frac{\partial a_2}{\partial z_2} = 1-a_2^2 $$

Since:

$$ a_2=-0.4621 $$

we get:

$$ 1-(-0.4621)^2 \approx0.7864 $$

Therefore:

$$ \frac{\partial L}{\partial z_2} = (-2)(0.7864) = -1.5728 $$

Then:

$$ \frac{\partial z_2}{\partial w_2}=x=1 $$

so:

$$ \boxed{\frac{\partial L}{\partial w_2}\approx-1.5728} $$

And:

$$ \boxed{\frac{\partial L}{\partial b_2}\approx-1.5728} $$

#### 12. Our complete answer

We've manually calculated:

```text
Parameter     Gradient
w1            -1.5728
b1            -1.5728
w2            -1.5728
b2            -1.5728
v1            -0.9242
v2            +0.9242
b3            -2
```

Those are precisely the kinds of numbers PyTorch will put into `parameter.grad`.


#### 13. The big picture

We can now visualize backpropagation as moving backward through the forward-pass graph.

Forward:
```text
                FORWARD
                   →
                   
x → z₁ → a₁ ───────┐
                   │
                   ▼
                  ŷ
                   │
                   ▼
                   L
                   │
x → z₂ → a₂ ───────┘
```

Backward:
```text
                BACKWARD
                   ←

x ← z₁ ← a₁ ←──────┐
                   │
                   ▲
                  ŷ
                   ▲
                   │
                   L
                   │
x ← z₂ ← a₂ ←──────┘
```

And at every node, we're applying:

$$ \boxed{ \text{gradient entering node} \times \text{local derivative} = \text{gradient leaving node} } $$

That's backpropagation.

#### 14. Now let PyTorch prove our calculations

Now, run [`Lab06_paper_pencil_through_one_backpropagation.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab06_paper_pencil_through_one_backpropagation.ipynb), we should get gradients very close to the values we calculated by hand.

**That's the moment we'd aim for**: we look at `model.w1.grad` and can say "I know exactly where that number came from.".

One final distinction that is worth remembering. There are actually three different things we've now separated:

**Forward propagation**
```text
parameters + input
        ↓
    prediction
```

**Backpropagation**
```text
loss
 ↓
chain rule
 ↓
∂L/∂W, ∂L/∂b
```

**Optimization**
```text
gradients
    ↓
optimizer.step()
    ↓
new W and b
```

So when we write `loss.backward()`, **we are not updating the model**. We're only calculating the gradients. The actual parameter update happens later in `optimizer.step()`. And that distinction is fundamental to understanding PyTorch.

### Lab 07: Manually perform an optimizer.step()

This is the perfect next exercise. And for this purpose, we do not use Adam. We want the simplest possible optimizer so that the update is completely transparent.

**1. The simplest optimizer: vanilla gradient descent**

The simplest optimizer is gradient descent (GD).

For every parameter $\theta$:

$$ \boxed{ \theta_{\text{new}} = \theta_{\text{old}} - \eta \frac{\partial L}{\partial\theta} } $$

where:

$\theta$ = a parameter (w1, b1, etc.)

$\frac{\partial L}{\partial\theta}$ = gradient calculated by `loss.backward()`

$\eta$ = learning rate

For our exercise, let's choose:

$$ \eta=0.1 $$

So if:
```python
w1 = 0.5
gradient = -1.5728
```
then:

$$ w_1^{new} = 0.5 - 0.1(-1.5728) $$ $$ =0.5+0.15728 $$ $$ \boxed{w_1^{new}=0.65728} $$

That's the entire optimizer algorithm.

**2. PyTorch's simplest optimizer**

PyTorch actually provides vanilla gradient descent as: `torch.optim.SGD`

Despite the name SGD (Stochastic Gradient Descent), when we're using our single training example, the update is effectively just the simple gradient-descent equation above.

So let's add:
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

Notice that the optimizer receives: `model.parameters()`

It therefore knows about:
```text
w1
b1
w2
b2
v1
v2
b3
```

**3. Let's perform exactly one training step**

Using our `TinyNet`:

```python
model = TinyNet()

x = torch.tensor([[1.0]])
y = torch.tensor([[1.0]])

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)
```

Then:

**Forward**
```python
y_hat = model(x)
# Loss
loss = (y_hat - y).pow(2)
```

**Backpropagation**

```python
loss.backward()
```

At this point, **nothing has been updated yet**.

The model contains:
```text
parameters
    +
gradients
```

but the parameters themselves are still unchanged.

**Optimizer step**
```python
optimizer.step()
```

Now the parameters are updated.

**4. Let's make the process completely visible**

Refer to the code in [`Lab07_manual_weight_update_vs_optimizer_step.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab07_manual_weight_update_vs_optimizer_step.ipynb) and run the script. We should see the manual calculation and PyTorch result agree, up to floating-point precision.

**5. Let's manually calculate all the updates**

Using the gradients we calculated in the previous exercise:
```text
Parameter	Old value	Gradient	Update	    New value
w_1         0.5         -1.5728     +0.15728	0.65728
b_1         0           -1.5728     +0.15728	0.15728
w_2         -0.5        -1.5728     +0.15728	-0.34272
b_2         0           -1.5728     +0.15728	0.15728
v_1         1           -0.9242     +0.09242	1.09242
v_2         1           +0.9242     -0.09242	0.90758
b_3         0           -2          +0.2    	0.2
```
For example:

$w_2$

$$ w_2^{new} = -0.5-0.1(-1.5728) $$ $$ =-0.5+0.15728 $$ $$ \boxed{-0.34272} $$

$v_2$

This one is particularly nice because the gradient is positive:

$$ v_2^{new} = 1-0.1(0.9242) $$ 
$$ \boxed{0.90758} $$

So the sign of the gradient determines the direction of the update.

**6. Notice something profound about gradient descent**

Look at this:

```text
gradient < 0
     ↓
parameter increases


gradient > 0
     ↓
parameter decreases
```

Why?

Because we're trying to **minimize** the loss:

$$ \theta_{\text{new}} = \theta_{\text{old}} - \eta\nabla_\theta L $$

The gradient points toward the direction of increasing loss.

Therefore we move in the **opposite direction**.

That's why it's called **gradient descent**.

**7. We can actually implement the optimizer ourself**

This is a great little experiment.

After:

```python
loss.backward()
```

we could manually perform:

```python
learning_rate = 0.1

with torch.no_grad():

    for parameter in model.parameters():

        parameter -= learning_rate * parameter.grad
```

That's essentially vanilla gradient descent. We wrap this logic in a function and name it `UpdateWeights`. This is our simplest custom optimizer. See [`Lab07_custom_optimizer.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab07_custom_optimizer.ipynb).

The `torch.no_grad()` is important here. We're changing the parameters themselves and **don't want those parameter-update operations to become part of the autograd graph**.

So conceptually:
```text
loss.backward()
       │
       ▼
calculate gradients
       │
       ▼
┌──────────────────────────┐
│ parameter -= lr * grad   │
└──────────────────────────┘
       │
       ▼
new parameters
```

And that's basically what `torch.optim.SGD` is automating for us, along with additional functionality such as momentum and weight decay when those options are enabled.

**8. There's one PyTorch detail we should understand now**

Remember that earlier we discussed:

```python
optimizer.zero_grad()
```

Why do we need it?

Because PyTorch **accumulates gradients**.

Suppose we do:

```python
loss.backward()
```

and get:
```text
w1.grad = -1.5728
```

If we do another forward/backward pass without clearing the gradient, PyTorch adds the new gradient to the existing one.

So the normal training loop is:
```python
for ...:

    optimizer.zero_grad()

    y_hat = model(x)

    loss = loss_fn(y_hat, y)

    loss.backward()

    optimizer.step()
```

Think of it as:
```text
┌──────────────────────┐
│ zero old gradients   │
└──────────┬───────────┘
           ↓
      forward pass
           ↓
       calculate loss
           ↓
      backward pass
           ↓
        gradients
           ↓
     optimizer.step()
           │
           └───────────► next iteration
```

**9. One subtle terminology point**

As for the "simplest optimizer", we'd distinguish two things:

**Gradient Descent**

$$ \theta\leftarrow\theta-\eta\nabla L $$

This is the _algorithm_ we want to understand.

**SGD optimizer in PyTorch**
```python
torch.optim.SGD(...)
```

is PyTorch's implementation of stochastic gradient descent, with optional features such as momentum.

With our **one-example** experiment, plain SGD without momentum is exactly the simple update we're doing.

So for learning, this is perfect.

Our `TinyNet` learning journey now looks like this

We've essentially built a complete neural-network training system from first principles:
```text
                 FORWARD
                    │
                    ▼
              model(x)
                    │
                    ▼
                prediction
                    │
                    ▼
                  loss
                    │
                    ▼
              BACKPROPAGATION
              loss.backward()
                    │
                    ▼
                gradients
                    │
                    ▼
              OPTIMIZATION
              θ ← θ - η∇L
                    │
                    ▼
              new parameters
```

And we've now manually verified both halves:

1. `loss.backward()` → chain rule → gradients
2. `optimizer.step()` → gradients → parameter updates

That's a very solid conceptual foundation.

We'd recommend a take-home assignment that is particularly illuminating: run two consecutive training steps on the same `(x=1, y=1)` example and manually follow how the second forward pass changes because the parameters from Step 1 have changed. That will make the iterative nature of neural-network training completely concrete—and also give us a natural reason for `zero_grad()`.

#### Q & A

**How does loss.backward() work? "loss" seems to be a naked tensor to me**

We could have asked this question in an earlier experiment. But it is never too late to ask it or any question :). This is indeed a very common point of confusion. `loss` looks "naked" to us (it's just a tensor with values), but it is not naked to PyTorch.

For the following code snippet:

```python
model = TinyNet()

x = torch.tensor([[1.0]])
y = torch.tensor([[1.0]])

y_hat = model(x)

loss = (y_hat - y).pow(2)
```

Let's see what happens under the hood, step by step:

1. `y_hat = model(x)`

`model(x)` runs through W1, b1, W2, b2, etc., all of which are `nn.Parameter` with `requires_grad=True`.

Every operation inside `forward` (matmul, add, tanh) produces tensors that also have `requires_grad=True` and a `grad_fn`.

So `y_hat.requires_grad == True`, and `y_hat.grad_fn` is something like `AddBackward0` (the last op in our forward).

2. `(y_hat - y)`

`y_hat` requires `grad`; `y` does not (it's data).

Subtraction between a grad-requiring tensor and a non-grad tensor → result requires `grad`.

Result has `grad_fn = SubBackward0`, pointing to y_hat's grad_fn.

3. .pow(2)

Squaring a grad-requiring tensor → result requires `grad`.

Result has `grad_fn = PowBackward0`, pointing to `SubBackward0`.

So `loss` is a tensor of shape (1, 1), with `requires_grad=True`, and with `grad_fn = PowBackward0` — the root of a graph that traces all the way back through `SubBackward0`, `AddBackward0` (from `y_hat`), `MulBackward`, `TanhBackward`, etc..

To quickly visualize this computational graph, we can recursively traverse the `grad_fn` attributes of our loss tensor:

```python
import torch

def print_graph(tensor, indent=0):
    if tensor.grad_fn is not None:
        print(" " * indent + str(tensor.grad_fn))
        for next_fn, _ in tensor.grad_fn.next_functions:
            if next_fn is not None:
                print_graph_fn(next_fn, indent + 2)
    elif tensor.requires_grad:
        print(" " * indent + "leaf (requires_grad=True)")

def print_graph_fn(grad_fn, indent=0):
    print(" " * indent + str(grad_fn))
    for next_fn, _ in grad_fn.next_functions:
        if next_fn is not None:
            print_graph_fn(next_fn, indent + 2)

...

# After our forward pass
loss = (y_hat - y).pow(2)

print_graph(loss)
```

We will see something like the following, a hierarchical tree of operations (like PowBackward0, SubBackward0, etc.), showing how our loss traces back through the network:

```text
<PowBackward0 object at 0x7e912452db10>
  <SubBackward0 object at 0x7e912452d990>
    <AddBackward0 object at 0x7e912452d3c0>
      <AddBackward0 object at 0x7e912452df90>
        <MulBackward0 object at 0x7e912452e1a0>
          <TanhBackward0 object at 0x7e912452e350>
            <AddBackward0 object at 0x7e912452e3e0>
              <MulBackward0 object at 0x7e912452e440>
                <AccumulateGrad object at 0x7e912452ed70>
              <AccumulateGrad object at 0x7e912452e4a0>
          <AccumulateGrad object at 0x7e912452e380>
        <MulBackward0 object at 0x7e912452e200>
          <TanhBackward0 object at 0x7e912452e380>
            <AddBackward0 object at 0x7e912452e3e0>
              <MulBackward0 object at 0x7e912452e4a0>
                <AccumulateGrad object at 0x7e912452ed70>
              <AccumulateGrad object at 0x7e912452e530>
          <AccumulateGrad object at 0x7e912452e3b0>
      <AccumulateGrad object at 0x7e912452e140>

```

For a proper visual diagram (like a flowchart), `torchviz` is a popular choice. It uses `Graphviz` to render the `autograd` graph.

```python
from torchviz import make_dot

# After our forward pass
loss = (y_hat - y).pow(2)

# Create the visualization
make_dot(loss, params=dict(model.named_parameters())).render("computational_graph", format="png")
```

This will generate a PNG image file showing the graph of operations like the following:

![Computational Graph](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Lab07_computational_graph.png)

We can also use `show_attrs=True` and `show_saved=True` to see more details about what `autograd` saves for the backward pass.

If we run this in Google Colab, we can save a copy of the graph in our Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

from torchviz import make_dot

# Create the visualization
make_dot(loss, 
         params=dict(model.named_parameters())
         ).render("/content/drive/MyDrive/computational_graph",
                  format="png"
          )
```

That's why `loss.backward()` works. It starts at `loss.grad_fn` and walks backward.

Here's a bonus :)

PyTorch's built-in `torch.fx` can symbolically trace our model to produce a textual representation of the forward graph in a tabular format and print out the generated python code from the graph.

```python
import torch.fx

# Trace our model
traced_model = torch.fx.symbolic_trace(model)

# Print the internal graph (tabular format)
traced_model.graph.print_tabular()

# **Print** the generated Python code
print(traced_model.code)
```

This is the output:

```text
opcode         name    target                                                   args            kwargs
-------------  ------  -------------------------------------------------------  --------------  --------
placeholder    x       x                                                        ()              {}
get_attr       w1      w1                                                       ()              {}
call_function  mul     <built-in function mul>                                  (x, w1)         {}
get_attr       b1      b1                                                       ()              {}
call_function  add     <built-in function add>                                  (mul, b1)       {}
call_function  tanh    <built-in method tanh of type object at 0x7e913bf81ce0>  (add,)          {}
get_attr       w2      w2                                                       ()              {}
call_function  mul_1   <built-in function mul>                                  (x, w2)         {}
get_attr       b2      b2                                                       ()              {}
call_function  add_1   <built-in function add>                                  (mul_1, b2)     {}
call_function  tanh_1  <built-in method tanh of type object at 0x7e913bf81ce0>  (add_1,)        {}
get_attr       v1      v1                                                       ()              {}
call_function  mul_2   <built-in function mul>                                  (tanh, v1)      {}
get_attr       v2      v2                                                       ()              {}
call_function  mul_3   <built-in function mul>                                  (tanh_1, v2)    {}
call_function  add_2   <built-in function add>                                  (mul_2, mul_3)  {}
get_attr       b3      b3                                                       ()              {}
call_function  add_3   <built-in function add>                                  (add_2, b3)     {}
output         output  output                                                   (add_3,)        {}



def forward(self, x):
    w1 = self.w1
    mul = x * w1;  w1 = None
    b1 = self.b1
    add = mul + b1;  mul = b1 = None
    tanh = torch.tanh(add);  add = None
    w2 = self.w2
    mul_1 = x * w2;  x = w2 = None
    b2 = self.b2
    add_1 = mul_1 + b2;  mul_1 = b2 = None
    tanh_1 = torch.tanh(add_1);  add_1 = None
    v1 = self.v1
    mul_2 = tanh * v1;  tanh = v1 = None
    v2 = self.v2
    mul_3 = tanh_1 * v2;  tanh_1 = v2 = None
    add_2 = mul_2 + mul_3;  mul_2 = mul_3 = None
    b3 = self.b3
    add_3 = add_2 + b3;  add_2 = b3 = None
    return add_3
```

**How does the optimizer know which parameter is which?**

New comers to neural network using PyTorch often have this confusion: 

"When I print out model.parameters(), I don't see parameter names. If parameter names are not included in what is returned from model.parameters(), then how an optimizer knows which parameter is which and how to update each parameter?"

Suppose:
```python
for p in model.parameters():
    print(p)
```

We get something conceptually like:
```text
Parameter containing:
tensor(...)

Parameter containing:
tensor(...)

Parameter containing:
tensor(...)
```

No names.

So how does SGD know what to update?

The answer is:

    It doesn't need names.

The optimizer simply receives **references to the actual `Parameter` objects**.

Consider:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

Conceptually:
```text
model
 │
 ├── W1 ────────┐
 ├── b1         │
 ├── W2         │
 ├── b2         │
 └── ...        │
                ↓
          optimizer
```
The optimizer gets the actual objects.

Each Parameter contains things like:
```text
Parameter
 ├── tensor data
 └── .grad
```
After:
```python
loss.backward()
```
we might have:

```text
W1.data       W1.grad
   │             │
   ↓             ↓
[values]      [gradients]

b1.data       b1.grad
   │             │
   ↓             ↓
[values]      [gradients]
```
Then SGD essentially performs:

$$ \theta \leftarrow \theta-\eta\nabla_\theta L $$

for each parameter object.

Conceptually:
```python
for p in model.parameters():
    p -= learning_rate * p.grad
```
So the optimizer doesn't need to know:

    "This is W1."

It only needs to know:

    "Here is a tensor. It has a gradient. Update this tensor according to SGD."

**Then what are parameter names for?**

Names are primarily useful for bookkeeping and inspection.

For example:
```python
for name, p in model.named_parameters():
    print(name, p.shape)
```
might produce:
```text
network.0.weight torch.Size([32, 1])
network.0.bias   torch.Size([32])
network.2.weight torch.Size([32, 32])
network.2.bias   torch.Size([32])
network.4.weight torch.Size([1, 32])
network.4.bias   torch.Size([1])
```
The optimizer doesn't fundamentally need those strings.

They are useful for:

- inspecting a model
- saving/loading state dictionaries
- debugging
- selectively freezing parameters
- logging
- checking parameter shapes

**Why does SineNet say network.0.weight but TinyNet says W1?**

"For SineNet model, I only see named parameters like `network.0.bia` and `network.0.weight`, instead of W1, b1, as a result of print results from `named_parameters()`. But for `TinyNet` model, printing named parameters returns W1, b1, W2, b2, W3, and b3. Why?"

The answer is: PyTorch parameter names are determined by how parameters/modules are registered inside our model.

Let's look at these two cases.

**SineNet**

We have:
```python
class SineNet(nn.Module):
    def __init__(self, hidden_layer_width=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(1, hidden_layer_width),
            nn.Tanh(),

            nn.Linear(hidden_layer_width, hidden_layer_width),
            nn.Tanh(),

            nn.Linear(hidden_layer_width, 1)
        )
```
Notice that we gave the whole `Sequential` object the name: `self.network`. Inside it, PyTorch registers the modules by position:

```text
network
 │
 ├── 0 → Linear(1, 32)
 │
 ├── 1 → Tanh
 │
 ├── 2 → Linear(32, 32)
 │
 ├── 3 → Tanh
 │
 └── 4 → Linear(32, 1)
```

Therefore the first Linear layer's parameters are named:
```text
network.0.weight
network.0.bias
```

The second:
```text
network.2.weight
network.2.bias
```
The third:
```text
network.4.weight
network.4.bias
```
That's where `network.0.weight` comes from. It's basically a **path** through the module hierarchy:

```text
self.network
    ↓
index 0
    ↓
weight
```

**TinyNet is different**

Our `TinyNet` manually declares:
```python
self.W1 = nn.Parameter(...)
self.b1 = nn.Parameter(...)

self.W2 = nn.Parameter(...)
self.b2 = nn.Parameter(...)

self.W3 = nn.Parameter(...)
self.b3 = nn.Parameter(...)
```
Here we're directly registering parameters under those exact attribute names.

Therefore:
```python
for name, p in model.named_parameters():
    print(name)
```
gives:
```text
W1
b1
W2
b2
W3
b3
```
because we named them that way.

**This gives us a nice mental model**

Think of an `nn.Module` as a tree.

For `TinyNet`:
```text
TinyNet
│
├── W1
├── b1
├── W2
├── b2
├── W3
└── b3
```

For our `Sequential` `SineNet`:
```text
SineNet
│
└── network
    │
    ├── 0 → Linear
    │       ├── weight
    │       └── bias
    │
    ├── 1 → Tanh
    │
    ├── 2 → Linear
    │       ├── weight
    │       └── bias
    │
    ├── 3 → Tanh
    │
    └── 4 → Linear
            ├── weight
            └── bias
```

So PyTorch constructs names by following that tree:
```text
network
   +
   0
   +
weight
   ↓
network.0.weight
```
And there's a very important connection to our previous question "If parameters() doesn't contain names, how does the optimizer know which parameter is which?"

Now we can see that there are really **three separate concepts**:

① The parameter object: `p`

This is the actual trainable tensor.

② The parameter name: "W1" or "network.0.weight"

This is a bookkeeping identifier.

③ The gradient: `p.grad`

This is the gradient associated with that particular parameter object.

So after `loss.backward()`, we have something like:
```text
                  parameter              gradient
                  ─────────              ────────
W1       ───────→  tensor(...)    ←──── tensor(...)
b1       ───────→  tensor(...)    ←──── tensor(...)
W2       ───────→  tensor(...)    ←──── tensor(...)
b2       ───────→  tensor(...)    ←──── tensor(...)
...
```

And SGD simply walks through those parameter objects and updates them.

**The name is not what connects the gradient to the parameter.**

The gradient is attached directly to the `Parameter` object: `p.grad`. That's a fundamental PyTorch concept.

**One experiment we'd recommend**

Run this on our TinyNet:
```python
for name, p in model.named_parameters():
    print(
        name,
        "id =", id(p),
        "shape =", tuple(p.shape),
        "grad =", p.grad
    )
```

Then compare:
```python
list(model.parameters())
```
with:
```python
list(model.named_parameters())
```

The important realization should be:
```text
parameters()

    Parameter
    Parameter
    Parameter
    ...

named_parameters()

    ("W1", Parameter)
    ("b1", Parameter)
    ("W2", Parameter)
    ...
```

In other words, `named_parameters()` **is essentially giving us the same parameter objects plus their names**.

And this is exactly why our manual TinyNet is such a good learning tool: we've gone from `nn.Linear` to understanding that underneath it we essentially have

$$ W,\ b $$

registered as trainable `Parameter` objects, which autograd gives gradients to, and the optimizer updates.

### Lab 08: Return to nn.Linear / DataLoader with batching / standard PyTorch style -> understand what PyTorch is doing for us 

With a solid understanding of basic building blocks of a neural network, mechanics of both a forward pass and a back propagation, it's time to return to standard PyTorch style, using its high level API. 

So, let's refactor our first lab [`Lab01_SineFunctionApproximation_nn_Sequential.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab01_SineFunctionApproximation_nn_Sequential.ipynb) while preserving the experiment's core design: the same SineNet architecture, randomly sampled sine data, 80/20 training-validation split, input normalization, MSE loss, Adam optimizer, and interpolation/extrapolation evaluation.

The main change is to introduce Dataset and DataLoader so that training happens in mini-batches, rather than passing all 800 training examples through the model in a single forward pass per epoch.

See [`Lab08_SineFunctionApproximation_dataloader_with_batching.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab08_SineFunctionApproximation_dataloader_with_batching.ipynb)

**1. What changes conceptually?**

Our original training loop performs one update per epoch:

```text
800 training examples

One forward pass → one loss → one backward pass

One optimizer update per epoch
```

With a batch size of 64, the new training loop works like this:

```text
800 training examples
   ↓
Batch 1: 64 examples
Batch 2: 64 examples
...
Batch 13: 32 examples
   ↓
13 optimizer updates per epoch
```

Each batch gets its own _forward pass_, _loss calculation_, _backpropagation_, and _optimizer update_. With 800 training examples and a batch size of 64, there are `13 batches` per epoch.

One important distinction: we'll continue evaluating the entire validation set once per epoch, without calculating gradients. That makes validation loss comparable across epochs.

**2. The new Dataset and DataLoader**

Because our data is already stored in tensors, PyTorch's TensorDataset is sufficient. We don't need to write a custom Dataset class yet.

```python
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(x_train_norm, y_train)
val_dataset = TensorDataset(x_val_norm, y_val)

batch_size = 64

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False
)
```

Here's what each component does:

```text
*Component*           *Responsibility*
TensorDataset       Pairs each input x with its target y.
DataLoader          Retrieves examples in batches.
batch_size=64       Returns up to 64 input-target pairs per iteration.
shuffle=True        Randomizes the order of training examples each epoch.
shuffle=False       Keeps validation ordering unchanged.
```

The final training batch contains only 32 examples because 800 is not divisible by 64.

**3. The new training loop**

This is the most important part of the refactor:

```python
for epoch in range(epochs):

    model.train()

    for batch_x, batch_y in train_loader:

        predictions = model(batch_x)
        loss = loss_fn(predictions, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

Notice that our forward propagation, loss calculation, backpropagation, and optimizer update are unchanged in principle.

The difference is that they now happen **once per batch instead of once per epoch**.

The shapes illustrate this nicely:

- batch_x: (64, 1) for a full batch

- predictions: (64, 1)

- batch_y: (64, 1)

- loss: a tensor with `grad_fn`

For the final batch, the first three shapes become (32, 1).

Validation now iterates through `val_loader` under `torch.no_grad()`. The script computes an example-weighted average loss across all validation batches, so the smaller final batch is handled correctly.

**4. One important training detail: epochs versus optimizer updates**

Our original code uses 20,000 epochs, but each epoch performs just one optimizer update. The refactored version starts with 2,000 epochs.

```text
Configuration           Original    Refactored
Training examples       800         800
Batch size              800         64
Updates per epoch       1           13
Epochs                  20,000      2,000
Total optimizer updates 20,000      26,000
```

The refactored run therefore performs a similar order of magnitude of updates, although mini-batch gradients differ from full-batch gradients. The training curves and final MSE will not necessarily match the original run.

**5. What we'd observe**

When we run the new script, pay attention to three things:

- Loss curves: Training loss may fluctuate more between epochs because each mini-batch produces a different gradient.

- Training speed: Mini-batching introduces more optimizer updates per epoch, but each update processes fewer examples.

- Generalization: Compare interpolation and extrapolation MSE with our previous experiment. Batching changes the optimization process; it does not guarantee better extrapolation.

One final distinction: we're using `TensorDataset`, which is itself a PyTorch Dataset implementation. Later, when we work with data stored in files, images, or other sources that need on-demand processing, writing our own class inheriting from `torch.utils.data.Dataset` will become more useful. For this experiment, `TensorDataset` keeps the focus on understanding the batching mechanism without adding unnecessary complexity.

**6.Interesting observation of loss**

When `N=1000`, batch version with `batch_size=64` and `epochs=1540` performs substantially worse than full-batch version (Lab 01, i.e., the whole training dataset is fed into the neural network at once during training) for `20000` epochs.

```text
*batch_size=64, epochs=1540*
Interpolation MSE: 0.500012
Extrapolation MSE: 2.258733

*full-batch, epochs=20000*
Interpolation MSE: 0.121237
Extrapolation MSE: 0.469854
```

The above two setups are equivalent in their optimizer update counts. Here's why. Full-batch version does exactly 20000 updates. For batch-size=64, in one epoch, 800 examples will contain 13 batches and updates (one update per batch). Thus 1540 epoches gives us 20020 updates.

By maintaining approximately 20000 updates, we tried different combinations of batch_size and epochs and found that batch version (i.e., `Lab 08`) consistently performs worse than its full-batch version before batch_size hits `800`, as shown below:

```text
batch_size  updates/batch   epochs      NTPPSC  loss_eval
32          25              800           1       0.501912 
64          13              1540          2       0.500012
128          7              2860          4       0.498502
256          4              5000          8       0.498833
400          2              10000         13      0.434270
800          1              20000         25      0.017335
```

where NTPPSC stands for "number of training points per Sine cycle".

Why?

Our intutition is that a batch of 64 points (or other values `<800`) is too small to reveal the oscillation patterns of sine. Therefore, each update from a batch may nudge weights insufficiently or incorrectly.

So, **Does a batch of 64 contain enough sine-wave information?**

Our intuition has a mathematical basis, especially because sine is an oscillatory function.
Our training domain is \([-100,100]\). The sine function completes approximately

$$
\frac{200}{2\pi}\approx31.8
$$

full oscillations in that interval.

With 800 training examples, we have on average:

```math
\frac{800}{31.8}\approx25.2
```

training points per oscillation.

But with a batch size of 64, the average is only:

$$
\frac{64}{31.8}\approx2.0
$$

points per oscillation within an individual batch.

That is a meaningful difference.

![Illustration: data available to each gradient update](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/images/Sine_approximation_64points-vs-800points-in-a-batch.png)

There is an important caveat: our DataLoader uses shuffle=True, so each batch is not a contiguous interval of x-values. Its points are scattered across the domain. Consequently, a batch of 64 may span nearly the entire input range, but it still has fewer observations per oscillation.

So our hypothesis is partly right: a larger batch gives each gradient update a more complete and stable picture of the function across its oscillations.

However, mini-batch training does not permanently discard the other observations. Across one epoch, all 800 training examples are processed. The issue is that each update uses only a subset, and the model's parameters change between batches.

**Why can full-batch training work better here?**

Recall the gradient descent update:
```math
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
```

For full-batch training, the gradient is computed from all 800 training examples:
```math
\nabla_\theta L_{\text{full}}
=\frac{1}{800}\sum_{i=1}^{800}
\nabla_\theta \ell_i(\theta)
```

For a mini-batch of 64 examples, it is instead:
```math
\nabla_\theta L_{\text{batch}}
=\frac{1}{64}\sum_{i\in B}
\nabla_\theta \ell_i(\theta)
```

The mini-batch gradient is an estimate of the full-data gradient. It varies depending on which examples are in the batch.

For our sine experiment, that variability can matter because:
- The target function oscillates repeatedly over the training interval.
- The network uses nonlinear tanh activations.
- Different parts of the network interact when representing those oscillations.
- A mini-batch update can improve the fit for its particular sample while temporarily worsening it elsewhere.

Full-batch gradients average over all training points before changing the parameters. This can give smoother optimization dynamics for this particular problem.

But mini-batch gradients are not inherently worse. In many larger neural-network training problems, mini-batches are more efficient and work extremely well.

### Lab 09: Detect the available compute device at runtime and move the model and its data to that device.

This is a long-awaited experiment, which I would have made the first in our progression. But I decided to save the best for last — so here we go.

Our preference of a compute device: `CUDA → MPS → CPU`. So, running on GPUs is our first choice for training large models with massive dataset. 

So, we will show how to add runtime device detection and move compute with data to a selected device for our `SineNet`. See [`Lab09_SineFunctionApproximation_dataloader_with_batching_device.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab09_SineFunctionApproximation_dataloader_with_batching_device.ipynb).


Note though, GPU acceleration is not guaranteed to make this particular experiment faster. Our network and dataset are small, so GPU launch overhead and data transfers may outweigh the computation saved. The main benefit here is learning a device-aware PyTorch workflow that will also work for larger models and datasets.


**1. Runtime device detection**

The script now checks for CUDA, then Apple's Metal (MPS) backend, and falls back to the CPU.

```python
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
```

This supports NVIDIA GPUs through CUDA, compatible Apple Silicon Macs through MPS, and CPU-only machines.

**2. Move the model to the selected device**

```python
model = SineNet().to(device)
```

This moves the model's parameters to the selected device. The optimizer can then operate on those parameters as usual.

**3. Move each batch to the same device**

Inside the training loop:

```python
for batch_x, batch_y in train_loader:
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    predictions = model(batch_x)
    loss = loss_fn(predictions, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

The DataLoader continues to provide CPU tensors by default. We explicitly move each batch to the selected device before computing predictions.

Validation follows the same principle.

**4. Evaluate on the selected device, then return predictions to CPU**

The test inputs and targets are moved to the device for inference and loss calculation. The predictions are then moved back to CPU:
```python
pred_outside_device = model(x_test_outside_norm.to(device))

outside_loss = loss_fn(
    pred_outside_device,
    y_test_outside.to(device)
)

pred_outside = pred_outside_device.cpu()
```
This last step is important because Matplotlib plotting code commonly expects CPU tensors when converting them to NumPy arrays.

**5. How to verify it works**

Run the script and look for the printed device, for example:

Using device: cpu

or, on a compatible machine:

Using device: cuda

The model, batches, and loss calculations will all use that device.


## Project Structure

```text
.
├── README.md
├── images
│   ├── Fourier-feature_...-Width128.png
...
│   └── sinefunctionfitting-truth-vs-prediction.png
└── notebooks
    ├── Lab01_SineFunctionApproximation_FourierFeatures_MLP.ipynb
    ├── Lab01_SineFunctionApproximation_nn_Sequential.ipynb
    ├── Lab03_SineFunctionApproximation_nn_parameter.ipynb
    ├── Lab03_SineFunctionApproximation_nn_parameter_xavier.ipynb
    ├── Lab05_inspection_one_example_forward_loss_backward_optimizer.ipynb
    ├── Lab06_paper_pencil_through_one_backpropagation.ipynb
    ├── Lab07_custom_optimizer.ipynb
    ├── Lab07_manual_weight_update_vs_optimizer_step.ipynb
    ├── Lab08_SineFunctionApproximation_dataloader_with_batching.ipynb
    └── Lab09_SineFunctionApproximation_dataloader_with_batching_device.ipynb
```

## Installation

Pick Google Colab as our environment so nothing needs to be installed manually.

Otherwsise, install `PyTorch` and `matplotlib` (optionally `torchviz`).

## How to Run

All experiments are written in Jupyter Notebook (.ipynb).

Simplest way to run all these experiments is to load .ipynb files to Google Colab and run from there. For Sine Function Approximation experiments (i.e. Lab01, Lab02, Lab03, Lab08, and Lab09), choosing T4 GPU for our runtime may accelerate training (note: currently, we don't have runtime device detection and GPU acceleration support in our code for these experiments; so, in order for this to work, read `Lab 09` for instructions and use [`Lab09_SineFunctionApproximation_dataloader_with_batching_device.ipynb`](https://github.com/lichtjiang/neural-networks-from-first-principles-pytorch/blob/main/notebooks/Lab09_SineFunctionApproximation_dataloader_with_batching_device.ipynb) as a reference). For the remaining experiments, running on CPUs is just fine.