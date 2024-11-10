Backpropagation is a key process for training a neural network, and it involves updating the weights and biases of the network by calculating gradients and propagating errors backward through each layer. Here's a detailed **step-by-step** breakdown of how backpropagation works at each layer:

### 1. **Forward Pass (Initial Step)**

Before backpropagation starts, the input data passes through the network from the input layer to the output layer. In this step, activations are computed and stored for later use in the backward pass.

- **Input to the network:** The input data \(X\) (e.g., an image or a set of features).
- **For each layer**:

  - **Dense Layer:** A linear transformation is applied to the input data:
    \[
    z = X W + b
    \]
    where:

    - \( X \) is the input data to the layer.
    - \( W \) is the weights of the layer.
    - \( b \) is the biases of the layer.
    - \( z \) is the output of the weighted sum (before activation).
  - **Activation Function (e.g., ReLU):** The activation function is applied element-wise to the output of the dense layer to introduce non-linearity:
    \[
    a = \text{ReLU}(z) = \max(0, z)
    \]
    where:

    - \( a \) is the output after applying ReLU (or another activation function like sigmoid, tanh, etc.).
  - **Batch Normalization (if present):** Normalizes the activations to maintain stable learning by reducing internal covariate shift. It computes:
    \[
    \hat{X} = \frac{X - \mu}{\sqrt{\sigma^2 + \epsilon}}
    \]
    where:

    - \( \mu \) and \( \sigma^2 \) are the mean and variance of the batch.
    - \( \epsilon \) is a small constant to prevent division by zero.
    - After normalization, scaling and shifting are applied using learned parameters \( \gamma \) and \( \beta \).
  - **Dropout (if present):** Randomly zeroes some of the activations during training to prevent overfitting. This is only applied during training, not during inference.

### 2. **Compute Loss**

Once the forward pass is completed, we calculate the **loss** based on the network's output and the true target labels. For classification problems, **cross-entropy loss** is often used.

- **Cross-Entropy Loss (for multi-class classification):**
  \[
  \text{Loss} = -\sum_{i} y_i \log(\hat{y}_i)
  \]
  where:
  - \( y_i \) is the true label (in one-hot encoding).
  - \( \hat{y}_i \) is the predicted probability of the class \(i\) (output of the softmax in the output layer).

The loss is used to determine how much the predictions deviate from the actual values, guiding the network in the update process.

### 3. **Backward Pass (Error Propagation)**

Backpropagation is essentially an application of the **chain rule of calculus** to compute gradients of the loss with respect to the weights and biases of the network. Here's what happens at each layer during the backward pass:

#### a. **Output Layer (Softmax + Loss)**

For the output layer, we first calculate the **gradient of the loss** with respect to the network's output (before softmax) and then propagate that gradient backward through the layers.

1. **Softmax Layer** (for multi-class classification):

   - The **softmax activation** gives a probability distribution over the classes:
     \[
     \hat{y}_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}
     \]
     where \( z_i \) is the raw output (logits) of the network.
2. **Loss Gradient** (cross-entropy with softmax combined):

   - The **derivative of the cross-entropy loss** with respect to the logits is simplified as:
     \[
     \frac{\partial \text{Loss}}{\partial z_i} = \hat{y}_i - y_i
     \]
     where:

     - \( \hat{y}_i \) is the predicted probability of class \(i\).
     - \( y_i \) is the true label in one-hot encoding.
   - This gradient indicates how much the logits need to be adjusted to reduce the loss.

#### b. **Hidden Layers (Backpropagating Gradients)**

Next, the error (gradient) is propagated backward from the output to the hidden layers, adjusting the weights and biases at each layer.

1. **ReLU Activation Layer:**

   - For each element of the input \( z \) (the pre-activation output), the derivative of the ReLU activation is computed:
     \[
     \frac{\partial \text{ReLU}}{\partial z} =
     \begin{cases}
     1 & \text{if } z > 0 \\
     0 & \text{if } z \leq 0
     \end{cases}
     \]
   - The gradient is passed backward through the ReLU activation, and only the elements of the gradient corresponding to positive activations are passed through.
2. **Dense (Fully Connected) Layer:**

   - The gradients are propagated to the weights and biases:
     \[
     \frac{\partial \text{Loss}}{\partial W} = X^T \cdot \frac{\partial \text{Loss}}{\partial Z}
     \]
     where:

     - \( X^T \) is the transposed input to the layer.
     - \( \frac{\partial \text{Loss}}{\partial Z} \) is the gradient of the loss with respect to the pre-activation \( Z \).
   - The gradients are also computed for the biases:
     \[
     \frac{\partial \text{Loss}}{\partial b} = \sum \frac{\partial \text{Loss}}{\partial Z}
     \]
3. **Batch Normalization Layer:**

   - During backpropagation, the gradients are propagated through the batch normalization operations. This involves computing gradients with respect to the normalized input, the scaling parameter \( \gamma \), and the shifting parameter \( \beta \).
   - The gradient of the loss with respect to the inputs of batch normalization is:
     \[
     \frac{\partial \text{Loss}}{\partial X} = \frac{\partial \text{Loss}}{\partial \hat{X}} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}}
     \]
   - Gradients are also computed for the parameters \( \gamma \) and \( \beta \).
4. **Dropout Layer:**

   - During backpropagation, the dropout mask is applied to the gradients to ensure that only the "active" neurons during the forward pass contribute to the gradient.

### 4. **Parameter Update (Optimizer Step)**

Once the gradients are computed for all layers, the weights and biases are updated using an optimization algorithm like **Stochastic Gradient Descent (SGD)** or **Adam**.

- The update rule for each parameter \( \theta \) (weights or biases) is:
  \[
  \theta = \theta - \eta \frac{\partial \text{Loss}}{\partial \theta}
  \]
  where:
  - \( \eta \) is the learning rate.
  - \( \frac{\partial \text{Loss}}{\partial \theta} \) is the gradient of the loss with respect to the parameter.

### 5. **Repeat**

This process is repeated for a number of **epochs** (iterations over the entire dataset), and after each pass through the data, the weights are updated to minimize the loss.

---

### Summary of Steps per Layer in Backpropagation:

1. **Output Layer (Softmax + Loss):**

   - Compute output probabilities (softmax).
   - Calculate loss (e.g., cross-entropy).
   - Compute gradient of loss w.r.t. logits.
2. **Hidden Layers (Dense + Activation + Batch Normalization + Dropout):**

   - Compute gradients w.r.t. weights and biases using chain rule.
   - Pass gradients backward through activation functions (ReLU, etc.).
   - Compute gradients for batch normalization and dropout layers.
3. **Update Parameters:**

   - Use the optimizer to update the weights and biases.

This loop continues for each batch and through all epochs until the model converges to a set of parameters that minimize the loss function.
