# -*- coding: utf-8 -*-

import torch as t

# Forward pass

# Input
x1 = t.Tensor([[1], [2], [3]])

# Hidden layer
w1 = t.rand((3, 3), requires_grad = True)
b1 = t.rand((3, 1), requires_grad = True)
w1x1 = t.mm(w1, x1)
x2 = w1x1 + b1

# Output layer
w2 = t.rand((1, 3), requires_grad = True)
b2 = t.rand(1, requires_grad = True)
w2x2 = t.mm(w2, x2)
y_pred = w2x2 + b2

# Loss
y_train = t.Tensor([[7.]])
err = y_pred - y_train
loss = err ** 2

# Backword pass

loss.backward()

# Keep in mind the chai rule for gradients
d_err = 2 * err
d_y_pred = d_err * 1
d_w2x2 = d_y_pred * 1
d_b2 = d_y_pred * 1
assert d_b2 == b2.grad
d_w2 = (d_w2x2 * x2).T
assert t.all(d_w2 == w2.grad)
d_x2 = (d_w2x2 * w2).T
d_w1x1 = d_x2 * 1
d_b1 = d_x2 * 1
assert t.all(d_b1 == b1.grad)
d_w1 = d_w1x1 * x1.T
assert t.all(d_w1 == w1.grad)