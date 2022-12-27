"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for theta_id, theta in enumerate(self.params):
            grad = theta.grad.detach() + self.weight_decay * theta.data.detach()
            if theta_id not in self.u:
                u_cur = (1. - self.momentum) * grad
            else:
                u_cur = self.u[theta_id] * self.momentum + (1. - self.momentum) * grad

            self.u[theta_id] = u_cur.detach()

            theta.data -= self.lr * self.u[theta_id]
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for theta_id, theta in enumerate(self.params):
            # print(f"theta_id: {theta_id} theta max: {theta.numpy().max()}")
            # print(f"theta grad device: {theta.grad.device} theta device: {theta.device}")
            grad = theta.grad.detach() + self.weight_decay * theta.data.detach()
            if theta_id not in self.m:
                m_cur = (1 - self.beta1) * grad
            else:
                m_cur = self.m[theta_id] * self.beta1 + (1 - self.beta1) * grad
            if theta_id not in self.v:
                v_cur = (1 - self.beta2) * (grad ** 2)
            else:
                v_cur = self.v[theta_id] * self.beta2 + (1 - self.beta2) * (grad ** 2)

            self.m[theta_id] = m_cur.detach()
            self.v[theta_id] = v_cur.detach()

            m_next_hat = m_cur / (1 - self.beta1 ** self.t)
            v_next_hat = v_cur / (1 - self.beta2 ** self.t)
            theta.data -= self.lr * m_next_hat / ((v_next_hat ** 0.5) + self.eps)
        ### END YOUR SOLUTION
