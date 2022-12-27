import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss, opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    print()

    if opt:
        train_cifar10(model, dataloader, optimizer=opt, loss_fn=loss_fn)
    else:
        evaluate_cifar10(model, dataloader, loss_fn=loss_fn)
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss, device=device):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn()
    model.train()

    for _ in range(n_epochs):
        correct, total_loss, num_sample = 0, 0, 0
        for step, batch in enumerate(dataloader):
            img, label = ndl.Tensor(batch[0], device=device), ndl.Tensor(batch[1], device=device)
            out = model(img)
            loss = loss_fn(out, label)

            opt.reset_grad()
            loss.backward()
            opt.step()

            num_sample += label.shape[0]
            total_loss += loss.numpy().item() * label.shape[0]
            correct += np.sum(np.argmax(out.numpy(), axis=1) == label.numpy()).item()

            rate = (step + 1) / len(dataloader)
            a = "*" * int(rate * 50)
            b = "." * int((1-rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}  correct:{}".format(int(rate*100), a, b, loss.numpy().item(), correct), end="")
        print()

    return correct/num_sample, total_loss/num_sample
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, total_loss, num_sample = 0, 0, 0
    loss_fn = loss_fn()
    model.eval()

 
    for step, batch in enumerate(dataloader):
        img, label = ndl.Tensor(batch[0], device=device), ndl.Tensor(batch[1], device=device)
        out = model(img)
        loss = loss_fn(out, label)
        
        num_sample += label.shape[0]
        total_loss += loss.numpy().item() * label.shape[0]
        correct += np.sum(np.argmax(out.numpy(), axis=1) == label.numpy()).item()

        rate = (step + 1) / len(dataloader)
        a = "*" * int(rate * 50)
        b = "." * int((1-rate) * 50)
        print("\reval loss: {:^3.0f}%[{}->{}]{:.3f}  correct:{}".format(int(rate*100), a, b, loss.numpy().item(), correct), end="")
    print()

    return correct/num_sample, total_loss/num_sample
    ### END YOUR SOLUTION



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    print()

    if opt:
        train_ptb(model, data, seq_len=seq_len, optimizer=opt, clip=clip, loss_fn=loss_fn, device=device, dtype=dtype)
    else:
        evaluate_ptb(model, data, seq_len=seq_len, loss_fn=loss_fn, device=device, dtype=dtype)
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = loss_fn()
    num_step = data.shape[0] - 1
    model.train()

    for _ in range(n_epochs):
        error, total_loss, num_sample, state = 0, 0, 0, None
        for step in range(0, num_step, seq_len):
            X, y = ndl.data.get_batch(data, step, seq_len, device=device, dtype=dtype)
            out, state = model(X, state)
            loss = loss_fn(out, y)
            if isinstance(state, tuple):
                state = tuple([s.data for s in list(state)])
            else:
                state = state.data

            opt.reset_grad()
            loss.backward()

            if clip:
                opt.clip_grad_norm(max_norm=clip)
            
            opt.step()


            num_sample += y.shape[0]
            total_loss += loss.numpy().item() * y.shape[0]
            error += np.sum(np.argmax(out.numpy(), axis=1) != y.numpy()).item()

            rate = ((step + 1) // seq_len) / (num_step // seq_len)
            a = "*" * int(rate * 50)
            b = "." * int((1-rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f} process: {}/{} error:{}".format(int(rate*100), a, b, loss.numpy().item(), (step+1) // seq_len, num_step // seq_len, error), end="")
        print()

        # for p in model.parameters():
        #     if len(p.shape) == 2 and p.shape[1] == 30:
        #         print(p.grad.sum(axes=0))

    return error/num_sample, total_loss/num_sample
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    error, total_loss, num_sample, state = 0, 0, 0, None
    loss_fn = loss_fn()
    num_step = data.shape[0] - 1
    model.eval()

 
    for step in range(0, num_step, seq_len):
        X, y = ndl.data.get_batch(data, step, seq_len, device=device, dtype=dtype)
        out, state = model(X, state)
        loss = loss_fn(out, y)
        
        num_sample += y.shape[0]
        total_loss += loss.numpy().item() * y.shape[0]
        error += np.sum(np.argmax(out.numpy(), axis=1) != y.numpy()).item()

        rate = ((step + 1) // seq_len) / (num_step // seq_len)
        a = "*" * int(rate * 50)
        b = "." * int((1-rate) * 50)
        print("\reval loss: {:^3.0f}%[{}->{}]{:.3f} process: {}/{} error:{}".format(int(rate*100), a, b, loss.numpy().item(), (step+1) // seq_len, num_step // seq_len, error), end="")
    print()

    return error/num_sample, total_loss/num_sample
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
