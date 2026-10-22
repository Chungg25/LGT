import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    
    # Sigmoid learning rate decay
    elif args.lradj == 'sigmoid':
        k = 0.5 # logistic growth rate
        s = 10  # decreasing curve smoothing rate
        w = 10  # warm-up coefficient
        lr_adjust = {epoch: args.learning_rate / (1 + np.exp(-k * (epoch - w))) - args.learning_rate / (1 + np.exp(-k/s * (epoch - w*s)))}
    
    elif args.lradj == 'inverse_sqrt':
        t = float(epoch)
        # compute k so that S(w) = p_peak
        eps = 1e-9
        w
        p = min(max(0.95, eps), 1.0-eps)
        k = -2.0 / float(w) * math.log(1.0 / p - 1.0)
        def S(x, kk, cc):
            return 1.0 / (1.0 + math.exp(-kk * (x - cc)))
        S1 = S(t, k, w)
        S2 = S(t, k / s, s * w)
        m = max(0.0, S1 - S2)
        # normalize over [0..T]
        denom = 0.0
        for tt in range(0, T+1):
            denom = max(denom, S(tt, k, w) - S(tt, k / s, s * w))
        denom = denom if denom > 1e-12 else 1.0
        mult = m / denom
        lr_adjust = {epoch: args.learning_rate * S * D}

    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
 
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def test_params_memory(model, x_shape, device='cuda:0'):
    batch_size = 1
    seq_len, c_in = x_shape
    pred_len = getattr(model, 'pred_len', 96)
    label_len = getattr(model, 'label_len', 48)

    dummy_x = torch.randn(batch_size, seq_len, c_in).to(device)
    dummy_x_mark = torch.randn(batch_size, seq_len, 4).to(device)
    dummy_dec_inp = torch.randn(batch_size, label_len + pred_len, c_in).to(device)
    dummy_y_mark = torch.randn(batch_size, label_len + pred_len, 4).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        _ = model(dummy_x, dummy_x_mark, dummy_dec_inp, dummy_y_mark)
    torch.cuda.synchronize()
    max_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameter count: {num_params / 1e6:.2f}M")
    print(f"[MEMORY] Max memory allocated during inference: {max_memory:.2f} MB")