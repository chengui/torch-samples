import re
import math
import time
import torch
import random
import collections
from torch.nn.functional import one_hot


def preprocess(line):
    return re.sub('[^A-Za-z]+', ' ', line).strip().lower()

def read_time_machine():
    with open('timemachine.txt', 'r') as f:
        lines = f.readlines()
        lines = [preprocess(line) for line in lines]
        return lines

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]

class Vocab:
    def __init__(self, tokens=[], min_freq=0, reserved_tokens=[]):
        counter = collections.Counter(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        uniq_tokens = filter(lambda x: x[1] >= min_freq, self._token_freqs)
        uniq_tokens = list(map(lambda x: x[0], uniq_tokens))
        self.idx2token = ['<unk>'] + reserved_tokens + uniq_tokens
        self.token2idx = {v: k for k, v in enumerate(self.idx2token)}

    def __len__(self):
        return len(self.idx2token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token2idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx2token[indices]
        return [self.idx2token[index] for index in indices]

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def load_corpus_time_machine(max_tokens=-1):
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    tokens = [token for line in tokens for token in line]
    vocab = Vocab(tokens)

    corpus = [vocab[token] for token in tokens]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

def seq_data_iter_random(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps-1)
    num_batches = (len(corpus) - offset - 1) // (num_steps * batch_size)
    num_tokens = num_steps * batch_size * num_batches
    Xs = torch.tensor(corpus[offset:offset+num_tokens]).reshape(-1, num_steps)
    Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens]).reshape(-1, num_steps)
    idx = torch.randperm(Xs.shape[0])
    Xs = Xs[idx,:].view(batch_size, num_batches, num_steps)
    Ys = Ys[idx,:].view(batch_size, num_batches, num_steps)
    for i in range(num_batches):
        X = Xs[:,i,:].view(batch_size, num_steps)
        Y = Ys[:,i,:].view(batch_size, num_steps)
        yield X, Y

def seq_data_iter_sequential(corpus, batch_size, num_steps):
    offset = random.randint(0, num_steps-1)
    num_batches = (len(corpus) - offset - 1) // (num_steps * batch_size)
    num_tokens = num_steps * batch_size * num_batches
    Xs = torch.tensor(corpus[offset:offset+num_tokens]).reshape(-1, num_steps)
    Ys = torch.tensor(corpus[offset+1:offset+1+num_tokens]).reshape(-1, num_steps)
    Xs = Xs.view(batch_size, num_batches, num_steps)
    Ys = Ys.view(batch_size, num_batches, num_steps)
    for i in range(num_batches):
        X = Xs[:,i,:].view(batch_size, num_steps)
        Y = Ys[:,i,:].view(batch_size, num_steps)
        yield X, Y

class SeqDataLoader:
    def __init__(self, batch_size, num_steps, user_random_iter, max_tokens):
        if user_random_iter:
            self._data_iter_fn = seq_data_iter_random
        else:
            self._data_iter_fn = seq_data_iter_sequential
        self.corpus, self.vocab = load_corpus_time_machine(max_tokens)
        self.batch_size, self.num_steps = batch_size, num_steps

    def __iter__(self):
        return self._data_iter_fn(self.corpus, self.batch_size, self.num_steps)

def load_data_time_machine(batch_size, num_steps, use_random_iter=False, max_tokens=10000):
    data_iter = SeqDataLoader(batch_size, num_steps, use_random_iter, max_tokens)
    return data_iter, data_iter.vocab

#corpus, vocab = load_corpus_time_machine()
#print(len(corpus), len(vocab))
#my_seq = list(range(35))
#for X, Y in seq_data_iter_random(my_seq, batch_size=3, num_steps=5):
#    print('X: ', X, '\nY: ', Y)
#data_iter, vocab = load_data_time_machine(32, 35)
#for X, Y in data_iter:
#    print(X, Y)
class RNN:
    def __init__(self, input_size, num_hiddens):
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.params = self.get_params(input_size, num_hiddens)

    def parameters(self):
        return self.params

    def get_params(self, input_size, num_hiddens):
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        num_inputs = num_outputs = input_size
        # hidden layer
        w_xh = normal((num_inputs, num_hiddens))
        w_hh = normal((num_hiddens, num_hiddens))
        b_h = torch.zeros(num_hiddens)
        # output layer
        w_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs)
        # gradient
        params = [w_xh, w_hh, b_h, w_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def __call__(self, X, state):
        X = one_hot(X.T, self.input_size).type(torch.float32)
        w_xh, w_hh, b_h, w_hq, b_q = self.params
        H = state
        outputs = []
        for x in X:
            H = torch.tanh(torch.mm(x, w_xh) + torch.mm(H, w_hh) + b_h)
            y = torch.mm(H, w_hq) + b_q
            outputs.append(y)
        return torch.cat(outputs, dim=0), H

    def begin_state(self, batch_size):
        return torch.zeros((batch_size, self.num_hiddens))

class RNN2(torch.nn.Module):
    def __init__(self, input_size, num_hiddens):
        super().__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.rnn = torch.nn.RNN(input_size, num_hiddens)
        self.linear = torch.nn.Linear(num_hiddens, input_size)

    def forward(self, X, state):
        X = one_hot(X.T, self.input_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens))

class GRU:
    def __init__(self, input_size, num_hiddens):
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.params = self.get_params(input_size, num_hiddens)

    def parameters(self):
        return self.params

    def get_params(self, input_size, num_hiddens):
        num_inputs = num_outputs = input_size
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        def three():
            return (
                normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens)
            )
        # update gate
        w_xz, w_hz, b_z = three()
        # reset gate
        w_xr, w_hr, b_r = three()
        # tilda
        w_xh, w_hh, b_h = three()
        # output layer
        w_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs)
        # gradient
        params = [w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def __call__(self, inputs, state):
        inputs = one_hot(inputs.T, self.input_size).type(torch.float32)
        w_xz, w_hz, b_z, w_xr, w_hr, b_r, w_xh, w_hh, b_h, w_hq, b_q = self.params
        H = state
        outputs = []
        for X in inputs:
            Z = torch.sigmoid((X @ w_xz) + (H @ w_hz) + b_z)
            R = torch.sigmoid((X @ w_xr) + (H @ w_hr) + b_r)
            H_tilda = torch.tanh((X @ w_xh) + ((R * H) @ w_hh) + b_h)
            H = Z * H + (1 - Z) * H_tilda
            Y = H @ w_hq + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), H

    def begin_state(self, batch_size):
        return torch.zeros((batch_size, self.num_hiddens))

class GRU2(torch.nn.Module):
    def __init__(self, input_size, num_hiddens):
        super().__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.rnn = torch.nn.GRU(input_size, num_hiddens)
        self.linear = torch.nn.Linear(num_hiddens, input_size)

    def forward(self, X, state):
        X = one_hot(X.T, self.input_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        return torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens))

class LSTM:
    def __init__(self, input_size, num_hiddens):
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.params = self.get_params(input_size, num_hiddens)

    def parameters(self):
        return self.params

    def get_params(self, input_size, num_hiddens):
        num_inputs = num_outputs = input_size
        def normal(shape):
            return torch.randn(size=shape) * 0.01
        def three():
            return (
                normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens)
            )
        # input gate
        w_xi, w_hi, b_i = three()
        # forget gate
        w_xf, w_hf, b_f = three()
        # output gate
        w_xo, w_ho, b_o = three()
        # tilda
        w_xc, w_hc, b_c = three()
        # output layer
        w_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs)
        # gradient
        params = [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o,
                  w_xc, w_hc, b_c, w_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def __call__(self, inputs, state):
        inputs = one_hot(inputs.T, self.input_size).type(torch.float32)
        [w_xi, w_hi, b_i, w_xf, w_hf, b_f, w_xo, w_ho, b_o,
         w_xc, w_hc, b_c, w_hq, b_q] = self.params
        (H, C) = state
        outputs = []
        for X in inputs:
            I = torch.sigmoid((X @ w_xi) + (H @ w_hi) + b_i)
            F = torch.sigmoid((X @ w_xf) + (H @ w_hf) + b_f)
            O = torch.sigmoid((X @ w_xo) + (H @ w_ho) + b_o)
            C_tilda = torch.tanh((X @ w_xc) + (H @ w_hc) + b_c)
            C = F * C + I * C_tilda
            H = O * torch.tanh(C)
            Y = (H @ w_hq) + b_q
            outputs.append(Y)
        return torch.cat(outputs, dim=0), (H, C)

    def begin_state(self, batch_size):
        return (torch.zeros((batch_size, self.num_hiddens)),
                torch.zeros((batch_size, self.num_hiddens)))

class LSTM2(torch.nn.Module):
    def __init__(self, input_size, num_hiddens, num_layers=2, bidirectional=True):
        super().__init__()
        self.input_size = input_size
        self.num_hiddens = num_hiddens
        self.rnn = torch.nn.LSTM(input_size, num_hiddens, num_layers, bidirectional=bidirectional)
        if bidirectional:
            num_hiddens *= 2
        self.linear = torch.nn.Linear(num_hiddens, input_size)

    def forward(self, X, state):
        X = one_hot(X.T, self.input_size).type(torch.float32)
        Y, state = self.rnn(X, state)
        output = self.linear(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, batch_size=1):
        return (torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens)),
                torch.zeros((self.rnn.num_layers, batch_size, self.num_hiddens)))

def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad**2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def predict(prefix, num_preds, net, vocab):
    h_state = net.begin_state(1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor([outputs[-1]]).reshape((1, 1))
    for y in prefix[1:]:
        _, h_state = net(get_input(), h_state)
        outputs.append(vocab[y])
    for _ in range(num_preds):
        y, h_state = net(get_input(), h_state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join(vocab.to_tokens(outputs))

def train_epoch(net, train_iter, loss_fn, optimizer):
    h_state = None
    sum_loss, num_words, start_time = 0, 0, time.time()
    for X, Y in train_iter:
        if h_state is None:
            h_state = net.begin_state(X.shape[0])
        y = Y.T.reshape(-1)
        y_hat, h_state = net(X, h_state)
        if isinstance(h_state, tuple):
            for s in h_state:
                s.detach_()
        else:
            h_state.detach_()
        loss = loss_fn(y_hat, y.long()).mean()
        optimizer.zero_grad()
        loss.backward()
        grad_clipping(net, 1)
        optimizer.step()
        sum_loss += loss.data * y.numel()
        num_words += y.numel()
    rate_words = num_words / (time.time() - start_time)
    return math.exp(sum_loss/num_words), rate_words

def train(net, train_iter, vocab, lr, num_epochs):
    _predict = lambda prefix: predict(prefix, 50, net, vocab)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr)
    for epoch in range(num_epochs):
        ppl, speed = train_epoch(net, train_iter, loss, optimizer)
        if (epoch + 1) % 10 == 0:
            print(f'epoch {epoch+1}: ppl={ppl:.1f}, speed={speed:.1f}')
            print(_predict('time traveller'))
    print(_predict('time traveller'))
    print(_predict('traveller'))

batch_size = 32
num_steps = 35
train_iter, vocab = load_data_time_machine(batch_size, num_steps)

input_size = len(vocab)
num_hiddens = 256
#num_hiddens = 512
num_epochs, lr = 500, 2
#net = RNN2(input_size, num_hiddens)
#net = GRU2(input_size, num_hiddens)
net = LSTM2(input_size, num_hiddens)

train(net, train_iter, vocab, lr, num_epochs)
