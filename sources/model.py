import torch
import torch.nn as nn
import numpy as np


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, emsize, nhid, nlayers, device, tie_weights=False,
                 dropout=0.00):
        super(RNNModel, self).__init__()
        self.device = device
        self.drop = nn.Dropout(dropout)
        self.embSize = emsize
        if self.embSize > 0:
            self.encoder = nn.Embedding(ntoken, self.embSize)
            ninp = self.embSize
        else:
            self.encoder = nn.Linear(ntoken, nhid)
            ninp = ntoken
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(nhid, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.ntoken = ntoken

    def init_weights(self):
        self.rnn.bias_ih_l0.data.zero_()
        self.rnn.weight_ih_l0.data = (
                torch.randn(self.rnn.weight_ih_l0.size()) * 0.1)
        self.rnn.bias_hh_l0.data.zero_()
        self.rnn.weight_hh_l0.data = (
                torch.randn(self.rnn.weight_hh_l0.size()) * 0.1)
        #
        self.decoder.bias.data.zero_()
        self.decoder.weight.data = (
                torch.randn(self.decoder.weight.size()) * 0.1)
        #        
        self.encoder.weight.data = (
                torch.randn(self.encoder.weight.size()) * 0.1)
        if self.embSize <= 0:
            # The linear encoder has bias but the embedding layer doesn't
            self.encoder.bias.data.zero_()

    def forward(self, input, hidden):
        if self.embSize > 0:
            inp = self.encoder(input)
        else:
            # to oneHot
            inputs = torch.empty(input.shape[0], input.shape[1], self.ntoken).to(self.device)
            for x in range(input.shape[0]):
                for y in range(input.shape[1]):
                    inputs[x, y, :] = torch.Tensor(np.arange(self.ntoken)).long().to(self.device) \
                                      == input[x, y]
            inp = self.encoder(inputs)
        output, hidden = self.rnn(inp, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class RNNModelOriginal(nn.Module):
    """
    model with the following components:
    - encoder (embeds inputs -> input embedding)
    - recurrent network (learns temporal relations betweens input sequences -> output embedding)
    - decoder (converts output embedding -> probabilities over vocabulary)
    """

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, bi_dir=False):
        super(RNNModelOriginal, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            assert nhid == ninp
            self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        if self.rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, bidirectional=bi_dir)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout, bidirectional=bi_dir)

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, _input, hidden):
        """
        performs forward pass through the network
        :param _input: shape [sequence_len, batch_size, embed_size] 
        :param hidden: [1, batch_size, hidden_size]
        :return: output -> [sequence_len, batch_size, vocab_size]
        """
        emb = self.drop(self.encoder(_input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)


class FFNNModel(nn.Module):
    """
        feed forward neural network with similar bells and whistles as the RNN original network (above)
    """

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(FFNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            assert nhid == ninp
            self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

        self.ffnn = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.ReLU(),
            nn.Linear(nhid, nhid)
        )

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, _input):
        """
        performs forward pass through the network
        :param _input: shape [sequence_len, batch_size, embed_size]
        :return: output -> [sequence_len, batch_size, vocab_size]
        """
        emb = self.drop(self.encoder(_input))
        output = self.ffnn(emb)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))
