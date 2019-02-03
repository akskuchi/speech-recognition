import argparse
import math
import time

import torch
import torch.nn as nn
import torch.onnx

import data
import model

# from nlgeval import NLGEval

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_batch(source, i):
    """
    get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    â”Œ a g m s â”� â”Œ b h n t â”�
    â”” b h n t â”˜ â”” c i o u â”˜
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM.
    :param source: source data batches
    :param i:
    :return:
    """
    seq_len = min(arguments.bptt, len(source) - 1 - i)
    _data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return _data, target


def construct_batches(tokens, bsz):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    â”Œ a g m s â”�
    â”‚ b h n t â”‚
    â”‚ c i o u â”‚
    â”‚ d j p v â”‚
    â”‚ e k q w â”‚
    â”” f l r x â”˜.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.

    :param tokens: loaded tokens
    :param bsz: batch size
    :return: tokens in batches
    """
    nbatch = tokens.size(0) // bsz
    tokens = tokens.narrow(0, 0, nbatch * bsz)
    tokens = tokens.view(bsz, -1).t().contiguous()
    return tokens.to(device)


def load_data(data_path):
    """
    load data from directory
    :param data_path: path to corpus
    :return: tokens for {train, validation and test sets}, overall dictionary
    """
    print("RNN Language MODEL: Loading gigaword corpus")
    return data.CorpusGigaword(data_path)


def repackage_hidden(h):
    """
    wraps hidden states in new Tensors, to detach them from their history.
    """
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batch_to_word_sequence(_data, corpus):
    data_npy = _data.detach().cpu().numpy()

    return [
               [corpus.dictionary.idx2word[w] for w in s]
               for s in data_npy.T
           ], data_npy.T.tolist()


def train(_model, corpus, train_data, criterion, args, epoch):
    _model.train()
    total_loss = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    if 'FFNN' not in args.model:
        hidden = _model.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        _data, targets = get_batch(train_data, i)
        if 'FFNN' not in args.model:
            hidden = repackage_hidden(hidden)
        _model.zero_grad()
        if 'FFNN' not in args.model:
            output, hidden = _model(_data, hidden)
        else:
            output = _model(_data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(_model.parameters(), args.clip)
        for p in _model.parameters():
            p.data.add_(-args.lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3f} | {batch:5f}/{len(train_data) / args.bptt:5f} batches '
                  f'| lr {args.lr:02.2f} | {(elapsed * 1000) / args.log_interval:5.2f} ms/batch '
                  f'| loss {cur_loss:5.2f} | {math.exp(cur_loss):8.2f} ppl')

            total_loss = 0
            start_time = time.time()

    return _model


def evaluate(_model, corpus, val_data, criterion, args):
    _model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if 'FFNN' not in args.model:
        hidden = _model.init_hidden(10)
    with torch.no_grad():
        for i in range(0, val_data.size(0) - 1, args.bptt):
            _data, targets = get_batch(val_data, i)
            if 'FFNN' not in args.model:
                output, hidden = _model(_data, hidden)
            else:
                output = _model(_data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(_data) * criterion(output_flat, targets).item()
            if 'FFNN' not in args.model:
                hidden = repackage_hidden(hidden)

    return total_loss / (len(val_data) - 1)


def main(args):
    """
    main, set the random seed manually for reproducibility
    :param args: language model arguments
    """
    torch.manual_seed(args.seed)
    corpus = load_data(args.data)

    train_data = construct_batches(corpus.train, args.batch_size)
    val_data = construct_batches(corpus.valid, 10)
    ntokens = len(corpus.dictionary)

    if 'FFNN' in args.model:
        _model = model.FFNNModel(ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    else:
        _model = model.RNNModelOriginal(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

    best_val_loss = None
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        _model = train(_model, corpus, train_data, nn.CrossEntropyLoss(), args, epoch)
        val_loss = evaluate(_model, corpus, val_data, nn.CrossEntropyLoss(), args)

        print('-' * 89 +
              f'\n| end of epoch {epoch:3d} | time: {(time.time() - epoch_start_time):5.2f}s '
              f'| valid loss {val_loss:5.2f} | valid ppl {math.exp(val_loss):8.2f}\n' +
              '-' * 89 + '\n')

        if not best_val_loss or val_loss < best_val_loss:
            torch.save(_model.state_dict(), args.save)
            best_val_loss = val_loss
        else:
            args.lr /= 4.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN Language Model for English Gigaword corpus')
    parser.add_argument('--data', type=str, default='../resources/gigaword_corpus',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='FFNN',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings (if 0: OneHot instead of embeddings)')
    parser.add_argument('--nhid', type=int, default=60,
                        help='number of hidden units per layer')
    parser.add_argument('--vthreshold', type=int, default=20,
                        help='minimum number of occurances to qualify as not rare')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=20,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--metrics-k', type=int, help='How many words to predict for metrics', default=3)
    parser.add_argument('--show-predictions-during-evaluation', action='store_true',
                        help='Whether to show predicted sentences during evaluation')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='../resources/models/gigaword_model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--save-statistics', type=str, default=None)

    arguments = parser.parse_args()
    main(arguments)
