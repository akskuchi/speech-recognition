import argparse
import torch
import torch.nn as nn

import math
import model
from train_validate import load_data, repackage_hidden, construct_batches
from jiwer import wer

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


def evaluate(_model, corpus, test_data, criterion, args):
    _model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    hidden = _model.init_hidden(10)
    with torch.no_grad():
        for i in range(0, test_data.size(0) - 1, args.bptt):
            _data, targets = get_batch(test_data, i)
            output, hidden = _model(_data, hidden)
            w_error_rate = wer(list(targets.numpy()), list(output.numpy()))
            output_flat = output.view(-1, ntokens)
            total_loss += len(_data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)

    return total_loss / (len(test_data) - 1)


def main(args):
    """
        main, set the random seed manually for reproducibility
        :param args: language model arguments
        """
    torch.manual_seed(args.seed)
    corpus = load_data(args.data)
    test_data = construct_batches(corpus.test, 10)
    ntokens = len(corpus.dictionary)

    _model = model.RNNModelOriginal(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    _model.load_state_dict(torch.load(args.save, map_location=lambda storage, loc: storage))

    test_loss = evaluate(_model, corpus, test_data, nn.CrossEntropyLoss(), args)
    print(f'loss: {test_loss:}, perplexity: {math.exp(test_loss):8.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch RNN Language Model for English Gigaword corpus')
    parser.add_argument('--data', type=str, default='../../data/gigaword_corpus',
                        help='location of the data corpus')
    parser.add_argument('--model', type=str, default='GRU',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings (if 0: OneHot instead of embeddings)')
    parser.add_argument('--nhid', type=int, default=60,
                        help='number of hidden units per layer')
    parser.add_argument('--vthreshold', type=int, default=20,
                        help='minimum number of occurances to qualify as not rare')
    parser.add_argument('--nlayers', type=int, default=1,
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
    parser.add_argument('--save', type=str, default='/Users/kaushiksurikuchi/Downloads/gigaword_60_20_1.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    parser.add_argument('--save-statistics', type=str, default=None)

    arguments = parser.parse_args()
    main(arguments)
