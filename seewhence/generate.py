import torch.nn as nn
import torch.nn.functional as F
import torch 
import numpy as np

vocab_to_int = {}

def numericalize(sentence, vocab_to_int=vocab_to_int, seq_len=40):
    ints = [int(vocab_to_int[w]) for w in sentence.split()]
    return np.concatenate([np.array(ints), np.zeros(tags_len-len(ints))])

def vectorize(device, net, words, n_vocab, vocab_to_int, top_k=5,seq_size=40):
    
    net.eval()        

    state_h, state_c = net.zero_state(seq_size)
    state_h = state_h.to(device)
    state_c = state_c.to(device)
    
    x = numericalize(words, vocab_to_int)
    ix = torch.tensor(x).long().to(device)
    
    output, (state_h, state_c) = net(ix, (state_h, state_c))
#     print(output[0])
    _, top_ix = torch.topk(output[0], k=top_k)
    
    return top_ix

class RNN(nn.Module):
    def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
        super(RNN, self).__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size,
                            lstm_size,
                            batch_first=False)
        self.dense = nn.Linear(lstm_size, n_vocab)
    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)

        return logits, state
    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.lstm_size),
                torch.zeros(1, batch_size, self.lstm_size))