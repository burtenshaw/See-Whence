import torch.nn as nn
import torch.nn.functional as F
import torch 

class TextSentiment(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

class FastText(nn.Module):
    def __init__(self, 
                 vocab_size,
                 embedding_dim, 
                 output_dim, 
                 pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        
        self.fc = nn.Linear(embedding_dim, output_dim)
        
    def forward(self, batch):
        
        embedded = self.embedding(batch.text)
        embedded = embedded.permute(1, 0, 2)
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1) 
        return self.fc(pooled)
    
class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        
        self.rnn = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch):
        text, text_lengths = batch.text
        embedded = self.dropout(self.embedding(text))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc(hidden)
    
    
class EmojiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, emoji_vocab_size, emoji_embedding_dim):
        super().__init__()
        
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.emoji_embedding = nn.Embedding(emoji_vocab_size, emoji_embedding_dim, padding_idx = pad_idx)
        self.lstm_dim = embedding_dim
        self.rnn = nn.LSTM(self.lstm_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch):
        text, text_lengths = batch.text
        emoji = batch.emoji
        
        text_embedded = self.dropout(self.text_embedding(text))
        emoji_embedded = self.emoji_embedding(emoji)
        embedded = torch.cat([text_embedded,emoji_embedded])
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
            
        return self.fc(hidden)
    
    
class BERTGRUSentiment(nn.Module):
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        
        super().__init__()
        
        self.bert = bert
        
        embedding_dim = bert.config.to_dict()['hidden_size']
        
        self.rnn = nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)
        
#         self.rnn = nn.LSTM(embedding_dim, 
#                            hidden_dim, 
#                            num_layers=n_layers, 
#                            bidirectional=bidirectional,
#                            batch_first = True,
#                            dropout=dropout)
        
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, batch):
        
        #text = [batch size, sent len]
                
        with torch.no_grad():
            embedded = self.bert(batch.text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        
        _, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
                
        #hidden = [batch size, hid dim]
        
        output = self.out(hidden)
        
        #output = [batch size, out dim]
        
        return output
    
class LSTM(nn.Module):
    def __init__(self, 
                 vocab_size,
                 emb_dim,
                 hidden_dim,
                 output_dim,
                 num_linear=1, 
                 dropout=0,
                 lstm_layers=1
                ):
        super().__init__() # don't forget to call this!
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        lstm_layers = 1 if dropout != 0 else dropout
        self.encoder = nn.LSTM(emb_dim, hidden_dim, num_layers=1, dropout=dropout)
        self.linear_layers = []
        for _ in range(num_linear - 1):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.linear_layers = nn.ModuleList(self.linear_layers)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        hidden, _ = self.encoder(self.embedding(batch.text[0]))
        feature = hidden[-1, :, :]
        
        for layer in self.linear_layers:
          feature = layer(feature)
        
        output = self.out(feature)
                
        return output