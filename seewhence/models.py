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
    
    def load_weights(self, pretrained_embeddings, PAD_IDX, UNK_IDX, EMBEDDING_DIM):
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
    
    def weight_reset(self):
        for name, m in self.named_children():
            if isinstance(m, nn.LSTM) or isinstance(m, nn.Linear):
                m.reset_parameters()
    
    
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
    
class HashtagEmojiLSTM(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 embedding_dim, 
                 hidden_dim, 
                 output_dim, 
                 n_layers, 
                 bidirectional, 
                 dropout, 
                 pad_idx, 
                 emoji_vocab_size, 
                 emoji_embedding_dim,
                 hashtag_vocab_size,
                 hashtag_embedding_dim,
                 embedding_dropout=0
                
                ):
        super().__init__()
        
        self.text_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
        self.emoji_embedding = nn.Embedding(emoji_vocab_size, emoji_embedding_dim, padding_idx = pad_idx)
        self.hashtag_embedding = nn.Embedding(hashtag_vocab_size, hashtag_embedding_dim, padding_idx = pad_idx)
        
        self.lstm_dim = embedding_dim
        self.rnn = nn.LSTM(self.lstm_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(embedding_dropout)
    
        
    def forward(self, batch):
        text, text_lengths = batch.text
        emoji = batch.emoji
        hashtag = batch.hashtags
        
        text_embedded = self.text_embedding(text)
        emoji_embedded = self.emoji_embedding(emoji)
        hashtag_embedded = self.hashtag_embedding(hashtag)

        embedded = torch.cat([text_embedded,emoji_embedded, hashtag_embedded])
        
        self.dropout(embedded)
        
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
        
        packed_output, (hidden, cell) = self.rnn(packed_embedded)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        linear = self.fc(hidden)
        
        return linear
    
       
    def load_weights(self,text_weights, emoji_weights, hashtag_weights, UNK_IDX, PAD_IDX, EMBEDDING_DIM=300):
        self.text_embedding.weight.data.copy_(text_weights)
        self.text_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.text_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        self.emoji_embedding.weight.data.copy_(emoji_weights)
        self.emoji_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.emoji_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

        self.hashtag_embedding.weight.data.copy_(hashtag_weights)
        self.hashtag_embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
        self.hashtag_embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)
        
    def weight_reset(self):
        for name, m in self.named_children():
            if isinstance(m, nn.LSTM) or isinstance(m, nn.Linear):
                m.reset_parameters()
#     class GenerateTweets(nn.Module):
#         def __init__(self, n_vocab, seq_size, embedding_size, lstm_size):
#             super(RNNModule, self).__init__()
#             self.seq_size = seq_size
#             self.lstm_size = lstm_size
#             self.embedding = nn.Embedding(n_vocab, embedding_size)
#             self.lstm = nn.LSTM(embedding_size,
#                                 lstm_size,
#                                 batch_first=False)
#             self.dense = nn.Linear(lstm_size, n_vocab)
#         def forward(self, x, prev_state):
#             embed = self.embedding(x)
#             output, state = self.lstm(embed, prev_state)
#             logits = self.dense(output)

#             return logits, state
#         def zero_state(self, batch_size):
#             return (torch.zeros(1, batch_size, self.lstm_size),
#                     torch.zeros(1, batch_size, self.lstm_size))