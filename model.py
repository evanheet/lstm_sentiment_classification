import torch
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors

#Define network architecture
class Architecture(nn.Module):
    def __init__(self, root, train, validation, test, embeddings, hidden_dim, hidden_layers, output_dim, dropout):
        super().__init__()
     
        # Set up fields
        TEXT = data.Field()
        LABEL = data.Field(sequential=False,dtype=torch.long)

        # Make splits for data
        train, val, test = datasets.SST.splits(TEXT, LABEL, root, train, validation, test)

        # Build the vocabulary
        TEXT.build_vocab(train, vectors=Vectors(name=embeddings))
        LABEL.build_vocab(train)
        
        # Make iterator for splits
        self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits((train, val, test), batch_size=64)

        # Get pretrained embeddings, dimensions, and training parameters
        pretrained_embeddings = TEXT.vocab.vectors
        input_dim = len(TEXT.vocab)
        embedding_dim = pretrained_embeddings.shape[1]
        self.train_iterations = len(self.train_iter)
        self.val_iterations = len(self.val_iter)
        
        # Copy embeddings and define laters
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.embedding.weight.data.copy_(pretrained_embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, hidden_layers, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, hidden = self.lstm(embedded)
        output = self.dropout(output)
        return self.fc(output[-1])