
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import re
import itertools
import math
import os
import numpy as np


class EncoderRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0, bidirection=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirection
        self.gru = nn.GRU(input_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout), bidirectional=(True if self.bidirectional else False))


    def forward(self, input_seq, input_lengths, hidden=None):
        # input_seq : batch of input sentences; shpae=(max_lengh, batch_size)
        # input_lengths : list of sentence lengths corresponding to each sentence in the batch
        # hidden_state, of shape : (n_layers x num_directions, batch_size, hidden_size)
        
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
        # GRU pass
        outputs, hidden = self.gru(packed, hidden)

        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)

        # Sum bidirectional GRU outputs
        if(self.bidirectional==1):
            outputs = outputs[:,:,:self.hidden_size] + outputs[:, : , self.hidden_size:]
        else:
            outputs = outputs[:,:,:self.hidden_size] #+ outputs[:, : , self.hidden_size:]

        return outputs, hidden
        # outputs : the output features h_t from the last layuer o fthe GRU, for each timestep (sum of bidirectional outputs)
        # outputs shape = ( max_length, batch_size, hidden_size)
        # hidden : hidden state for the last timestep, of shape = ( n_layers x num_directions, batch_size, hidden_size)

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output):
        # Element-Wise Multiply the current target decoder state with the encoder output and sum them
        return torch.sum(hidden*encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        # hidden of shape: (1, batch_size, hidden_size)
        # encoder_outputs of shape: (max_length, batch_size, hidden_size)
        # (1, batch_size, hidden_size) x (max_length, batch_size, hidden_size) = (max_length, batch_size, hidden_size)

        # Calculate the attention weights (energies)
        attn_energies = self.dot_score(hidden, encoder_outputs)  #(max_length, batch_size)
        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()
        # Return the softmax normalized probability scores ( with added dimentsion)
        return F.softmax(attn_energies, dim =1).unsqueeze(1)


class DecoderRNN(torch.nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.attn_model= attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(output_size, hidden_size, n_layers, dropout=(0 if n_layers ==1 else dropout))
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step: one time step (one word) of input sequence batch; shpae=(1,batch_size)
        # last_hidden: final hidden layer of GUR; shpae=(n_layers x num_directions, batch_dize, hidden_size)
        # encoder_outputs: encoder model's output; shape=(max_length, batch_size, hidden_size)

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_step, last_hidden)
        # rnn_output of shape : (1, batch, num_directions * hidden_size)
        # hidden of shape: (num_layers * num_directions, batch, hidden_size)

        # Predict next word using Luong eq. 6
        output = self.out(rnn_output)
        # output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
        # output: softmax normaized tensor giving probabilities of each word being the correct next word in the decoder sequence
        # shape = ( batch_size, boc.num_words)
        # hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)


class LuongAttnDecoderRNN(torch.nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model= attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(output_size, hidden_size, n_layers, dropout=(0 if n_layers ==1 else dropout))
        self.concat = nn.Linear(hidden_size*2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        #pylint: disable-msg=too-many-arguments
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # input_step: one time step (one word) of input sequence batch; shpae=(1,batch_size)
        # last_hidden: final hidden layer of GUR; shpae=(n_layers x num_directions, batch_dize, hidden_size)
        # encoder_outputs: encoder model's output; shape=(max_length, batch_size, hidden_size)

        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(input_step, last_hidden)
        # rnn_output of shape : (1, batch, num_directions * hidden_size)
        # hidden of shape: (num_layers * num_directions, batch, hidden_size)

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        # (batch_size, 1, max_length) bmm with (batch_size, max_length, hidden) = (batch_size, 1, hidden)
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))

        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        # output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden
        # output: softmax normaized tensor giving probabilities of each word being the correct next word in the decoder sequence
        # shape = ( batch_size, boc.num_words)
        # hidden: final hidden state of GRU; shape=(n_layers x num_directions, batch_size, hidden_size)


class FinalModel(torch.nn.Module):
    def __init__(self, inp_dim):
        super(FinalModel, self).__init__()
        self.out = nn.Linear(inp_dim,1)

    def forward(self, input):
        output = self.out(input)
        return output


class lossModel(torch.nn.Module):
    def __init__(self):
        super(lossModel, self).__init__()
        self.out = nn.Linear(3,1,bias=False)
        self.relu1 = nn.ReLU(inplace=False)

    def forward(self, input):
        output = self.out(input)
        output = self.relu1(output)
        return output 