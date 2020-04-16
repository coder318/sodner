import logging
from typing import Optional

import torch

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TreeFeature(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 input_dim: int,
                 feature_dim: int,
                 dropout: float,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TreeFeature, self).__init__(vocab, regularizer)

        self._input_dim = input_dim
        self._feature_dim = feature_dim
        self._dropout = dropout

        self.tf_embedding = torch.nn.Embedding(vocab.get_vocab_size('tf_labels'), self._feature_dim)
        torch.nn.init.xavier_normal_(self.tf_embedding.weight, gain=3)

        self.W = torch.nn.Linear(self._input_dim, 1, bias=False)
        torch.nn.init.xavier_normal_(self.W.weight)
        self.dropout_layers = torch.nn.Dropout(p=self._dropout)
        self.softmax_layers = torch.nn.Softmax(dim=2)

        _W = torch.nn.Linear(self._input_dim, self._feature_dim, bias=False)
        torch.nn.init.xavier_normal_(_W.weight)
        self._A_network = TimeDistributed(_W)

        # initializer(self)

    # tf: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, input_dim)
    # text_mask: (batch, sequence)
    def forward(self, tf, text_embeddings, text_mask):

        tf_mask = (tf[:, :, :] >= 0).float()

        # (batch, sequence, sequence, feature_dim)
        A = self.tf_embedding((tf * tf_mask).long()) * tf_mask.unsqueeze(-1)

        x = text_embeddings

        # (batch, sequence, feature_dim, input_dim)
        Ax = torch.matmul(A.transpose(2, 3), x.unsqueeze(1))
        # Ax = Ax / seq_len
        # (batch, sequence, feature_dim, 1)
        AxW = self.W(Ax)
        # (batch, sequence, feature_dim, 1)
        g = self.softmax_layers(AxW)
        # (batch, sequence, input_dim)
        gAx = torch.matmul(g.transpose(2, 3), Ax).squeeze(2)
        gAx = self.dropout_layers(gAx)

        output = self._A_network(gAx)
        output = output * text_mask.unsqueeze(-1)
        return output






