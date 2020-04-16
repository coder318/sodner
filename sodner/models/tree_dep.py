import logging
from typing import Optional

import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from allennlp.nn import InitializerApplicator, RegularizerApplicator


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TreeDep(Model):

    def __init__(self,
                 vocab: Vocabulary,
                 span_emb_dim: int,
                 feature_dim: int,
                 tree_prop: int = 1,
                 tree_dropout: float = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(TreeDep, self).__init__(vocab, regularizer)

        self._span_emb_dim = span_emb_dim
        assert span_emb_dim % 2 == 0

        self.layers = tree_prop

        self.W = torch.nn.ModuleList()
        self.gcn_drop = torch.nn.ModuleList()
        for layer in range(self.layers):
            self.W.append(torch.nn.Linear(span_emb_dim, span_emb_dim, bias=False))
            self.gcn_drop.append(torch.nn.Dropout(p=tree_dropout))

        self._A_network = TimeDistributed(torch.nn.Linear(span_emb_dim, feature_dim, bias=False))

        initializer(self)

    # adj: (batch, sequence, sequence)
    # text_embeddings: (batch, sequence, emb_dim)
    # text_mask: (batch, sequence)
    def forward(self, adj, text_embeddings, text_mask):

        denom = adj.sum(2).unsqueeze(2) + 1
        gcn_inputs = text_embeddings

        for l in range(self.layers):
            Ax = adj.bmm(gcn_inputs)
            AxW = self.W[l](Ax)
            AxW = AxW + self.W[l](gcn_inputs) # self loop
            AxW = AxW / denom
            gAxW = F.relu(AxW)
            gAxW = self.gcn_drop[l](gAxW)
            gcn_inputs = gAxW * text_mask.unsqueeze(-1)

        gcn_inputs = self._A_network(gcn_inputs)

        return gcn_inputs






