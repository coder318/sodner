import logging
from typing import Any, Dict, List, Tuple, Union
import json

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import (ListField, TextField, MetadataField, IndexField,
                                  SequenceLabelField, AdjacencyField)
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.dataset_readers.dataset_utils import enumerate_spans

from allennlp.data.fields.span_field import SpanField

logger = logging.getLogger(__name__)

class MissingDict(dict):

    def __init__(self, missing_val, generator=None) -> None:
        if generator:
            super().__init__(generator)
        else:
            super().__init__()
        self._missing_val = missing_val

    def __missing__(self, key):
        return self._missing_val


def format_label_fields(sentence: List[str],
                        ner: List[List[Union[int,str]]],
                        relations: List[List[Union[int,str]]],
                        sentence_start: int,
                        tree: Dict[str, Any],
                        dep_tree: Dict[str, Any],
                        tf: Dict[str, Any],
                        tree_feature_dict: List[str],
                        use_overlap_rel: bool) -> Tuple[Dict[Tuple[int,int],str],
                                                      Dict[Tuple[Tuple[int,int],Tuple[int,int]],str],
                                                      Dict[Tuple[int,int],int], Dict[Tuple[int, int],str],
                                                    Dict[Tuple[int, int],List[Tuple[int, int]]],
                                                    Dict[str,Any]]:

    ss = sentence_start
    # NER
    ner_dict = MissingDict("",
        (
            ((span_start-ss, span_end-ss), named_entity)
            for (span_start, span_end, named_entity) in ner
        )
    )

    # Relations
    relation_dict_values = []
    for (span1_start, span1_end, span2_start, span2_end, relation) in relations:
        if relation == 'Overlap' and not use_overlap_rel:
            continue
        relation_dict_values.append((((span1_start - ss, span1_end - ss), (span2_start - ss, span2_end - ss)), relation))
    relation_dict = MissingDict("", relation_dict_values)

    # Children in Syntax tree
    if 'match' in tree:
        children_dict = MissingDict([],
            (
                ((syntax_span[0], syntax_span[1]), [(tree['nodes'][child][4][0],tree['nodes'][child][4][1]) for child in children])
                for parent, children, syntax, word, syntax_span in tree['nodes']
            )
        )
    else:
        children_dict = MissingDict([])

    if 'nodes' in dep_tree:
        dep_children_dict = MissingDict("",
                                    (
                                        ((node_idx, adj_node_idx), "1")
                                        for node_idx, adj_node_idxes in enumerate(dep_tree['nodes']) for adj_node_idx in adj_node_idxes
                                    )
                                    )
    else:
        dep_children_dict = MissingDict("")

    if len(tree_feature_dict) != 0:
        missingdict_values = []
        for i, _ in enumerate(sentence):
            for j, _ in enumerate(sentence):
                feature = ""
                for k in tree_feature_dict:
                    if k in tf:
                        feature += "#"+tf[k][i][j]
                missingdict_values.append(((i, j), feature))

        tf_dict = MissingDict("", missingdict_values)

    else:
        tf_dict = MissingDict("")

    return ner_dict, relation_dict, children_dict, dep_children_dict, tf_dict


@DatasetReader.register("ie_json")
class IEJsonReader(DatasetReader):

    def __init__(self,
                 max_span_width: int,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 context_width: int = 1,
                 debug: bool = False,
                 lazy: bool = False,
                 label_scheme: str = 'flat',
                 tree_span_filter: bool = False,
                 tree_match_filter: bool = False,
                 tree_feature_dict: List[str] = None,
                 use_overlap_rel: bool = False) -> None:
        super().__init__(lazy)
        assert (context_width % 2 == 1) and (context_width > 0)
        self.k = int( (context_width - 1) / 2)
        self._max_span_width = max_span_width
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._debug = debug
        self._n_debug_docs = 2
        self._label_scheme = label_scheme
        self._tree_span_filter = tree_span_filter
        self._tree_match_filter = tree_match_filter
        self._tree_feature_dict = tree_feature_dict
        self.use_overlap_rel = use_overlap_rel

    @overrides
    def _read(self, file_path: str):

        with open(file_path, "r") as f:
            lines = f.readlines()
        if self._debug:
            lines = lines[:self._n_debug_docs]


        for line in lines:
            # Loop over the documents.
            sentence_start = 0
            js = json.loads(line)
            doc_key = js["doc_key"]
            dataset = js["dataset"] if "dataset" in js else None

            n_sentences = len(js["sentences"])
            js["sentence_groups"] = [[self._normalize_word(word) for sentence in js["sentences"][max(0, i-self.k):min(n_sentences, i + self.k + 1)] for word in sentence] for i in range(n_sentences)]
            js["sentence_start_index"] = [sum(len(js["sentences"][i-j-1]) for j in range(min(self.k, i))) if i > 0 else 0 for i in range(n_sentences)]
            js["sentence_end_index"] = [js["sentence_start_index"][i] + len(js["sentences"][i]) for i in range(n_sentences)]
            for sentence_group_nr in range(len(js["sentence_groups"])):
                if len(js["sentence_groups"][sentence_group_nr]) > 300:
                    js["sentence_groups"][sentence_group_nr] = js["sentences"][sentence_group_nr]
                    js["sentence_start_index"][sentence_group_nr] = 0
                    js["sentence_end_index"][sentence_group_nr] = len(js["sentences"][sentence_group_nr])
                    if len(js["sentence_groups"][sentence_group_nr])>300:
                        import ipdb;

            for field in ["ner", "relations", 'trees', 'dep', 'tf']:
                if field not in js:
                    js[field] = [[] for _ in range(n_sentences)]

            zipped = zip(js["sentences"], js["ner"], js["relations"], js["sentence_groups"],
                         js["sentence_start_index"], js["sentence_end_index"], js['trees'], js['dep'], js['tf'])

            # Loop over the sentences.
            for sentence_num, (sentence, ner, relations, groups, start_ix, end_ix, tree, dep, tf) in enumerate(zipped):

                ner_dict, relation_dict, children_dict, dep_children_dict, tf_dict \
                    = format_label_fields(sentence, ner, relations, sentence_start, tree, dep, tf, self._tree_feature_dict, self.use_overlap_rel)
                sentence_start += len(sentence)
                instance = self.text_to_instance(
                    sentence, ner_dict, relation_dict,
                    doc_key, dataset, sentence_num, groups, start_ix, end_ix, tree, children_dict, dep_children_dict, tf_dict)
                yield instance


    @overrides
    def text_to_instance(self,
                         sentence: List[str],
                         ner_dict: Dict[Tuple[int, int], str],
                         relation_dict,
                         doc_key: str,
                         dataset: str,
                         sentence_num: int,
                         groups: List[str],
                         start_ix: int,
                         end_ix: int,
                         tree: Dict[str, Any],
                         children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],
                         dep_children_dict: Dict[Tuple[int, int],List[Tuple[int, int]]],
                         tf_dict: Dict[Tuple[int, int], Any]):

        sentence = [self._normalize_word(word) for word in sentence]

        text_field = TextField([Token(word) for word in sentence], self._token_indexers)
        text_field_with_context = TextField([Token(word) for word in groups], self._token_indexers)


        # Put together the metadata.
        metadata = dict(sentence=sentence,
                        ner_dict=ner_dict,
                        relation_dict=relation_dict,
                        doc_key=doc_key,
                        dataset=dataset,
                        groups=groups,
                        start_ix=start_ix,
                        end_ix=end_ix,
                        sentence_num=sentence_num,
                        tree=tree,
                        children_dict=children_dict,
                        dep_children_dict=dep_children_dict
                        )
        metadata_field = MetadataField(metadata)

        # Generate fields for text spans, ner labels
        spans = []
        span_ner_labels = []
        span_children_labels = []
        raw_spans = []

        for start, end in enumerate_spans(sentence, max_span_width=self._max_span_width):
            span_ix = (start, end)
            span_ner_labels.append(ner_dict[span_ix])
            spans.append(SpanField(start, end, text_field))
            raw_spans.append(span_ix)

        span_field = ListField(spans)

        for span in raw_spans:

            if len(children_dict[span]) == 0:
                children_field = ListField([IndexField(-1, span_field) ])
            else:
                children_field = []
                for children_span in children_dict[span]:
                    if children_span in raw_spans:
                        children_field.append(IndexField(raw_spans.index(children_span), span_field))
                    else:
                        children_field.append(IndexField(-1, span_field))
                children_field = ListField(children_field)

            span_children_labels.append(children_field)


        n_tokens = len(sentence)
        candidate_indices = [(i, j) for i in range(n_tokens) for j in range(n_tokens)]
        dep_adjs = []
        dep_adjs_indices = []
        tf_indices = []
        tf_features = []
        for token_pair in candidate_indices:
            dep_adj_label = dep_children_dict[token_pair]
            if dep_adj_label:
                dep_adjs_indices.append(token_pair)
                dep_adjs.append(dep_adj_label)

            feature = tf_dict[token_pair]
            if feature:
                tf_indices.append(token_pair)
                tf_features.append(feature)


        ner_label_field = SequenceLabelField(span_ner_labels, span_field,
                                             label_namespace="ner_labels")


        n_spans = len(spans)
        span_tuples = [(span.span_start, span.span_end) for span in spans]
        candidate_indices = [(i, j) for i in range(n_spans) for j in range(n_spans)]

        relations = []
        relation_indices = []
        for i, j in candidate_indices:
            span_pair = (span_tuples[i], span_tuples[j])
            relation_label = relation_dict[span_pair]
            if relation_label:
                relation_indices.append((i, j))
                relations.append(relation_label)

        relation_label_field = AdjacencyField(
            indices=relation_indices, sequence_field=span_field, labels=relations,
            label_namespace="relation_labels")

        # Syntax
        span_children_field = ListField(span_children_labels)
        dep_span_children_field = AdjacencyField(
            indices=dep_adjs_indices, sequence_field=text_field, labels=dep_adjs,
            label_namespace="dep_adj_labels")

        tf_field = AdjacencyField(indices=tf_indices, sequence_field=text_field, labels=tf_features,
                                     label_namespace="tf_labels")

        fields = dict(text=text_field_with_context,
                      spans=span_field,
                      ner_labels=ner_label_field,
                      relation_labels=relation_label_field,
                      metadata=metadata_field,
                      span_children=span_children_field,
                      dep_span_children=dep_span_children_field,
                      tf=tf_field)

        return Instance(fields)

    @overrides
    def _instances_from_cache_file(self, cache_filename):
        pass

    @overrides
    def _instances_to_cache_file(self, cache_filename, instances):
        pass


    @staticmethod
    def _normalize_word(word):
        if word == "/." or word == "/?":
            return word[1:]
        else:
            return word

