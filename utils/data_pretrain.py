import pickle
import os
import numpy as np
import torch
from transformers import (OpenAIGPTTokenizer, BertTokenizer, XLNetTokenizer, RobertaTokenizer, AutoTokenizer)
try:
    from transformers import AlbertTokenizer
except:
    pass
from modeling import units, rnn_tools

import json
from tqdm import tqdm

GPT_SPECIAL_TOKENS = ['_start_', '_delimiter_', '_classify_']


class MultiGPUSparseAdjDataBatchGenerator(object):
    def __init__(self, args, mode, device0, device1, batch_size, indexes, qids, HF_labels, Diag_labels,
                 diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2,
                 tensors0, tensors1, adj_data=None):
        self.args = args
        self.mode = mode
        self.device0 = device0
        self.device1 = device1
        self.batch_size = batch_size
        self.indexes = indexes
        self.qids = qids
        self.HF_labels = HF_labels
        self.Diag_labels = Diag_labels
        self.diagnosis_codes = diagnosis_codes
        self.seq_time_step = seq_time_step
        self.seq_time_step2 = seq_time_step2
        self.mask_mult = mask_mult
        self.mask_final = mask_final
        self.mask_code = mask_code
        self.lengths = lengths
        self.tensors0 = tensors0  # encoder_data
        self.tensors1 = tensors1
        self.adj_data = adj_data

    def __len__(self):
        return (self.indexes.size(0) - 1) // self.batch_size + 1

    # 盲猜这个用来作循环
    def __iter__(self):
        bs = self.batch_size
        n = self.indexes.size(0)  # 训练集数目
        if self.mode=='train' and self.args.drop_partial_batch:
            print ('dropping partial batch')
            n = (n//bs) *bs
        elif self.mode=='train' and self.args.fill_partial_batch:
            print ('filling partial batch')
            remain = n % bs
            if remain > 0:
                extra = np.random.choice(self.indexes[:-remain], size=(bs-remain), replace=False)
                self.indexes = torch.cat([self.indexes, torch.tensor(extra)])
                n = self.indexes.size(0)
                assert n % bs == 0

        for a in range(0, n, bs):
            b = min(n, a + bs)
            batch_indexes = self.indexes[a:b]
            batch_qids = [self.qids[idx] for idx in batch_indexes]
            batch_HF_labels = self._to_device(self.HF_labels[batch_indexes], self.device1)
            batch_Diag_labels = self._to_device(self.Diag_labels[batch_indexes], self.device1)
            batch_diagnosis_codes = self._to_device(self.diagnosis_codes[batch_indexes], self.device1)
            batch_seq_time_step = self._to_device(self.seq_time_step[batch_indexes], self.device1)
            batch_seq_time_step2 = self._to_device(self.seq_time_step2[batch_indexes], self.device1)
            batch_mask_mult = self._to_device(self.mask_mult[batch_indexes], self.device1)
            batch_mask_final = self._to_device(self.mask_final[batch_indexes], self.device1)
            batch_mask_code = self._to_device(self.mask_code[batch_indexes], self.device1)
            batch_lengths = self._to_device(self.lengths[batch_indexes], self.device1)
            batch_tensors0 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors0]
            batch_tensors1 = [self._to_device(x[batch_indexes], self.device1) for x in self.tensors1]

            edge_index_all, edge_type_all = self.adj_data
            #edge_index_all: nested list of shape (n_samples, num_choice), where each entry is tensor[2, E]
            #edge_type_all:  nested list of shape (n_samples, num_choice), where each entry is tensor[E, ]
            edge_index = self._to_device([edge_index_all[i] for i in batch_indexes], self.device1)
            edge_type  = self._to_device([edge_type_all[i] for i in batch_indexes], self.device1)

            yield tuple([batch_qids, batch_HF_labels, batch_Diag_labels,
                         batch_diagnosis_codes, batch_seq_time_step, batch_mask_mult, batch_mask_final, batch_mask_code,batch_lengths, batch_seq_time_step,
                         *batch_tensors0,  *batch_tensors1, edge_index, edge_type])

    def _to_device(self, obj, device):
        if isinstance(obj, (tuple, list)):
            return [self._to_device(item, device) for item in obj]
        else:
            return obj.to(device)


def load_sparse_adj_data_with_contextnode(adj_pk_path, max_node_num, num_choice, args):
    cache_path = adj_pk_path +'.loaded_cache'
    use_cache = False

    if use_cache and not os.path.exists(cache_path):
        use_cache = False

    if use_cache:
        with open(cache_path, 'rb') as f:
            adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel = pickle.load(f)
    else:
        with open(adj_pk_path, 'rb') as fin:
            adj_concept_pairs = pickle.load(fin)

        n_samples = len(adj_concept_pairs) # this is actually n_questions x n_choices
        edge_index, edge_type = [], []
        adj_lengths = torch.zeros((n_samples,), dtype=torch.long)  # adj有多少个，用tensor张量表示
        concept_ids = torch.full((n_samples, max_node_num), 1, dtype=torch.long)
        node_type_ids = torch.full((n_samples, max_node_num), 2, dtype=torch.long)  # default 2: "other node"
        node_scores = torch.zeros((n_samples, max_node_num, 1), dtype=torch.float)

        adj_lengths_ori = adj_lengths.clone()
        # 每个问题匹配一个答案，作为一条adj
        for idx, _data in tqdm(enumerate(adj_concept_pairs), total=n_samples, desc='loading adj matrices'):
            adj, concepts, qm, am, cid2score = _data['adj'], _data['concepts'], _data['qmask'], _data['amask'], _data['cid2score']
            #adj: e.g. <4233x249 (n_nodes*half_n_rels x n_nodes) sparse matrix of type '<class 'numpy.bool'>' with 2905 stored elements in COOrdinate format>
            #concepts: np.array(num_nodes, ), where entry is concept id
            #qm: np.array(num_nodes, ), where entry is True/False
            #am: np.array(num_nodes, ), where entry is True/False
            assert len(concepts) == len(set(concepts))
            qam = qm | am
            #sanity check: should be T,..,T,F,F,..F
            assert qam[0] == True
            F_start = False
            for TF in qam:
                if TF == False:
                    F_start = True
                else:
                    assert F_start == False
            num_concept = min(len(concepts), max_node_num-1) + 1 #this is the final number of nodes including contextnode but excluding PAD
            adj_lengths_ori[idx] = len(concepts)
            adj_lengths[idx] = num_concept

            #Prepare nodes
            concepts = concepts[:num_concept-1]
            concept_ids[idx, 1:num_concept] = torch.tensor(concepts +1)  #To accomodate contextnode, original concept_ids incremented by 1
            concept_ids[idx, 0] = 0 #this is the "concept_id" for contextnode

            #Prepare node scores
            if (cid2score is not None):
                for _j_ in range(num_concept):
                    _cid = int(concept_ids[idx, _j_]) - 1
                    assert _cid in cid2score
                    node_scores[idx, _j_, 0] = torch.tensor(cid2score[_cid])

            #Prepare node types
            node_type_ids[idx, 0] = 3 #contextnode
            node_type_ids[idx, 1:num_concept][torch.tensor(qm, dtype=torch.bool)[:num_concept-1]] = 0
            node_type_ids[idx, 1:num_concept][torch.tensor(am, dtype=torch.bool)[:num_concept-1]] = 1

            #Load adj
            ij = torch.tensor(adj.row, dtype=torch.int64) #(num_matrix_entries, ), where each entry is coordinate
            k = torch.tensor(adj.col, dtype=torch.int64)  #(num_matrix_entries, ), where each entry is coordinate
            n_node = adj.shape[1]
            half_n_rel = adj.shape[0] // n_node
            i, j = ij // n_node, ij % n_node

            #Prepare edges
            i += 2; j += 1; k += 1  # **** increment coordinate by 1, rel_id by 2 ****
            extra_i, extra_j, extra_k = [], [], []
            for _coord, q_tf in enumerate(qm):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if q_tf:
                    extra_i.append(0) #rel from contextnode to question concept
                    extra_j.append(0) #contextnode coordinate
                    extra_k.append(_new_coord) #question concept coordinate
            for _coord, a_tf in enumerate(am):
                _new_coord = _coord + 1
                if _new_coord > num_concept:
                    break
                if a_tf:
                    extra_i.append(1) #rel from contextnode to answer concept
                    extra_j.append(0) #contextnode coordinate
                    extra_k.append(_new_coord) #answer concept coordinate

            half_n_rel += 2 #should be 19 now
            if len(extra_i) > 0:
                i = torch.cat([i, torch.tensor(extra_i)], dim=0)
                j = torch.cat([j, torch.tensor(extra_j)], dim=0)
                k = torch.cat([k, torch.tensor(extra_k)], dim=0)
            ########################

            mask = (j < max_node_num) & (k < max_node_num)
            i, j, k = i[mask], j[mask], k[mask]
            i, j, k = torch.cat((i, i + half_n_rel), 0), torch.cat((j, k), 0), torch.cat((k, j), 0)  # add inverse relations
            edge_index.append(torch.stack([j,k], dim=0)) #each entry is [2, E]
            edge_type.append(i) #each entry is [E, ]

        with open(cache_path, 'wb') as f:
            pickle.dump([adj_lengths_ori, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type, half_n_rel], f)


    ori_adj_mean  = adj_lengths_ori.float().mean().item()
    ori_adj_sigma = np.sqrt(((adj_lengths_ori.float() - ori_adj_mean)**2).mean().item())
    print('| ori_adj_len: mu {:.2f} sigma {:.2f} | adj_len: {:.2f} |'.format(ori_adj_mean, ori_adj_sigma, adj_lengths.float().mean().item()) +
        ' prune_rate： {:.2f} |'.format((adj_lengths_ori > adj_lengths).float().mean().item()) +
        ' qc_num: {:.2f} | ac_num: {:.2f} |'.format((node_type_ids == 0).float().sum(1).mean().item(),
                                                    (node_type_ids == 1).float().sum(1).mean().item()))

    edge_index = list(map(list, zip(*(iter(edge_index),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[2, E] #this operation corresponds to .view(n_questions, n_choices)
    edge_type = list(map(list, zip(*(iter(edge_type),) * num_choice))) #list of size (n_questions, n_choices), where each entry is tensor[E, ]

    concept_ids, node_type_ids, node_scores, adj_lengths = [x.view(-1, num_choice, *x.size()[1:]) for x in (concept_ids, node_type_ids, node_scores, adj_lengths)]
    #concept_ids: (n_questions, num_choice, max_node_num)
    #node_type_ids: (n_questions, num_choice, max_node_num)
    #node_scores: (n_questions, num_choice, max_node_num)
    #adj_lengths: (n_questions,　num_choice)
    return concept_ids, node_type_ids, node_scores, adj_lengths, (edge_index, edge_type) #, half_n_rel * 2 + 1





def load_gpt_input_tensors(statement_jsonl_path, max_seq_length):
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    def load_qa_dataset(dataset_path):
        """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
        with open(dataset_path, "r", encoding="utf-8") as fin:
            output = []
            for line in fin:
                input_json = json.loads(line)
                label = ord(input_json.get("answerKey", "A")) - ord("A")
                output.append((input_json['id'], input_json["question"]["stem"], *[ending["text"] for ending in input_json["question"]["choices"]], label))
        return output

    def pre_process_datasets(encoded_datasets, num_choices, max_seq_length, start_token, delimiter_token, clf_token):
        """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

            To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
            input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
        """
        tensor_datasets = []
        for dataset in encoded_datasets:
            n_batch = len(dataset)
            input_ids = np.zeros((n_batch, num_choices, max_seq_length), dtype=np.int64)
            mc_token_ids = np.zeros((n_batch, num_choices), dtype=np.int64)
            lm_labels = np.full((n_batch, num_choices, max_seq_length), fill_value=-1, dtype=np.int64)
            mc_labels = np.zeros((n_batch,), dtype=np.int64)
            for i, data, in enumerate(dataset):
                q, mc_label = data[0], data[-1]
                choices = data[1:-1]
                for j in range(len(choices)):
                    _truncate_seq_pair(q, choices[j], max_seq_length - 3)
                    qa = [start_token] + q + [delimiter_token] + choices[j] + [clf_token]
                    input_ids[i, j, :len(qa)] = qa
                    mc_token_ids[i, j] = len(qa) - 1
                    lm_labels[i, j, :len(qa) - 1] = qa[1:]
                mc_labels[i] = mc_label
            all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
            tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
        return tensor_datasets

    def tokenize_and_encode(tokenizer, obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        else:
            return list(tokenize_and_encode(tokenizer, o) for o in obj)

    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(GPT_SPECIAL_TOKENS)

    dataset = load_qa_dataset(statement_jsonl_path)
    examples_ids = [data[0] for data in dataset]
    dataset = [data[1:] for data in dataset]  # discard example ids
    num_choices = len(dataset[0]) - 2

    encoded_dataset = tokenize_and_encode(tokenizer, dataset)

    (input_ids, mc_token_ids, lm_labels, mc_labels), = pre_process_datasets([encoded_dataset], num_choices, max_seq_length, *special_tokens_ids)
    return examples_ids, mc_labels, input_ids, mc_token_ids, lm_labels


def get_gpt_token_num():
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    tokenizer.add_tokens(GPT_SPECIAL_TOKENS)
    return len(tokenizer)

# hitanet
def load_data_hita(statement_jsonl_path, model_type, model_name, max_seq_len):
    class InputExample(object):

        def __init__(self, example_id, question=None, HF_label=None, Diag_label=None):
            self.example_id = example_id
            self.question = question
            self.HF_label = HF_label
            self.Diag_label = Diag_label

    class InputFeatures(object):

        def __init__(self, example_id, choices_features, HF_label, Diag_label):
            self.example_id = example_id
            self.choices_features = [
                {
                    'input_ids': input_ids,
                    'input_mask': input_mask,
                    'segment_ids': segment_ids,
                    'output_mask': output_mask,
                }
                for _, input_ids, input_mask, segment_ids, output_mask in choices_features
            ]
            self.HF_label = HF_label
            self.Diag_label = Diag_label

    code2id = np.load('../data/icd2idx-mimic-iii.npy', allow_pickle=True).item()
    id2code = {}
    for code in code2id:
        key = code2id[code]
        value = code
        id2code[key] = value

    def read_examples(input_file, max_seq_len):
        '''
        The patient was diagnosed with XXX, XXX at the first diagnosis.
        After XXX days, the patient visited the doctor again and was diagnosed with XXX.
        After XXX days, the patient visited the doctor again and was diagnosed with XXX.
        '''
        diagnosis_codes = []
        example_ids = []
        n_diagnosis_codes = len(code2id)
        labels = []
        with open(input_file, "r", encoding="utf-8") as f:
            examples = []
            for line in f.readlines():
                json_dic = json.loads(line)
                example_id = json_dic["id"]
                record_icd = json_dic["medical_records"]["record_icd"]
                for i in range(len(record_icd)):
                    for j in range(len(record_icd[i])):
                        try:
                            record_icd[i][j] = code2id[record_icd[i][j]]
                        except:
                            record_icd[i][j] = 0

                diagnosis_codes.append(record_icd)
                example_ids.append(example_id)

                examples.append(
                    InputExample(
                        example_id = example_id,
                        question=None,
                        HF_label=None,
                        Diag_label=None
                    ))
            diagnosis_codes = units.adjust_input_pretrain(diagnosis_codes, max_seq_len, n_diagnosis_codes)
            lengths = np.array([max_seq_len + 1 for seq in diagnosis_codes])
            lengths = torch.from_numpy(lengths)
            diagnosis_codes, mask, mask_final, mask_code_, MLM_masked_batch, MLM_mask = units.pad_matrix_pretrain(diagnosis_codes, n_diagnosis_codes, max_seq_len + 1)
            diagnosis_codes = torch.LongTensor(diagnosis_codes)  # torch.Size([6000, 21, 39])
            mask_mult = torch.ByteTensor(1 - mask).unsqueeze(2)
            mask_final = torch.Tensor(mask_final).unsqueeze(2)
            mask_code = torch.Tensor(mask_code_).unsqueeze(3)
        return examples, labels, diagnosis_codes, mask_mult, mask_final, mask_code, lengths, MLM_masked_batch, MLM_mask

    def convert_examples_to_features(examples, HF_labels, Diag_labels, max_seq_length,
                                     tokenizer,
                                    cls_token_at_end=False,
                                    cls_token='[CLS]',
                                    cls_token_segment_id=1,
                                    sep_token='[SEP]',
                                    sequence_a_segment_id=0,
                                    sequence_b_segment_id=1,
                                    sep_token_extra=False,
                                    pad_token_segment_id=0,
                                    pad_on_left=False,
                                    pad_token=0,
                                    mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # 标签映射

        features = []
        for ex_index, example in enumerate(tqdm(examples)):
            choices_features = []
            # zip打包，每个context对应每个ending（比方说10个ending，3个context，则会有30个(context, ending)）
            context = example.question
            HF_label = HF_labels[ex_index]
            Diag_label = Diag_labels[ex_index]

            features.append(InputFeatures(example_id=example.example_id, choices_features=[],
                                          HF_label=HF_label, Diag_label=Diag_label))

        return features

    def _truncate_seq_pair(tokens_a, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        # 死循环
        while True:
            total_length = len(tokens_a)
            if total_length <= max_length:
                break
            else:
                tokens_a.pop()

    def select_field(features, field):
        return [[choice[field] for choice in feature.choices_features] for feature in features]

    def convert_features_to_tensors(features):
        # token的id，可解码出token吧（张量）
        all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
        # 输入mask，是为了更好地训练模型？
        all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
        # 对应token的标识，0表示token_a，即问题。1表示token_b即答案。还有cls和pad的，分别为1和0
        all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
        # 输出mask，最终根据这个，来作预测
        all_output_mask = torch.tensor(select_field(features, 'output_mask'), dtype=torch.bool)
        # 标签（实际答案，最后一次就诊记录）
        # HF_label = torch.tensor([f.HF_label for f in features], dtype=torch.long)
        # Diag_label = torch.tensor([f.Diag_label for f in features], dtype=torch.long)
        return 'HF_label', 'Diag_label', all_input_ids, all_input_mask, all_segment_ids, all_output_mask

    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(model_name) # 加载BertTokenizer
    examples, labels, diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2 = \
        read_examples(statement_jsonl_path, max_seq_len) # 读取statement数据，作为example
    features = convert_examples_to_features(examples, [example.HF_label for example in examples],
                                            [example.Diag_label for example in examples],
                                            max_seq_len, tokenizer,
                                            cls_token_at_end=bool(model_type in ['xlnet']),  # xlnet has a cls token at the end
                                            cls_token=tokenizer.cls_token,
                                            sep_token=tokenizer.sep_token,
                                            sep_token_extra=bool(model_type in ['roberta', 'albert']),
                                            cls_token_segment_id=2 if model_type in ['xlnet'] else 0,
                                            pad_on_left=bool(model_type in ['xlnet']),  # pad on the left for xlnet
                                            pad_token_segment_id=4 if model_type in ['xlnet'] else 0,
                                            sequence_b_segment_id=0 if model_type in ['roberta', 'albert'] else 1)
    example_ids = [f.example_id for f in features]
    # 将features转化为张量
    HF_label, Diag_label, *data_tensors = convert_features_to_tensors(features)
    return (example_ids, HF_label, Diag_label,
            diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths, seq_time_step2, # 改，用于hita使用
            *data_tensors)


# 数据加载入口
def load_input_tensors(input_jsonl_path, model_type, model_name, max_seq_length):
    # LSTM模型
    if model_type in ('lstm',):
        raise NotImplementedError
    # GPT模型，NLP领域（无监督预训练，有监督微调
    elif model_type in ('gpt',):
        return load_gpt_input_tensors(input_jsonl_path, max_seq_length)
    # bert（SapBERT属于这一类）
    # xlnet
    # roberta
    # albert
    elif model_type in ('bert', 'xlnet', 'roberta', 'albert'):
        return load_data_hita(input_jsonl_path, model_type, model_name, max_seq_length)


def load_info(statement_path: str):
    n = sum(1 for _ in open(statement_path, "r"))
    num_choice = None
    with open(statement_path, "r", encoding="utf-8") as fin:
        ids = []
        labels = []
        for line in fin:
            input_json = json.loads(line)
            labels.append(ord(input_json.get("answerKey", "A")) - ord("A"))
            ids.append(input_json['id'])
            if num_choice is None:
                num_choice = len(input_json["question"]["choices"])
        labels = torch.tensor(labels, dtype=torch.long)

    return ids, labels, num_choice


def load_statement_dict(statement_path):
    all_dict = {}
    with open(statement_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            instance_dict = json.loads(line)
            qid = instance_dict['id']
            all_dict[qid] = {
                'question': instance_dict['question']['stem'],
                'answers': [dic['text'] for dic in instance_dict['question']['choices']]
            }
    return all_dict


if __name__ == '__main__':
    load_data_hita('../data/mimic/statement/demo.statement.jsonl', 'bert', 'bert-base-uncased', 21)