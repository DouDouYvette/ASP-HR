import torch
import torch.nn as nn
from long_seq import process_long_input
from losses import ATLoss
from torch.nn.utils.rnn import pad_sequence
from MRGARN import DynaRelGAT, MRGARN
from utils import EmbedLayer
from residual import BiLstm_reason
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class DocREModel(nn.Module):
    def __init__(self, args, config, model, emb_size=1536, num_labels=-1, max_entity=35):
        super().__init__()
        self.config = config
        self.model = model
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss(args.num_class)
        self.extractor_trans = nn.Linear(config.hidden_size, emb_size)
        self.ht_extractor = nn.Linear(emb_size * 4, emb_size * 2)
        self.MIP_Linear = nn.Linear(emb_size * 5, emb_size * 4)
        self.MIP_Linear2 = nn.Linear(emb_size * 4, emb_size * 2)
        self.bilinear = nn.Linear(emb_size * 2, config.num_labels)
        self.emb_size = emb_size
        self.num_labels = num_labels
        self.max_entity = max_entity
        self.type_dim = 20
        self.drop = nn.Dropout(0.2)
        self.rgat =MRGARN(emb_size + self.type_dim, emb_size, emb_size, args.gat_headnum, args.gat_layers, 5,
                              args.gat_drop)
        self.type_embed = EmbedLayer(num_embeddings=3, embedding_dim=self.type_dim, dropout=0.2)
        self.sigmoid = nn.Sigmoid()
        self.fusion = BiLstm_reason(1536, 768)
        self.relu = nn.ReLU(inplace=True)
        self.lstm_reason_e = BiLstm_reason(1536, 768)
        self.W_q = nn.Parameter(torch.randn(emb_size, emb_size))
        self.W_k = nn.Parameter(torch.randn(emb_size, emb_size))
        self.W_dis_q = nn.Parameter(torch.randn(emb_size))
        self.W_dis_k = nn.Parameter(torch.randn(emb_size))
        self.bias_dis = nn.Parameter(torch.zeros(1))
        self.W_h = nn.Parameter(torch.randn(1))
        self.W_c = nn.Parameter(torch.randn(1))
        self.scale_factor = nn.Parameter(torch.tensor(0.0))
        self.semantic_weight = nn.Parameter(torch.tensor(0.3))
        self.distance_weight = nn.Parameter(torch.tensor(0.5))
        self.gate_linear = nn.Linear(2, 1)

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def make_graph(self, sequence_output, attention, entity_pos, link_pos, nodes_info):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        nodes_batch = []
        entity_att_batch = []
        entity_node_batch = []
        mention_pos_batch = []
        mention_att_batch = []
        for i in range(len(entity_pos)):
            entity_nodes, mention_nodes, link_nodes = [], [], []
            entity_att = []
            mention_att = []
            mention_pos = []
            for start, end in link_pos[i]:
                if end + offset < c:
                    link_rep = sequence_output[i, start + offset: end + offset]
                    link_att = attention[i, :, start + offset: end + offset, start + offset: end + offset]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                elif start + offset < c:
                    link_rep = sequence_output[i, start + offset:]
                    link_att = attention[i, :, start + offset:, start + offset:]
                    link_att = torch.mean(link_att, dim=0)
                    link_rep = torch.mean(torch.matmul(link_att, link_rep), dim=0)
                else:
                    link_rep = torch.zeros(self.config.hidden_size).to(sequence_output)
                link_nodes.append(link_rep)
            for e in entity_pos[i]:
                mention_pos.append(len(mention_att))
                if len(e) > 1:
                    m_emb, e_att = [], []
                    for start, end, e_id, h_lid, t_lid, sid in e:
                        if start + offset < c:
                            mention_nodes.append(sequence_output[i, start + offset])
                            m_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                            mention_att.append(attention[i, :, start + offset])
                        else:
                            mention_nodes.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            m_emb.append(torch.zeros(self.config.hidden_size).to(sequence_output))
                            e_att.append(torch.zeros(h, c).to(attention))
                            mention_att.append(torch.zeros(h, c).to(attention))
                    if len(m_emb) > 0:
                        m_emb = torch.logsumexp(torch.stack(m_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end, e_id, h_lid, t_lid, sid = e[0]
                    if start + offset < c:
                        mention_nodes.append(sequence_output[i, start + offset])
                        m_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                        mention_att.append(attention[i, :, start + offset])
                    else:
                        m_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                        mention_att.append(torch.zeros(h, c).to(attention))
                entity_nodes.append(m_emb)
                entity_att.append(e_att)
            mention_pos.append(len(mention_att))
            entity_att = torch.stack(entity_att, dim=0)
            entity_att_batch.append(entity_att)
            entity_nodes = torch.stack(entity_nodes, dim=0)
            mention_nodes = torch.stack(mention_nodes, dim=0)
            mention_att = torch.stack(mention_att, dim=0)
            link_nodes = torch.stack(link_nodes, dim=0)
            nodes = torch.cat([entity_nodes, mention_nodes, link_nodes], dim=0)
            nodes_type = self.type_embed(nodes_info[i][:, 6].to(sequence_output.device))
            nodes = torch.cat([nodes, nodes_type], dim=1)
            nodes_batch.append(nodes)
            entity_node_batch.append(entity_nodes)
            mention_att_batch.append(mention_att)
            mention_pos_batch.append(mention_pos)
        nodes_batch = pad_sequence(nodes_batch, batch_first=True, padding_value=0.0)
        return nodes_batch, entity_att_batch, entity_node_batch, mention_att_batch, mention_pos_batch

    def relation_map(self, gcn_nodes, entity, entity_att, entity_pos, sequence_output, mention_att):
        entity_s, mention_s = [], []
        entity_c, mention_c = [], []
        nodes = gcn_nodes[-1]
        m_num_max = 0
        e_num_max = 0
        for i in range(len(entity_pos)):
            m_num, _, _ = mention_att[i].size()
            m_num_max = m_num if m_num > m_num_max else m_num_max
            e_num = len(entity_pos[i])
            e_num_max = e_num if e_num > e_num_max else e_num_max
        for i in range(len(entity_pos)):
            e_num = len(entity_pos[i])
            entity_stru = nodes[i][: e_num]
            m_num, head_num, dim = mention_att[i].size()
            mention_stru = nodes[i][e_num: e_num + m_num]
            e_att = entity_att[i].mean(1)
            e_att = e_att / (e_att.sum(1, keepdim=True) + 1e-5)
            e_context = torch.einsum('ij, jl->il', e_att, sequence_output[i])
            m_att = mention_att[i].mean(1)
            m_att = m_att / (m_att.sum(1, keepdim=True) + 1e-5)
            m_context = torch.einsum('ij,jl->il', m_att, sequence_output[i])
            n, h = entity_stru.size()
            e_s = torch.zeros([e_num_max, h]).to(sequence_output)
            e_s[:n] = entity_stru
            entity_s.append(e_s)
            m, h = mention_stru.size()
            m_s = torch.zeros([m_num_max, h]).to(sequence_output)
            m_s[:m] = mention_stru
            mention_s.append(m_s)
            n, h_2 = e_context.size()
            e_c = torch.zeros([e_num_max, h_2]).to(sequence_output)
            e_c[:n] = e_context
            entity_c.append(e_c)
            m, h = m_context.size()
            m_c = torch.zeros([m_num_max, h]).to(sequence_output)
            m_c[:m] = m_context
            mention_c.append(m_c)
        entity_c = torch.stack(entity_c, dim=0)
        entity_s = torch.stack(entity_s, dim=0)
        mention_c = torch.stack(mention_c, dim=0)
        mention_s = torch.stack(mention_s, dim=0)
        return entity_c, entity_s, mention_c, mention_s

    def Enhance_entity(self, feat_struc, feat_context):
        feat_mix = feat_struc + feat_context
        feat_s = self.fusion(feat_mix)
        feat_c = self.fusion(feat_mix)
        feat_sc = feat_s + feat_c
        w_feat_sc = self.sigmoid(feat_sc)
        feat_fusion = feat_struc * (1 - w_feat_sc) + feat_context * w_feat_sc
        return feat_fusion

    def Enhance_mention(self, feat_struc, feat_context):
        feat_mix = feat_struc + feat_context
        feat_s = self.fusion(feat_mix)
        feat_c = self.fusion(feat_mix)
        feat_sc = feat_s + feat_c
        w_feat_sc = self.sigmoid(feat_sc)
        feat_fusion = feat_struc * (1 - w_feat_sc) + feat_context * w_feat_sc
        return feat_fusion

    def Enhance_entity_pair(self, entity_pair):
        E_entity_pair = self.lstm_reason_e(entity_pair)
        E_entity_pair = self.relu(E_entity_pair)
        E_entity_pair = self.lstm_reason_e(E_entity_pair)
        E_entity_pair = self.relu(E_entity_pair)
        E_entity_pair = self.lstm_reason_e(E_entity_pair)
        E_entity_pair = self.relu(E_entity_pair)
        E_entity_pair = E_entity_pair + entity_pair
        E_entity_pair = torch.einsum('ijk,ilk->ikjl', E_entity_pair, E_entity_pair)
        return E_entity_pair

    def Enhance_mention_pair(self, mention_pair):
        E_mention_pair = self.lstm_reason_e(mention_pair)
        E_mention_pair = self.relu(E_mention_pair)
        E_mention_pair = self.lstm_reason_e(E_mention_pair)
        E_mention_pair = self.relu(E_mention_pair)
        E_mention_pair = self.lstm_reason_e(E_mention_pair)
        E_mention_pair = self.relu(E_mention_pair)
        E_mention_pair = E_mention_pair + mention_pair
        E_mention_pair = torch.einsum('ijk,ilk->ikjl', E_mention_pair, E_mention_pair)
        return E_mention_pair

    def extract_entity_features(self, nodes):
        entity_features = nodes[:, :self.max_entity, :]
        return entity_features

    def get_mention_distance(self, mention1_sent_id, mention2_sent_id, mention1_emb, mention2_emb, mention1_pos,
                             mention2_pos):
        sent_distance = abs(mention1_sent_id - mention2_sent_id)
        semantic_distance = 1 - F.cosine_similarity(mention1_emb.unsqueeze(0), mention2_emb.unsqueeze(0)).item()
        combined_distance = (1 - self.semantic_weight) * sent_distance + self.semantic_weight * semantic_distance
        return combined_distance

    def get_context_feature(self, entity1_pos, entity2_pos, sequence_output):
        entity1_mentions = []
        for j in range(len(sequence_output)):
            for item in entity1_pos:
                if len(item) >= 6 and j == item[0]:  # 检查长度并确保索引匹配
                    entity1_mentions.append(sequence_output[j])
                    break
        entity2_mentions = []
        for k in range(len(sequence_output)):
            for item in entity2_pos:
                if len(item) >= 6 and k == item[0]:
                    entity2_mentions.append(sequence_output[k])
                    break

        if len(entity1_mentions) == 0:
            entity1_mentions = [torch.zeros(sequence_output.size(-1)).to(sequence_output.device)]
        if len(entity2_mentions) == 0:
            entity2_mentions = [torch.zeros(sequence_output.size(-1)).to(sequence_output.device)]

        entity1_mentions = torch.stack(entity1_mentions)
        entity2_mentions = torch.stack(entity2_mentions)
        context_feature = torch.mean(torch.cat([entity1_mentions, entity2_mentions], dim=0), dim=0)
        return context_feature

    def secondary_reasoning(self, initial_relations, entity_features, adjacency):
        adjacency = adjacency.to_dense()
        num_entities = entity_features.size(1)
        entity_adjacency = adjacency[:, -1, :num_entities, :num_entities]
        secondary_relations = torch.einsum('bni,bnj->bnij', entity_features, entity_adjacency)
        alpha = torch.sigmoid(self.scale_factor)
        combined_relations = initial_relations + alpha * secondary_relations
        return combined_relations

    def secondary_reasoning_loss(self, initial_relations, entity_features, adjacency, labels=None):
        secondary_relations = self.secondary_reasoning(initial_relations, entity_features, adjacency)
        loss = self.loss_fnt(secondary_relations, labels)
        return loss
    

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                adjacency=None,
                link_pos=None,
                nodes_info=None,
                ):
        sequence_output, attention = self.encode(input_ids, attention_mask)
        sequence_output = self.extractor_trans(sequence_output)
        nodes, entity_att, entity_node_batch, mention_att, mentions_pos = self.make_graph(sequence_output, attention,
                                                                                          entity_pos, link_pos,
                                                                                          nodes_info)
        num_entities = len(entity_pos)
        mdeia_entity_features = []
        for i in range(num_entities):
            entity_i_mentions = []
            for j in range(len(sequence_output)):
                for item in entity_pos[i]:
                    if len(item) >= 6:
                        start = item[0]
                        if j == start:
                            entity_i_mentions.append(sequence_output[j])
                            break
            if len(entity_i_mentions) > 0:
                entity_i_mentions = torch.stack(entity_i_mentions)
            else:
                entity_i_mentions = torch.zeros(0, sequence_output.size(-1)).to(sequence_output.device)
            entity_i_features = []
            for j in range(num_entities):
                if i != j:
                    entity_j_mentions = []
                    for k in range(len(sequence_output)):
                        for item in entity_pos[j]:
                            if len(item) >= 6:
                                start = item[0]
                                if k == start:
                                    entity_j_mentions.append(sequence_output[k])
                                    break
                    if len(entity_j_mentions) > 0:
                        entity_j_mentions = torch.stack(entity_j_mentions)
                    else:
                        entity_j_mentions = torch.zeros(0, sequence_output.size(-1)).to(sequence_output.device)
                    att_scores = []
                    for p in range(entity_i_mentions.size(0)):
                        scores = []
                        for l in range(entity_j_mentions.size(0)):
                            dis = self.get_mention_distance(entity_pos[i][p][5], entity_pos[j][l][5],
                                                            entity_i_mentions[p], entity_j_mentions[l])
                            db = torch.matmul(entity_i_mentions[p], self.W_dis_q) + torch.matmul(entity_j_mentions[l],
                                                                                                 self.W_dis_k) + self.bias_dis
                            weighted_dis = self.distance_weight * dis
                            score = torch.matmul(torch.matmul(entity_i_mentions[p], self.W_q),
                                                 torch.matmul(entity_j_mentions[l], self.W_k).t()) + db+weighted_dis
                            scores.append(score)
                            att_score = F.softmax(scores, dim=0)
                            att_scores.append(att_score)
                        att_scores = torch.stack(att_scores)
                        att_scores = torch.zeros(0, entity_j_mentions.size(0)).to(sequence_output.device)

                    entity_j_weighted_features = torch.matmul(att_scores.t(), entity_j_mentions)
                    entity_i_feature = torch.tanh(
                        self.W_h * entity_j_weighted_features + self.W_c * self.get_context_feature(entity_pos[i],
                                                                                                    entity_pos[j],
                                                                        sequence_output))
                    entity_i_features.append(entity_i_feature)
                mdeia_entity_features.append(entity_i_features)
                nodes = torch.cat([nodes, mdeia_entity_features], dim=-1)
        gcn_nodes = self.rgat(nodes, adjacency)
        entity_c, entity_s, mention_c, mention_s = self.relation_map(gcn_nodes, entity_node_batch, entity_att,
                                                                     entity_pos, sequence_output, mention_att)
        entity_pair = self.Enhance_entity(entity_s, entity_c)
        mention_pair = self.Enhance_mention(mention_s, mention_c)
        E_entity_pair = self.Enhance_entity_pair(entity_pair)
        E_mention_pair = self.Enhance_mention_pair(mention_pair)
        entity_pair = torch.einsum('ijk,ilk->ikjl', entity_pair, entity_pair)
        mention_pair = torch.einsum('ijk,ilk->ikjl', mention_pair, mention_pair)
        relation = []
        entity_h = []
        entity_t = []
        sc_feature_e = []
        nodes_re = torch.cat([gcn_nodes[0], gcn_nodes[-1]], dim=-1)
        for i in range(len(entity_pos)):
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            r_v1 = E_entity_pair[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)
            r_v2 = []
            for j in range(ht_i.shape[0]):
                h_e_pos = ht_i[j, 0]
                t_e_pos = ht_i[j, 1]
                e_m_feat = E_mention_pair[i, :, mentions_pos[i][h_e_pos]:mentions_pos[i][h_e_pos + 1],
                           mentions_pos[i][t_e_pos]:mentions_pos[i][t_e_pos + 1]]
                e_feat = torch.mean(e_m_feat, dim=[1, 2]).reshape([1, -1])
                r_v2.append(e_feat)
            r_v2 = torch.cat(r_v2, dim=0)
            relation.append(torch.cat([r_v1, r_v2], dim=-1))
            f_e = entity_pair[i, :, ht_i[:, 0], ht_i[:, 1]].transpose(1, 0)
            sc_feature_e.append(f_e)
            e_h = torch.index_select(nodes_re[i], 0, ht_i[:, 0])
            e_t = torch.index_select(nodes_re[i], 0, ht_i[:, 1])
            entity_h.append(e_h)
            entity_t.append(e_t)
        relation = torch.cat(relation, dim=0)
        sc_feature_e = torch.cat(sc_feature_e, dim=0)
        entity_h = torch.cat(entity_h, dim=0)
        entity_t = torch.cat(entity_t, dim=0)
        entity_ht = self.ht_extractor(torch.cat([entity_h, entity_t], dim=-1))
        relation_rep = self.MIP_Linear(torch.cat([relation, sc_feature_e, entity_ht], dim=-1))
        relation_rep = torch.tanh(self.MIP_Linear2(relation_rep))
        logits = self.bilinear(relation_rep)
        initial_relations = self.loss_fnt.get_label(logits, num_labels=self.num_labels)
        entity_features = self.extract_entity_features(nodes)
        secondary_relations = self.secondary_reasoning(initial_relations, entity_features, adjacency)
        final_relations = initial_relations + secondary_relations
        output = (final_relations,)
        if labels is not None:
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            loss = self.loss_fnt(logits.float(), labels.float())
            output = (loss.to(sequence_output),) + output
        return output


