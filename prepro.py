from tqdm import tqdm
import ujson as json
import numpy as np                                          
from adj_utils import sparse_mxs_to_torch_sparse_tensor     
import scipy.sparse as sp                                   

docred_rel2id = json.load(open('dataset/gda/rel2id.json', 'r'))
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []  # 初始化一个空列表 res，用于存储分割后的子列表。
    for i in range(0, len(l),
                   n):  # 使用 for 循环遍历列表 l，循环的步长为 n。这里，range(0, len(l), n) 生成一个从 0 开始，到 len(l) 结束（但不包括 len(l)），步长为 n 的序列。这个序列的每个元素 i 将作为子列表的起始索引。
        assert len(l[i:i + n]) == n  # 使用 assert 语句检查从索引 i 到 i + n 的切片 l[i:i + n] 的长度是否等于 n
        res += [l[i:i + n]]  # 将这个子列表添加到结果列表 res 中。
    return res

def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    train_triplesall=[]
    ent2idxall=[]
    title_all=[]
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]
            title_all.append(pmid)

            if pmid not in pmids:
                
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()

                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                i_s = 0     
                sent_pos = {}     
                for sent in sents:
                    sent_pos[i_s] = len(new_sents)    
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                    i_s += 1        
                sent_pos[i_s] = len(new_sents)     
                sents = new_sents
                link_node = []                              
                for l in range(len(sent_pos) - 2):          
                    link_node += [[l, l, l, l, l, l, 2]]          
                mention_pos = []
                link_pos = []                               
                for i in range(len(sent_pos) - 2):          
                    link_pos.append((sent_pos[i], sent_pos[i+2]))           
                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                        h_sid, t_sid = p[10], p[16]     
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                        t_sid, h_sid = p[10], p[16]     
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_sid = map(int, h_sid.split(':'))     
                    t_sid = map(int, t_sid.split(':'))      
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    h_sid = [idx for idx in h_sid]      
                    t_sid = [idx for idx in t_sid]      
                    h_h_lids, h_t_lids, t_h_lids, t_t_lids = [], [], [], []
                    if h_id not in ent2idx:
                        h_eid = [len(ent2idx)] * len(h_start)   
                        ent2idx[h_id] = len(ent2idx)
                        for id in h_sid:                                    
                            if id < 1:                                      
                                h_h_lid = -1                                
                                h_t_lid = id                                
                            elif id > len(link_pos) - 1:                   
                                h_h_lid = id - 1                            
                                h_t_lid = -1                               
                            else:                                          
                                h_h_lid = id - 1                           
                                h_t_lid = id                                
                            h_h_lids.append(h_h_lid)                        
                            h_t_lids.append(h_t_lid)                       
                        mention_pos.append(list(zip(h_start, h_end, h_eid, h_h_lids, h_t_lids, h_sid)))    

                    if t_id not in ent2idx:
                        t_eid = [len(ent2idx)] * len(t_start)       
                        ent2idx[t_id] = len(ent2idx)
                        for id in t_sid:                                    
                            if id < 1:                                      
                                t_h_lid = -1                                
                                t_t_lid = id                                
                            elif id > len(link_pos) - 1:                    
                                t_h_lid = id - 1                            
                                t_t_lid = -1                               
                            else:                                           
                                t_h_lid = id - 1                            
                                t_t_lid = id                                
                            t_h_lids.append(t_h_lid)                        
                            t_t_lids.append(t_t_lid)                        
                        mention_pos.append(list(zip(t_start, t_end, t_eid, t_h_lids, t_t_lids, t_sid)))    

                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    dist = p[2]         
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r, 'dist': dist}]               
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r, 'dist': dist})           
                train_triplesall.append(train_triples)
                ent2idxall.append(ent2idx)
                    

                entity_node = []        
                mention_node = []       
                for idx in range(len(mention_pos)):     
                    entity_node += [[idx, idx, idx, idx, idx, idx, 0]]          
                    for item in mention_pos[idx]:           
                        mention_node += [list(item)+[1]]        

                nodes = entity_node + mention_node + link_node      
                nodes = np.array(nodes)         

                xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')       
                l_type, r_type = nodes[xv, 6], nodes[yv, 6]         
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]           
                l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]       
                l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]       
                l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]           


                adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
                adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
                adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
                adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
                adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
                adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
                adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])      
                adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1])
                adj_temp = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adj_temp = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adjacency[2] = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
                adjacency[2] = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adjacency[2])      
                adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
                for x, y in zip(xv.ravel(), yv.ravel()):                                                       
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:                                                   
                        z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))   
                        temp_ = np.where((l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)                
                        temp_ = np.where((l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, temp_)                   
                        adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0                                  
                        adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0                                  

                adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])      

                relations, hts, dists = [], [], []           
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["dist"] == "CROSS":
                            dist = 1                            
                        elif mention["dist"] == "NON-CROSS":
                            dist = 0                            
                    relations.append(relation)
                    hts.append([h, t])
                    dists.append(dist)                  

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': mention_pos,
                           'labels': relations,
                           'dists': dists,              
                           'hts': hts,
                           'title': pmid,
                           'adjacency': adjacency,      
                           'link_pos': link_pos,       
                           'nodes_info': nodes,         
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features,train_triplesall,ent2idxall,title_all


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]
            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)
                ent2idx = {}
                train_triples = {}
                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))
                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))
                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                i_s = 0                    
                sent_pos = {}               
                for sent in sents:
                    sent_pos[i_s] = len(new_sents)    
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                    i_s += 1       
                sent_pos[i_s] = len(new_sents)     
                sents = new_sents

                if len(sents) > max_seq_length:         
                    pmids.remove(pmid)                  
                    continue                           
                link_node = []  
                link_pos = []
                if len(sent_pos) > 2:
                    for l in range(len(sent_pos) - 2):  
                        link_node += [[l, l, l, l, l, l, 2]]  
                        link_pos.append((sent_pos[l], sent_pos[l + 2]))  
                else:
                    link_node = [[0, 0, 0, 0, 0, 0, 2]]             
                    link_pos.append((sent_pos[0], sent_pos[1]))     

                mention_pos = []
                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                        h_sid, t_sid = p[10], p[16]     
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                        t_sid, h_sid = p[10], p[16]     
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_sid = map(int, h_sid.split(':'))  
                    t_sid = map(int, t_sid.split(':'))  
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    h_sid = [idx for idx in h_sid]  
                    t_sid = [idx for idx in t_sid]  
                    h_h_lids, h_t_lids, t_h_lids, t_t_lids = [], [], [], []  
                    if h_id not in ent2idx:
                        h_eid = [len(ent2idx)] * len(h_start)    
                        ent2idx[h_id] = len(ent2idx)
                        for id in h_sid:                                   
                            if id < 1:                                      
                                h_h_lid = -1                               
                                h_t_lid = id                               
                            elif id > len(link_pos) - 1:                  
                                h_h_lid = id - 1                           
                                h_t_lid = -1                                
                            else:                                          
                                h_h_lid = id - 1                          
                                h_t_lid = id                                
                            h_h_lids.append(h_h_lid)                        
                            h_t_lids.append(h_t_lid)                        
                        mention_pos.append(list(zip(h_start, h_end, h_eid, h_h_lids, h_t_lids, h_sid)))  
                    if t_id not in ent2idx:
                        t_eid = [len(ent2idx)] * len(t_start)       
                        ent2idx[t_id] = len(ent2idx)
                        for id in t_sid:                                    
                            if id < 1:                                      
                                t_h_lid = -1                                
                                t_t_lid = id                                
                            elif id > len(link_pos) - 1:                    
                                t_h_lid = id - 1                            
                                t_t_lid = -1                                
                            else:                                           
                                t_h_lid = id - 1                            
                                t_t_lid = id                               
                            t_h_lids.append(t_h_lid)                        
                            t_t_lids.append(t_t_lid)                        
                        mention_pos.append(list(zip(t_start, t_end, t_eid, t_h_lids, t_t_lids, t_sid)))  

                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    dist = p[2]             
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r, "dist": dist}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r, "dist": dist})

                entity_node = []  
                mention_node = []  
                for idx in range(len(mention_pos)):  
                    entity_node += [[idx, idx, idx, idx, idx, idx, 0]]  
                    for item in mention_pos[idx]:  
                        mention_node += [list(item) + [1]]  

                nodes = entity_node + mention_node + link_node  
                nodes = np.array(nodes)  
                xv, yv = np.meshgrid(np.arange(nodes.shape[0]), np.arange(nodes.shape[0]), indexing='ij')  
                l_type, r_type = nodes[xv, 6], nodes[yv, 6]  
                l_eid, r_eid = nodes[xv, 2], nodes[yv, 2]  
                l_h_lid, r_h_lid = nodes[xv, 3], nodes[yv, 3]  
                l_t_lid, r_t_lid = nodes[xv, 4], nodes[yv, 4]  
                l_sid, r_sid = nodes[xv, 5], nodes[yv, 5]  

                adj_temp = np.full((l_type.shape[0], r_type.shape[0]), 0, 'i')
                adjacency = np.full((5, l_type.shape[0], r_type.shape[0]), 0.0)
                adj_temp = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adj_temp)
                adjacency[0] = np.where((l_type == 1) & (r_type == 1) & (l_sid == r_sid), 1, adjacency[0])
                adj_temp = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adj_temp)
                adj_temp = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adj_temp)
                adjacency[1] = np.where((l_type == 0) & (r_type == 1) & (l_eid == r_eid), 1, adjacency[1])  
                adjacency[1] = np.where((l_type == 1) & (r_type == 0) & (l_eid == r_eid), 1, adjacency[1]) 
                adj_temp = np.where(
                    (l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adj_temp = np.where(
                    (l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1, adj_temp)
                adjacency[2] = np.where(
                    (l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                    adjacency[2])  
                adjacency[2] = np.where(
                    (l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                    adjacency[2])  
                adj_temp = np.where((l_type == 2) & (r_type == 2), 1, adj_temp)
                adjacency[3] = np.where((l_type == 2) & (r_type == 2), 1, adjacency[3])
                for x, y in zip(xv.ravel(), yv.ravel()):  
                    if nodes[x, 5] == 0 and nodes[y, 5] == 2:  
                        z = np.where((l_eid == nodes[x, 2]) & (l_type == 1) & (r_type == 2) & np.logical_or(
                            r_h_lid == nodes[y, 3], r_t_lid == nodes[y, 4]))  
                        temp_ = np.where(
                            (l_type == 1) & (r_type == 2) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            adj_temp)  
                        temp_ = np.where(
                            (l_type == 2) & (r_type == 1) & np.logical_or(l_h_lid == r_h_lid, l_t_lid == r_t_lid), 1,
                            temp_)  
                        adjacency[4][x, y] = 1 if (temp_[z] == 1).any() else 0  
                        adjacency[4][y, x] = 1 if (temp_[z] == 1).any() else 0  

                adjacency = sparse_mxs_to_torch_sparse_tensor([sp.coo_matrix(adjacency[i]) for i in range(5)])  

                relations, hts, dists = [], [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                        if mention["dist"] == "CROSS":
                            dist = 1                           
                        elif mention["dist"] == "NON-CROSS":
                            dist = 0                           
                    relations.append(relation)
                    hts.append([h, t])
                    dists.append(dist)                          

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)    

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': mention_pos,
                           'labels': relations,
                           'dists': dists,          
                           'hts': hts,
                           'title': pmid,
                           'adjacency': adjacency,  
                           'link_pos': link_pos, 
                           'nodes_info': nodes,  
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
