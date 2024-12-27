from tqdm import tqdm
import torch, os
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset
import jieba.posseg as pseg
from module.StructureEncoder import StructureDataEncoder
import numpy as np

class VitalSigDataset:
    def __init__(self):
        self.digit = StructureDataEncoder()
        self.num_classes = {
            '到院方式': 4,  
            '性别': 3,
            '出生日期': 5,
            'T℃': 3,
            'P(次/分)': 3,
            'R(次/分)': 3,
            'BP(mmHg)': 5,
            'SpO2': 3
        }        
    def one_hot(self, y, num_classes=None):
        """Convert to one-hot encoding."""
        y_tensor = torch.tensor(y)
        if num_classes is None:
            num_classes = y_tensor.max() + 1
        return torch.nn.functional.one_hot(y_tensor, num_classes=num_classes).float()

    def Structure(self, data):
        ar = self.one_hot(data['到院方式'].apply(lambda x: self.digit.Arr_way(x)).values, self.num_classes['到院方式'])
        g  = self.one_hot(data['性别'].apply(lambda x: self.digit.Gender(x)).values, self.num_classes['性别'])
        a  = self.one_hot(data['出生日期'].apply(lambda x: self.digit.Age(x)).values, self.num_classes['出生日期'])
        t  = self.one_hot(data['T℃'].apply(lambda x: self.digit.Temperature(x)).values, self.num_classes['T℃'])
        p  = self.one_hot(data['P(次/分)'].apply(lambda x: self.digit.Pulse(x)).values, self.num_classes['P(次/分)'])
        r  = self.one_hot(data['R(次/分)'].apply(lambda x: self.digit.Respiration(x)).values, self.num_classes['R(次/分)'])
        bp = self.one_hot(data['BP(mmHg)'].apply(lambda x: self.digit.BloodPressure(x)).values, self.num_classes['BP(mmHg)'])
        s  = self.one_hot(data['SpO2'].apply(lambda x: self.digit.SpO2(x)).values, self.num_classes['SpO2'])
        return ar, g, a, t, p, r, bp, s

    def StructureNoOneHot(self, data):
        # 对每一列应用相应的处理函数，并保持每个变量为独立的数组
        ar = np.array(data['到院方式'].apply(lambda x: self.digit.Arr_way(x)).values)
        g  = np.array(data['性别'].apply(lambda x: self.digit.Gender(x)).values)
        a  = np.array(data['出生日期'].apply(lambda x: self.digit.Age(x)).values)
        t  = np.array(data['T℃'].apply(lambda x: self.digit.Temperature(x)).values)
        p  = np.array(data['P(次/分)'].apply(lambda x: self.digit.Pulse(x)).values)
        r  = np.array(data['R(次/分)'].apply(lambda x: self.digit.Respiration(x)).values)
        bp = np.array(data['BP(mmHg)'].apply(lambda x: self.digit.BloodPressure(x)).values)
        s  = np.array(data['SpO2'].apply(lambda x: self.digit.SpO2(x)).values)

        # 使用hstack将这些数组横向拼接
        result = np.hstack([ar[:, np.newaxis], g[:, np.newaxis], a[:, np.newaxis], 
                            t[:, np.newaxis], p[:, np.newaxis], r[:, np.newaxis], 
                            bp[:, np.newaxis], s[:, np.newaxis]])

        return result


    def SFD_encoder(self, vs):
        batch_size, _ = vs.shape
        indices = vs.nonzero(as_tuple=True)[1].view(batch_size, -1)
        num_indices = indices.shape[1]
        distance_matrix = torch.zeros((batch_size, num_indices, num_indices), dtype=torch.float32)
        for idx in range(batch_size):
            feature_indices = indices[idx].float().view(-1, 1)
            dist_matrix = torch.cdist(feature_indices, feature_indices, p=1)
            distance_matrix[idx] = dist_matrix
        tri_indices = torch.triu_indices(distance_matrix.size(1), distance_matrix.size(2), offset=1)
        return distance_matrix[:, tri_indices[0], tri_indices[1]]

#************************************    包含加权-频次
class ChiefCompDataset(Dataset):
    def __init__(self, args, dataset, dataset_name):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = args.cache_dir
        self.cache_file = os.path.join(
            self.cache_dir, 
            f'cached_{dataset_name}_wmd.pt' if args.SFD 
            else f'cached_{dataset_name}_onehot.pt'
        )
        
        if os.path.exists(self.cache_file):
            print(f"Loading cached data for {dataset_name}...")
            self.data = torch.load(self.cache_file)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
            self.model = BertModel.from_pretrained('bert-base-chinese')
            self.model.to(self.device)
            self.model.eval()
            print(f"Encoding data for {dataset_name}...")
            self.data = self.encode_and_cache_data(dataset)
            torch.save(self.data, self.cache_file)
            print(f"Data encoded and cached for {dataset_name}.")

    # def filter_nouns_verbs(self, text):
    #     words = pseg.cut(text)  # 对文本进行词性标注
    #     filtered_words = [word for word, flag in words if flag.startswith('n') or flag.startswith('v')]
    #     return ' '.join(filtered_words)

    def BertEncoder(self, data, max_len=77):
        with torch.no_grad():
            encoded = self.tokenizer(data, 
                                     padding=True, 
                                     truncation=True, 
                                     return_tensors="pt", 
                                     max_length=max_len)
            input_ids = encoded['input_ids'].to(self.device)
            att_mask = encoded['attention_mask'].to(self.device)
            outputs = self.model(input_ids=input_ids, attention_mask=att_mask)
            tokens = outputs.last_hidden_state
        return tokens

    def encode_and_cache_data(self, dataset, batch_size=64):
        all_data = []
        for i in tqdm(range(0, len(dataset), batch_size), desc="Batch encoding"):
            batch = dataset[i:i + batch_size]
            vitalsigns, chiefcpt, labels_sety, labels_dept, lb_dept_cn = zip(*batch) 

            # filtered_chiefcpt = [self.filter_nouns_verbs(text) for text in chiefcpt]
            # filtered_lb_dept_cn = [self.filter_nouns_verbs(text) for text in lb_dept_cn]
            cc_tokens = self.BertEncoder(chiefcpt, max_len=20)
            lb_tokens = self.BertEncoder(lb_dept_cn, max_len=8)
            for j in range(len(batch)):
                all_data.append({
                    'VS': vitalsigns[j],
                    'Level': labels_sety[j],
                    'CC_tokens': cc_tokens[j],  
                    'Dept_digit': labels_dept[j],
                    'Dept_tokens': lb_tokens[j]
                })
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]