import json
import random
import os
#type: concat, rerank, cred, rerank_cred
random.seed(0)
class BaseDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = self.load_data()
        self.dataset_name = os.path.basename(dataset_path)
        #self.eval_data = self.convert_data()

    def load_data(self):
        with open(self.dataset_path, "r") as f:
            data = json.load(f)
        return data
    
    def load_eval_data(self, type_):
        dataset_path = os.path.join(self.dataset_path, self.dataset_name)
        for file in os.listdir(dataset_path):
            if type_ in file:
                with open(os.path.join(dataset_path, file), "r") as f:
                    eval_data = json.load(f)
        return eval_data

    def convert_data(self, type_, q_last, **kwargs):
        self.type_ = type_
        noise_ratio = kwargs.get('noise_ratio', None)
        total_num = kwargs.get('total_num', None)
        if noise_ratio and total_num:
            for i in range(len(self.data)):
                docs = self.data[i]['docs']
                pos_num = int(total_num * (1 - noise_ratio)+0.1)
                neg_num = total_num - pos_num
                
                pos = [doc for doc in docs if doc['cred'] == 'high']
                neg = [doc for doc in docs if doc not in pos]
                
                self.data[i]['docs'] = neg[:neg_num] + pos[:pos_num]

        eval_data = []
        for item in self.data:
            question = item['question']
            docs = item['docs']
            answer = item['answer']
            doc_prompt = ""
            if self.type_ == 'concat':
                random.shuffle(docs)
                for doc in docs:
                    if doc['title'] != "":
                        title = f"{doc['title']}:"
                    else:
                        title = ""
                    doc_prompt += f"{title}{doc['text']}\n"
            elif self.type_ == 'rerank' and not q_last:
                for i in range(len(docs)-1, -1, -1):
                    if docs[i]['title'] != "":
                        title = f"{docs[i]['title']}:"
                    else:
                        title = ""
                    doc_prompt += f"{title}{docs[i]['text']}\n"
            elif self.type_ == 'rerank' and q_last:
                for doc in docs:
                    if doc['title'] != "":
                        title = f"{doc['title']}:"
                    else:
                        title = ""
                    doc_prompt += f"{title}{doc['text']}\n"
            elif self.type_ == 'cred':
                random.shuffle(docs)
                for doc in docs:
                    cred = doc["cred"]
                    cred_ = cred[0].upper() + cred[1:].lower()
                    if doc['title'] != "":
                        title = f"{doc['title']}:"
                    else:
                        title = ""
                    doc_prompt += f"{cred_} credibility of text: {title}{doc['text']}\n"
            elif self.type_ == 'rerank_cred':
                for i in range(len(docs)-1, -1, -1):
                    cred = docs[i]["cred"]
                    cred_ = cred[0].upper() + cred[1:].lower()
                    if docs[i]['title'] != "":
                        title = f"{docs[i]['title']}:"
                    else:
                        title = ""
                    doc_prompt += f"{cred_} credibility of text: {title}{docs[i]['text']}\n"
            if q_last:
                prompt = f"Docs:{doc_prompt}\nQuestion:{question}\nAnswer:"
            else:
                prompt = f"Question:{question}\nDocs:{doc_prompt}\nAnswer:"
            eval_data.append({"conversations":[
                {
                    "from": "human",
                    "value": prompt
                },
                {
                    "from": "gpt",
                    "value": answer 
                }
            ] })
        return eval_data

    def save_eval_data(self, eval_data, **kwargs):
        noise_ratio = kwargs.get('noise_ratio', None)
        dir_name = os.path.dirname(self.dataset_path)
        file_name = os.path.basename(self.dataset_path)
        file_stem = os.path.splitext(file_name)[0]
        if noise_ratio:
            output_path = f"{file_stem}_{self.type_}_noise_ratio{noise_ratio}.json"
        else:
            #_qstart
            output_path = f"{file_stem}_{self.type_}.json"
        with open(os.path.join(dir_name, output_path), "w") as f:
            json.dump(eval_data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    dataset_path = ""
    
    type_ = "cred"
    dataset = BaseDataset(dataset_path)
    eval_data = dataset.convert_data(type_, False)
    dataset.save_eval_data(eval_data)
    