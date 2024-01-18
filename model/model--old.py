from pathlib import Path

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel, AutoConfig
from underthesea import word_tokenize

__all__ = ["ReadabilityClassifier"]

# Model weight: https://drive.google.com/file/d/1snF7EDcaRAXeJLnLJ_CrQkhDDELdiz-P/view?usp=sharing

# Utils for moving and detaching tensors to/from GPU

def move_to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {k: move_to(v, device) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [move_to(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to(list(obj), device))
    else:
        print(obj)
        raise TypeError("Invalid type for move_to")


def detach(obj):
    if torch.is_tensor(obj):
        return obj.detach()
    elif isinstance(obj, dict):
        res = {k: detach(v) for k, v in obj.items()}
        return res
    elif isinstance(obj, list):
        return [detach(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(detach(list(obj)))
    else:
        raise TypeError("Invalid type for detach")
    

class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights = None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = layer_weights if layer_weights is not None \
            else nn.Parameter(
                torch.tensor([1] * (num_hidden_layers+1 - layer_start), dtype=torch.float)
            )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start:, :, :, :]
        weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
        weighted_average = (weight_factor*all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
        return weighted_average
    
class CustomPhoBert(nn.Module):
    def __init__(self, extractor_name):
        super(CustomPhoBert, self).__init__()

        config = AutoConfig.from_pretrained(extractor_name)
        config.update({'output_hidden_states':True})

        self.extractor = AutoModel.from_pretrained(extractor_name , config=config)

        self.layer_start = 9
        self.pooler = WeightedLayerPooling(
            config.num_hidden_layers,
            layer_start=self.layer_start, layer_weights=None
        ) # batch , hidden_size

        self.drop = nn.Dropout (0.1)
        self.fc = nn.Linear (config.hidden_size, 4 )
        
    def forward(self, **batch):
        outputs = self.extractor(**batch)
        all_hidden_states = torch.stack(outputs[2])
        weighted_pooling_embeddings = self.pooler(all_hidden_states)
        weighted_pooling_embeddings = weighted_pooling_embeddings[:, 0]

        logits = self.fc(weighted_pooling_embeddings)

        return logits
    
class ReadabilityClassifier:
    backbone_name = "vinai/phobert-base-v2"

    def __init__(self, cp_path="model.pt"):
        cp_path = Path(cp_path)
        assert cp_path.exists(), "Checkpoint not found"

        if torch.cuda.is_available():
            print("Using GPU")
            self.device = torch.device("cuda")
        else:
            print("Using CPU")
            self.device = torch.device("cpu")

        self.model = CustomPhoBert(self.backbone_name).to(self.device)
        self.model.load_state_dict(torch.load(cp_path, map_location=self.device))

        self.tokenizer = AutoTokenizer.from_pretrained(self.backbone_name)

    def predict(self, text):
        text = self.__word_tokenize_recursive(text)
        if isinstance(text, str):
            text = [text]
        inp = self.tokenizer.batch_encode_plus(text, padding=True, truncation = True, max_length = 256, return_tensors="pt")   
        inp = {k: move_to(v, self.device) for k, v in inp.items()}
        with torch.no_grad():
            logits = self.model(**inp)
            logits = detach(logits)
            logits = logits.cpu().numpy()
            preds = logits.argmax(axis=-1)

        return str(preds[0])

    def __word_tokenize_recursive(self, texts):
        if not isinstance(texts, str):
            return [self.__word_tokenize_recursive(t) for t in texts]
        return word_tokenize(texts, format="text")
    
if __name__ == "__main__":
    model = ReadabilityClassifier()

    texts = [
        "Khái niệm diễn ngôn trong nghiên cứu văn học hôm nay. Thời gian gần đây khái niệm diễn ngôn đã xuất hiện rất nhiều trong các bài nghiên cứu đủ loại , nhiều đến mức không sao có thể định nghĩa thông suốt hết.",
        "Một con Sư Tử già , răng móng đã mòn đến mức nó chẳng còn dễ dàng gì mà kiếm được miếng mồi để ăn mà sống , nó bèn giả vờ nằm ốm ."
    ]

    print(model.predict(texts))
    print(model.predict(texts[0]))