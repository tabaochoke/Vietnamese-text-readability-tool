from pathlib import Path
import sys, os

import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModel, AutoConfig
from underthesea import word_tokenize

__all__ = ["ReadabilityClassifier"]

# Model weight: https://drive.google.com/drive/folders/1nKWW4iz5gfKx3h04wGwzf1sfJt0bMhML?usp=sharing

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


class CustomPhoBert(nn.Module):
    def __init__(self, extractor_name):
        super(CustomPhoBert, self).__init__()

        config = AutoConfig.from_pretrained(extractor_name)
        config.update({"output_hidden_states": True})
        self.model = AutoModel.from_pretrained(extractor_name, config=config)

        self.drop = nn.Dropout(0.2)

        self.lstm = nn.LSTM(
            input_size=768, hidden_size=512, num_layers=1, batch_first=True
        )
        self.fc = nn.Linear(512, 4)

    def forward(self, batch):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        if input_ids.shape[0] != 1:
            return torch.cat(
                [
                    self.forward(
                        {
                            "input_ids": input_ids[i].unsqueeze(0),
                            "token_type_ids": token_type_ids[i].unsqueeze(0),
                            "attention_mask": attention_mask[i].unsqueeze(0),
                        }
                    )

                    for i in range(input_ids.shape[0])
                ],
                dim=0,
            )

        len_input = input_ids.shape[1]
        tensor_list = []
        max_length = 256

        for i in range(0, len_input, max_length):
            input_ids_temp = input_ids[:, i:i + max_length]
            token_type_ids_temp = token_type_ids[:, i:i + max_length]
            attention_mask_temp = attention_mask[:, i:i + max_length]

            logits = self.model(
                input_ids=input_ids_temp,
                attention_mask=attention_mask_temp,
                token_type_ids=token_type_ids_temp,
            )

            tensor_list.append(logits.last_hidden_state)

        if len(tensor_list) >= 2:
            output_tensor = torch.stack(tensor_list[:-1])
            output_tensor = output_tensor.squeeze(dim=1)
        else:
            output_tensor = tensor_list[0]

        _, (final_hidden_state, _) = self.lstm(output_tensor)
        logits = self.fc(final_hidden_state[-1])

        return torch.mean(logits, axis=0).unsqueeze(0)


class ReadabilityClassifier:
    backbone_name = "vinai/phobert-base-v2"
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    cp_path = os.path.join(base_path, 'model', 'model.pt')
    def __init__(self, cp_path=cp_path):
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
        inp = self.tokenizer(
            text,
            padding="max_length",
            truncation=False,
            max_length=256,
            return_tensors="pt",
        )
        inp = {k: move_to(v, self.device) for k, v in inp.items()}
        with torch.no_grad():
            logits = self.model(inp)
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
        "Một con Sư Tử già , răng móng đã mòn đến mức nó chẳng còn dễ dàng gì mà kiếm được miếng mồi để ăn mà sống , nó bèn giả vờ nằm ốm .",
    ]

    print(model.predict(texts))
    print(model.predict(texts[0]))
