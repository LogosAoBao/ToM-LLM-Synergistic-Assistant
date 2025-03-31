# mock_models.py
"""
模拟模型实现，用于在没有实际预训练模型的情况下进行系统测试
"""

import torch
from torch import nn
import os
from typing import Dict, List, Optional, Union, Any


class MockModel(nn.Module):
    """模拟的预训练模型"""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {"model_type": "mock", "hidden_size": 768, "vocab_size": 30000}
        self.embeddings = nn.Embedding(self.config["vocab_size"], self.config["hidden_size"])
        self.encoder = nn.Linear(self.config["hidden_size"], self.config["hidden_size"])
        self.decoder = nn.Linear(self.config["hidden_size"], self.config["hidden_size"])

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids is None:
            input_ids = torch.zeros(1, 10).long()

        embeddings = self.embeddings(input_ids)
        hidden_states = self.encoder(embeddings)
        outputs = self.decoder(hidden_states)

        return {
            "last_hidden_state": outputs,
            "hidden_states": [outputs] * 4  # 模拟多层输出
        }

    def to(self, device):
        """模拟设备转移"""
        return self


class MockTokenizer:
    """模拟的分词器"""

    def __init__(self, config=None):
        self.config = config or {"model_type": "mock", "vocab_size": 30000}
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4}
        for i in range(5, self.config["vocab_size"]):
            self.vocab[f"token_{i}"] = i

    def encode(self, text, add_special_tokens=True, **kwargs):
        """模拟编码过程"""
        # 简单地将文本长度作为标记数量
        tokens = [2] if add_special_tokens else []  # [CLS]
        tokens.extend([1] * min(len(text.split()), 50))  # 用[UNK]填充
        if add_special_tokens:
            tokens.append(3)  # [SEP]
        return tokens

    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        """模拟解码过程"""
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [0, 2, 3, 4]]
        return " ".join([f"<{t}>" for t in token_ids])

    def __call__(self, text, padding=True, truncation=True, return_tensors=None, **kwargs):
        """模拟分词器调用"""
        if isinstance(text, str):
            tokens = self.encode(text)
        else:
            tokens = [self.encode(t) for t in text]
            max_len = max(len(t) for t in tokens)
            if padding:
                tokens = [t + [0] * (max_len - len(t)) for t in tokens]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(tokens),
                "attention_mask": torch.ones_like(torch.tensor(tokens))
            }
        else:
            return {
                "input_ids": tokens,
                "attention_mask": [[1] * len(t) for t in tokens]
            }


def create_mock_model_and_tokenizer(model_type="base"):
    """创建指定类型的模拟模型和分词器"""
    if model_type == "perception":
        config = {"model_type": "vision-language", "hidden_size": 1024, "vocab_size": 50000}
    elif model_type == "tom":
        config = {"model_type": "language", "hidden_size": 1536, "vocab_size": 30000}
    elif model_type == "planning":
        config = {"model_type": "language", "hidden_size": 1024, "vocab_size": 30000}
    else:
        config = {"model_type": "base", "hidden_size": 768, "vocab_size": 30000}

    model = MockModel(config)
    tokenizer = MockTokenizer(config)

    return model, tokenizer