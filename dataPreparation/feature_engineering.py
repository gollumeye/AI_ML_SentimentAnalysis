from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def get_context_embedding(text):

    tokens = tokenizer.encode(text, add_special_tokens=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(tokens)
        last_hidden_states = outputs.last_hidden_state

    return torch.mean(last_hidden_states, dim=1)