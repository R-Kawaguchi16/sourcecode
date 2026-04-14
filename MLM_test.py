import torch
from transformers import BertConfig, BertTokenizer, BertForMaskedLM
import re

text = '[CLS] i am interested in your bike [SEP]'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

print(text[6:-6] + '\n')

words = tokenizer.tokenize(text)
#print(words)

msk_ids = 3
words[msk_ids] = '[MASK]'

word_ids = tokenizer.convert_tokens_to_ids(words)
word_tensor = torch.tensor([word_ids])
#print(word_tensor)

#model.cuda()
model.eval()

x = word_tensor
y = model(x)
result = y[0]
#print(result.size())

_, msk_ids = torch.topk(result[0][3], k=5)
result_words = tokenizer.convert_ids_to_tokens(msk_ids.tolist())
#print(result_words)

sentence = ' '.join(words[1:-1])
print(sentence + '\n')

regex = re.compile(r'\[MASK\]')
for result_word in result_words:
    print(regex.sub(result_word, sentence))
    