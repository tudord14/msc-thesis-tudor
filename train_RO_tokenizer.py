from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
import json
# -> here we will train a monolingual tokenizer for Romanian
# -> we will use the exact training data that we extracted
# -> architecture:
# -> BPE tokenizer with a vocab size of 40k
# -> we will save the tokenizer 
# -> special tokens:
special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>"]
# -> stages: normalization, pre-tokenization, tokenization-model, post-processing
# -> split text into preliminary chunks
tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# -> for RAM efficiency
def batch_iterator(file_path, batch_size=1000):
    batch = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            article = json.loads(line)
            batch.append(article['text'])
            if len(batch) == batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

trainer = trainers.BpeTrainer(vocab_size=40000,
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), 
                            special_tokens=special_tokens, 
                            show_progress=True)
tokenizer.train_from_iterator(batch_iterator("REMOVAL_LIST.jsonl"), trainer=trainer)

# -> we do this to ensure that the tokenizer adds the special tokens correctly during encoding
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)
tokenizer.decoder = decoders.ByteLevel()
tokenizer.save("ro_tokenizer.json")

# -> test the tokenizer
encoded = tokenizer.encode("Acesta este un test al tokenizer-ului nostru pentru limba română.")
print(encoded.tokens)   
print(encoded.ids)
# -> decode the tokens back to text
decoded = tokenizer.decode(encoded.ids)
print(decoded)