from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import Sequence, NFD
import json
from tqdm import tqdm

# -> high-quality monolingual tokenizer for Romanian
# -> 21GB of text ensures robust subword statistics
# -> architecture:
# -> BPE tokenizer with vocab size 40k (good balance for 21GB corpus)
# -> special tokens for common NLP tasks
# -> normalization for consistency

special_tokens = ["<s>", "</s>", "<unk>", "<pad>", "<mask>", "<cls>", "<sep>"]
DATA = "/Volumes/KINGSTON/WEB_BOOKS_LITERARY.jsonl"
OUTPUT_TOKENIZER = "/Volumes/KINGSTON/ro_tokenizer.json"

# -> for RAM efficiency, batch iterator with progress tracking
def batch_iterator(file_path, batch_size=1000):
    batch = []
    line_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="reading training data"):
            try:
                article = json.loads(line)
                if 'text' in article and article['text']:
                    batch.append(article['text'])
                    line_count += 1
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            except json.JSONDecodeError:
                continue
    
    if batch:
        yield batch
    
    print(f"total lines processed: {line_count}")

print("initializing tokenizer...")
tokenizer = Tokenizer(models.BPE(unk_token="<unk>", byte_fallback=True))

# -> normalization: keep diacritics for Romanian
tokenizer.normalizer = Sequence([NFD(),])

# -> pre-tokenization: ByteLevel with space prefix to preserve spaces and diacritics
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

# -> trainer: 40k vocab is optimal for 21GB + Romanian morphology
print("training BPE tokenizer")
trainer = trainers.BpeTrainer(
    vocab_size=40000,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=special_tokens,
    show_progress=True,
    min_frequency=3,
)
tokenizer.train_from_iterator(batch_iterator(DATA), trainer=trainer)

# -> post-processing
print("setting up post-processing")
tokenizer.post_processor = processors.TemplateProcessing(
    single="<s> $A </s>",
    pair="<s> $A </s> $B:1 </s>:1",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

# -> decoder: ByteLevel decoder to reconstruct text perfectly
tokenizer.decoder = decoders.ByteLevel()

# -> save the trained tokenizer
print(f"saving tokenizer to {OUTPUT_TOKENIZER}...")
tokenizer.save(OUTPUT_TOKENIZER)

# -> test the tokenizer on some Romanian sentences
test_sentences = [
    "Acesta este un test al tokenizer-ului nostru pentru limba română.",
    "Competențele sociale și emoționale sunt esențiale în educație.",
    "Literatura română are o istorie bogată și fascinantă.",
]

for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    print(f"original: {sentence}")
    print(f"tokens: {encoded.tokens}")
    print(f"token IDs: {encoded.ids}")
    print(f"num tokens: {len(encoded.ids)}")
    
    # -> decode back
    decoded = tokenizer.decode(encoded.ids)
    print(f"decoded: {decoded}")

