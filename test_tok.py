from tokenizers import Tokenizer

# -> just testing the tokenizer fast
tokenizer = Tokenizer.from_file("ro_tokenizer_40k.json")

test_sentences = [
    "Acesta este un test al tokenizer-ului nostru pentru limba română.",
    "Competențele sociale și emoționale sunt esențiale în educație.",
    "Literatura română are o istorie bogată și fascinantă.",
]

for sentence in test_sentences:
    encoded = tokenizer.encode(sentence)
    print(f"Tokens: {encoded.tokens}")
    print(f"Decoded: {tokenizer.decode(encoded.ids)}")
