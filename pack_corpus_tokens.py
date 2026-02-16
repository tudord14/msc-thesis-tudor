from transformers import PreTrainedTokenizerFast
import json
import tqdm

# -> here we read the cleaned jsonl corpus, tokenize it and pack it into a new jsonl file where
# -> each object is a continuous stream of tokens up to 2048 tokens long, without splitting words across objects
# -> this way we dont have to use padding so we dont lose compute efficiency
corpus_path = "/Volumes/KINGSTON/WEB_BOOKS_LITERARY.jsonl"
output_path = "/Volumes/KINGSTON/PACKED_CORPUS_READY.jsonl"
tokenizer_path = "/Volumes/KINGSTON/ro_tokenizer.json"
CONTEXT_LENGTH = 2048

# -> initialize tokenizer
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
tokenizer.eos_token = "</s>" 
current_block = []

# -> stream through the corpus line by line to avoid loading everything into memory
# -> for each line, we tokenize the text and add the tokens to a continuous stream until we hit 2048 tokens
# -> at which point we save that block and start a new one
with open(corpus_path, 'r', encoding='utf-8') as f_in, open(output_path, 'w', encoding='utf-8') as f_out:
    for line in tqdm.tqdm(f_in, desc="streaming and packing"):
        try:
            data = json.loads(line)
            text = data['text']
            
            # -> tokenize the text
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.eos_token_id)
            
            # -> add tokens to the continuous stream
            for token_id in tokens:
                current_block.append(token_id)
                
                # -> as soon as we hit 2048, we save and clear
                if len(current_block) == CONTEXT_LENGTH:
                    json.dump({'input_ids': current_block}, f_out)
                    f_out.write('\n')
                    current_block = []
                    
        except Exception:
            continue

    # -> save the final partial block if it exists
    if current_block:
        json.dump({'input_ids': current_block}, f_out)
        f_out.write('\n')