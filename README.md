# msc-thesis-tudor

a) **Data stage:** </br>
- here i have already built a 20GB jsonl file of web crawled Romanian text which we will use for training </br>
- I also have around 5000 books in Romanian (170GB), in pdf format, which I have written a script to extract the text out of them -> we got around 3000 .txt clean files out of all of these (the others were in other formats or needed advanced OCR and time was not enough...)</br>
- these books needed even further cleaning so I swept through them manually and deleted the books that were noisy or just plain bad extracted by the pdf extract method</br>
- with the books and the crawled text we have a solid corpus with mixed domanial coverage and high-quality language sources </br>
- now we have the final data resources: web crawled corpus (around 20GB), extracted high-quality literary books (230MB) </br>
- I want to implement a training technique that I believe is quite strong: when training the model, first use the web general language text (20GB corpus) as a strong language base, then at the end of the web sequence we train on books so that the language is refined and distributions are strenghtened </br>
- I have written the script that trains the tokenizer for our model (we use BPE with 40k vocab trained on the whole corpus -> web+books) </br>
- finally concatenated the big final file that contains the web + books -> 21.93GB </br>
- trained the BPE 40k vocab size tokenizer </br>
- after tokenizing the full corpus, I have successfully packed the whole final corpus into 2048 length context blocks so we don't have to use padding and waste GPU compute time -> around 30GB of tokens</br>
- we are practically ready for training right now: corpus and tokenizer in place </br>

b) **Pretraining:** </br>
- we officially have 5,635,137,681 tokens in our training dataset!! (using a vocab size of 40.000)</br>
- we also have the training code for some model architectures that we want to try: </br>
1) Llama GQA: 4 KV heads for 16 Q heads; 23 layers -> aroudn 300M params </br>
   -> Training Time: 17-19 February (39h52m) - 1.25it/s
3) Mistral Sliding Window Attention + GQA: 23 layers -> around 300M params just like Llama </br>
   -> Training Time: 19-21 February (51h44m) - 1.63it/s
4) Falcon parallel attention + MQA: 1 KV head; 25 layers -> around 307M params </br>
   -> Training Time: 22-.. February (46h42m) - 1.47it/s
6) !!!DIFFICULT IMPLEMENTATION!!! Mamba2 State space model: no attention, 40 layers -> 305M params
7) Llama MHA standard baseline attention transformer: 21 layers -> 311M params

c) **Evaluation and Quantization**
- we will essentially quantize all the models into small variants and evaluate all the model on the same tasks </br>
- we will teste efficiency (RAM, storage, tokens per second, time to first token, etc...) and performance on perplexity and other Romanian tasks (to be decided...) </br>
