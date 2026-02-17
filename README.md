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
1) Llama GQA: 4 KV heads for 16 Q heads; 23 layers -> aroudn 300M params
2) Mistral Sliding Window Attention + GQA: 23 layers -> around 300M params just like Llama
3) Falcon parallel attention + MQA: 1 KV head; 25 layers -> around 307M params
4) Mamba2 State space model: no attention, 40 layers -> 305M params
5) Llama MHA standard baseline attention transformer: 21 layers -> 311M params
6) My personal transformer model where I created a specialized Attention mechanism specifically built for small language models (more to be explained later...)
- Tuesday 17 February: Llama GQA 300.3M Parameters Model start training at 14:50, it will take approx. 40 hours, without the evaluation step which isn't more than 10 minutes once every 2000 steps; after this model finishes we can start the Mistral GQA Sliding Attention 300.3M Parameters Model
