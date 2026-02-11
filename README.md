# msc-thesis-tudor


a) **Data stage:** </br>
- here i have already built a 20GB jsonl file of web crawled Romanian text which we will use for training </br>
- I also have around 5000 books in Romanian (170GB), in pdf format, which I have written a script to extract the text out of them -> we got around 3000 .txt clean files out of all of these (the others were in other formats or needed advanced OCR and time was not enough...)</br>
- with the books and the crawled text we have a solid corpus with mixed domanial coverage and high-quality language sources </br>
- now we have the final data resources: web crawled corpus (20GB), extracted high-quality literary books (1GB), and some wiki questions from huggingface </br>
- I want to implement a training technique that I believe is quite strong: when training the model, first use the web general language text (20GB corpus) as a strong language base, then at the end of the web sequence we train on books so that the language is refined and distributions are strenghtened and, as last resort, we include some question-answer knowledge into the training so that the model is truly general purpose and we thus make sure he has seen everything possible (I really hope this technique is effective...)
- I have written the script that trains the tokenizer for our model (we use BPE with 40k vocab trained on the whole corpus -> web+books+questions)

b) **Pretraining:** </br>
