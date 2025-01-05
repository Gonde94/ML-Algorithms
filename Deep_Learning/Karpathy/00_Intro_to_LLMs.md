### Intro to Large Language Models

<b><u>What is an LLM?</u></b>
Really, it is just two files: 1) the parameters file (the weights of the neural network) and, 2) code that runs those parameters. This is a self-contained package; it's all you need to run the model. The computational complexity comes in developing those parameters/weights.

<b><u>Training the model</u></b>
Think of training an LLM like compressing the internet. Llama-2-7B (7 billion parameters), for example, was trained on ~10TB of text data. This comes from a crawl of the internet. You would then train the model on ~6,000 GPUs for 12 days, at a cost of $2 million. Essentially the training compresses that text data into the model. For the larger models, these numbers are off by a factor of 10. 

<b><u>What is this neural network doing?</u></b>
It tries to predict the next word in a sequence.