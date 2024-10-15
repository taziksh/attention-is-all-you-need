# TODOs
- [x] Fix masking
- [x] Fix tokenizer
- [x] Positional encodings
- [x] Loss per token
- [x] `assert` every shape
- [x] fix loss calculation, use logits not softmax
- [x] fix learned weight dims
- [ ] qualitatively evaluate generations
- [ ] LayerNorm
- [ ] Batching
- [ ] Tokenizer (train on Shakespeare)
- [ ] KV cache

Loss graph for LAMBADA next word prediction      
<img width="1592" alt="Screenshot 2024-10-11 at 11 48 20 AM" src="https://github.com/user-attachments/assets/5910c8fe-91ff-41ff-8058-32c848311486">

# Approach
My goal was to end up with a deeper understanding of the Transformer architecture by writing it "from scratch". My primary sources were the Attention Is All You Need paper (Vaswani et al) and the Torch documentation. I was strict about LLMs, using them to help me check my reasoning and offer high-level feedback on code, but not for actually generating code. A lot of time spent just scribbling on pen and paper and bashing my head against the wall. Overall, pretty satisfied with the approach, as I feel more comfortable with Torch than I would have otherwise. Ofc I didn't do everything from scratch - towards the end I referred to Jalammar's visual guide to see if my intuition was on track, I used built in LayerNorm, BPE tokenizer etc. Building from scratch is an infinite rabbithole but I'm pretty happy with the compromise I struck in that fractal execise 

# Exercises

Here are some first principle questions to answer ([source](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md)):

- What is different architecturally from the Transformer, vs a normal
RNN, like an LSTM? (Specifically, how are recurrence and time managed?)
- Attention is defined as, Attention(Q,K,V) =
softmax(QK^T/sqrt(d_k))V. What are the dimensions for Q, K, and V? Why
do we use this setup? What other combinations could we do with (Q,K)
that also output weights?
- Are the dense layers different at each multi-head attention block? Why or why not?
- Why do we have so many skip connections, especially connecting the
input of an attention function to the output? Intuitively, what if we
didn't?
