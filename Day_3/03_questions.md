1. what is Greedy Decoding
   Greedy decoding selects the token with the highest probability at each step, leading to deterministic but potentially repetitive output.

2. what is Beam Search
   Beam search keeps track of the top-k most probable sequences at each step, exploring multiple paths to find the best overall sequence.

3. what is Pure Sampling
   Pure sampling randomly selects the next token based on the probability distribution, introducing randomness.

4. what is Top-k Sampling
   Top-k sampling limits the selection to the top-k most probable tokens, then samples from them, balancing diversity and quality.

5. what is Top-p (Nucleus) Sampling
   Top-p sampling selects the smallest set of tokens whose cumulative probability exceeds p, then samples from that set, allowing dynamic vocabulary size.

6. what is Temperature Effect
   Temperature scales the logits before applying softmax; lower values make output more deterministic, higher values more random.

7) what is top-k and top-p sampling? How do they differ and when would you use one over the other?
   Top-k samples from the top-k tokens, top-p from tokens until cumulative probability reaches p. Top-k is fixed number, top-p is dynamic. Use top-k for consistent diversity, top-p for adaptive based on distribution.

8) What is repetition penalty and how does it affect the generated text?
   Repetition penalty reduces the probability of tokens that have already been generated, encouraging more diverse and less repetitive text.

