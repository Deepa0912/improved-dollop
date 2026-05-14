# Questions for day 3

1) What is NLP?
      NLP is a field of artificial intelligence focused on understanding and generating human language.
2) What is tokenization?
      Tokenization is the process of splitting text into smaller units, such as words, subwords, or characters.
3) What is vocabulary in tokenization?
      Vocabulary is the set of tokens the model knows and can map to numeric IDs.
4) What is GPT?
      GPT is a generative transformer model trained to predict the next token in a sequence.
5) What is BERT?
      BERT is a bidirectional transformer model trained to understand context from both directions.
6) What is the difference between GPT and BERT?
      GPT is designed for generation and left-to-right prediction, while BERT is designed for understanding and uses bidirectional context.
7) What is the importance of tokenization in NLP?
      Tokenization is important because it converts raw text into discrete inputs that models can process.
8) Where do we import AutoTokenizer from and what is its purpose?
      We import AutoTokenizer from `transformers`; it automatically loads the correct tokenizer for a given pretrained model.
9) What is Subword tokenization and how does it differ from word-level tokenization?
      Subword tokenization breaks text into smaller pieces than words, allowing efficient handling of rare words and open vocabulary, unlike word-level tokenization which uses whole words only.
