1. What is Pipelines in NLP and how do they help in processing text data?
   ---Pipelines in NLP are high-level APIs that simplify the use of pretrained models for common tasks like text classification, generation, etc. They help by abstracting the preprocessing, model inference, and postprocessing steps into a single function call.

1. What are the different types of pipelines available in Hugging Face Transformers library?
   ---Common types include: text-classification, token-classification (NER), question-answering, text-generation, translation, summarization, etc.

1. What is NER?
   ---NER stands for Named Entity Recognition, a task that identifies and classifies named entities in text, such as persons, organizations, locations, etc.

1. What is sentiment classification?
   ---Sentiment classification is the task of determining the sentiment or emotion expressed in a piece of text, such as positive, negative, or neutral.

1. Explain the below code snippet:
    ```python
    generator = pipeline("text-generation", model="gpt2")

    prompt = "Artificial intelligence will transform the world by"
    results = generator(
        prompt,
        max_new_tokens=60,
        num_return_sequences=2,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )
    ```
   ---The code creates a text generation pipeline using GPT-2 model. It generates text starting from the prompt "Artificial intelligence will transform the world by", producing up to 60 new tokens, returning 2 sequences, with sampling enabled, temperature 0.8 for creativity, and top_p 0.9 for nucleus sampling.
