1. What is Pipelines in NLP and how do they help in processing text data?
1. What are the different types of pipelines available in Hugging Face Transformers library?
1. What is NER?
1. What is sentiment classification?
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
1.