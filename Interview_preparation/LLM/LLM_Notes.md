# Table of Contents
## [Finetuning Methods in LLM and Explanation](https://abvijaykumar.medium.com/fine-tuning-llm-parameter-efficient-fine-tuning-peft-lora-qlora-part-1-571a472612c4)
## [llm Finetuning methods ](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07)
## [RLHF Finetuning Method](https://www.labellerr.com/blog/reinforcement-learning-from-human-feedback/)

- [Evaluation Benchmarks for Emergent Abilities](#evaluation-benchmarks-for-emergent-abilities)
  - [BIG-Bench Suite](#big-bench-suite)
  - [TruthfulQA](#truthfulqa)
  - [Massive Multi-task Language Understanding (MMLU) Benchmark](#massive-multi-task-language-understanding-mmlu-benchmark)
  - [Word in Context (WiC) Benchmark](#word-in-context-wic-benchmark)
- [Other Factors That Could Give Rise To Emergent Abilities](#other-factors-that-could-give-rise-to-emergent-abilities)
  - [Open Source Large Language Models (LLMs)](#open-source-large-language-models-llms)
- [Understanding Hallucinations and Bias](#understanding-hallucinations-and-bias)
  - [Introduction](#introduction)
  - [Hallucinations in LLMs](#hallucinations-in-llms)
  - [Misinformation Spreading](#misinformation-spreading)
    - [Tuning the Text Generation Parameters](#tuning-the-text-generation-parameters)
    - [Leveraging External Documents with Retrievers Architectures](#leveraging-external-documents-with-retrievers-architectures)
  - [Bias in LLMs](#bias-in-llms)
    - [Constitutional AI](#constitutional-ai)
- [Hallucinations and Biases in Large Language Models (LLMs)](#hallucinations-and-biases-in-large-language-models-llms)
  - [Introduction](#introduction-1)
  - [Hallucinations in LLMs](#hallucinations-in-llms-1)
  - [Biases in LLMs](#biases-in-llms-1)
  - [Strategies to Mitigate Hallucinations and Biases](#strategies-to-mitigate-hallucinations-and-biases)
  - [Conclusion](#conclusion-1)

---

# Evaluation Benchmarks for Emergent Abilities

Several benchmarks are used to evaluate the emergent abilities of language models. These include the BIG-Bench suite, TruthfulQA, the Massive Multi-task Language Understanding (MMLU) benchmark, and the Word in Context (WiC) benchmark.

## BIG-Bench Suite

The BIG-Bench suite is a comprehensive set of over 200 benchmarks that test a model's capabilities across various tasks, including arithmetic operations, transliteration from the International Phonetic Alphabet (IPA), and word unscrambling. The performance of models like GPT-3 and LaMDA on these tasks starts near zero but jumps to significantly above random at a certain scale, demonstrating emergent abilities.

## TruthfulQA

TruthfulQA measures a model's capacity to provide truthful responses when addressing questions. It consists of two tasks: generation, where the model answers questions with 1 or 2 sentences, and multiple-choice, where the model chooses the correct answer from options or True/False statements. When scaled up to its largest size, models like Gopher show performance more than 20% above random, indicating the emergence of this ability.

## Massive Multi-task Language Understanding (MMLU) Benchmark

The MMLU benchmark evaluates models for their ability to demonstrate a broad range of world knowledge and problem-solving skills. It encompasses 57 tasks across various domains. Models like GPTs, Gopher, and Chinchilla do not perform better than guessing on average of all the topics at a specific scale, but scaling up enables performance to surpass random, indicating emergent abilities.

## Word in Context (WiC) Benchmark

WiC is a semantic understanding benchmark involving binary classification tasks for context-sensitive word embeddings. It aims to determine if target words share the same meaning in different contexts. Models like Chinchilla fail to achieve one-shot performance better than random even when scaled to their largest size. However, above-random performance eventually emerges when scaled to a much larger size, suggesting the emergence of this ability.

# Other Factors That Could Give Rise To Emergent Abilities

Multi-step reasoning and instruction following are strategies that could enhance model performance:
- Multi-step reasoning, known as chain-of-thought prompting, surpasses standard prompting when applied to sufficiently large models.
- Instruction following improves performance when applied to models of a specific size.

## Open Source Large Language Models (LLMs)

Some of the prominent open-source LLMs include LLaMA 2, Falcon, Dolly, Open Assistant, and Mistral.

# Understanding Hallucinations and Bias

## Introduction

In this lesson, we'll cover the concept of hallucinations in LLMs, highlighting their influence on AI applications and demonstrating how to mitigate them using techniques like the retriever's architectures. We'll also explore bias within LLMs with examples.

## Hallucinations in LLMs

In Large Language Models, hallucinations refer to cases where the model produces text that's incorrect and not based on reality. There are several possible reasons for these types of hallucinations:
- An LLM could be trained on a dataset that doesn’t have the knowledge required to answer a question.
- An LLM does not have a reliable way to check the factual accuracy of its responses.
- The training dataset used to train the LLM may include fictional and subjective content.

Strategies to mitigate hallucinations include tuning the text generation parameters, cleaning up the training data, precisely defining prompts (prompt engineering), and using retriever architectures to ground responses in specific retrieved documents.

## Misinformation Spreading

One significant risk associated with hallucinations in LLMs is their potential to generate content that, while appearing credible, is factually incorrect. Due to their limited capacity to understand the context and verify facts, LLMs can unintentionally spread misinformation.

### Tuning the Text Generation Parameters

The generated output of LLMs is greatly influenced by various model parameters, including temperature, frequency penalty, presence penalty, and top-p.

### Leveraging External Documents with Retrievers Architectures

Response accuracy can be improved by providing domain-specific knowledge to the LLM in the form of external documents. Retrieval-augmented generation (RAG) is a technique that enhances language model capabilities by sourcing data from external resources and integrating it with the context provided in the model's prompt.

## Bias in LLMs

Large language models like GPT-3.5 and GPT-4 have raised serious privacy and ethical concerns. Research has shown that these models are prone to inherent bias, leading to the generation of prejudiced or hateful language, intensifying the concerns regarding their use and governance.

Biases in LLMs arise from various sources: the data, the annotation process, the input representations, the models, and the research design. 

### Constitutional AI

'Constitutional AI' is a conceptual framework crafted by researchers at Anthropic. It aims to align AI systems with human values, ensuring that they become beneficial, safe, and trustworthy.

---

### Hallucinations and Biases in Large Language Models (LLMs)

#### Introduction
Hallucinations and biases are significant challenges faced by Large Language Models (LLMs), particularly in commercial applications. In this guide, we'll delve into strategies to effectively mitigate these issues in LLMs.

#### Hallucinations in LLMs
- **Definition:** Hallucinations occur when an LLM generates incorrect text responses that are not grounded in reality.
- **Mechanism

:** LLMs are trained to generate the most probable text continuation, leading to fabrication of answers when uncertain.
- **Impact:** Hallucinations can result in the spread of misinformation, posing dangerous implications, especially in critical domains like healthcare and education.

#### Biases in LLMs
- **Definition:** Biases manifest in LLMs, leading to specific tendencies in responses based on the biases present in the training data.
- **Example:** Gender bias may cause the model to predict certain occupations based on gender stereotypes.
- **Importance:** Addressing inherent biases and hallucinative tendencies is crucial for the responsible use of LLMs.

#### Strategies to Mitigate Hallucinations and Biases
1. **Good Data Preparation:** Clean and curate data from diverse and trustworthy sources to minimize biases and inaccuracies.
2. **Tweak Inference Parameters:** Adjust parameters like temperature, frequency penalty, presence penalty, and top-p to influence response generation and reduce hallucinations.
3. **Prompt Engineering:** Design precise prompts and refrain from generating responses when uncertain to avoid spreading misinformation.
4. **Leverage External Documents:** Use retrieval-augmented generation to ground responses in relevant external knowledge and enhance accuracy.
5. **Fine-tune on High-Quality Data:** Further train the model on unbiased data to improve its expertise and reduce biases.
6. **Constitutional AI:** Define principles for AI alignment and use reinforcement learning with AI feedback for training to promote ethical behavior.
7. **Stay Updated:** Keep track of new research and advancements in mitigating biases and hallucinations in LLMs to continuously improve mitigation strategies.

#### Conclusion
Addressing hallucinations and biases in LLMs is crucial for ensuring the reliability and fairness of AI-generated outputs. By implementing effective mitigation strategies and continuously refining model development processes, we can minimize the negative impacts of these issues and maximize the potential of LLMs for positive societal impact.