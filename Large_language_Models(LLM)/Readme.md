# Resources for Learning Generative AI

## Table of Contents
- [Complete Courses from Scratch to Production](#complete-courses-from-scratch-to-production)
  - [Articles](#articles-under-complete-courses-from-scratch-to-production)
  - [Courses](#courses-under-complete-courses-from-scratch-to-production)
  - [Notebooks/Hands-on Practice](#notebookshands-on-practice-under-complete-courses-from-scratch-to-production)
  - [YouTube Playlists](#youtube-playlists-under-complete-courses-from-scratch-to-production)
- [RAG](#rag)
  - [Articles](#articles-under-rag)
  - [YouTube Playlists](#youtube-playlists-under-rag)
- [Fine-tuning](#fine-tuning)
  - [Courses](#courses-under-fine-tuning)
  - [Notebooks/Hands-on Practice](#notebookshands-on-practice-under-fine-tuning)
- [LLMOps](#llmops)
  - [Courses](#courses-under-llmops)
- [Scaling Laws](#scaling-laws)

## Complete Courses from Scratch to Production
### Articles
- [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html)
### Courses
- [LLM Bootcamp Spring 2023: Prompt Engineering](https://fullstackdeeplearning.com/llm-bootcamp/spring-2023/prompt-engineering/)
- [Deeplearning.ai short courses on LLM](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [LLM  Course by mlbabonne](https://github.com/mlabonne/llm-course)
- [Nvidia LLM Course](https://www.nvidia.com/en-in/training/)
- [Weights and Bias LLM Courses](https://www.wandb.courses/pages/w-b-courses), [2](https://www.wandb.courses/courses/building-llm-powered-apps), [W&B Finetuning](https://www.wandb.courses/courses/training-fine-tuning-LLMs)
- [Generative AI & Large Language Models Courses by aciveloop](https://learn.activeloop.ai/)
- [Cohere Course](https://docs.cohere.com/docs/the-cohere-platform)
- [Hugging Face Courses](https://huggingface.co/learn)
  - [OpenSource Cookbook by Hugging face](https://huggingface.co/learn/cookbook/index)
  - [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter1/1?fw=pt)
- [NLP Specialization by Coursera](https://www.coursera.org/specializations/natural-language-processing)
- [More Courses from Coursera, edX, etc.]

### Notebooks/Hands-on Practice
- [LLM Notebooks](https://mlabonne.github.io/blog/posts/2023-06-07-Decoding_strategies.html)

### YouTube Playlists
- Krish Naik
  - [Foundational Generative AI by Ineuron](https://www.youtube.com/playlist?list=PLmQAMKHKeLZ-iTT-E2kK9uePrJ1Xua9VL)
  - [LLM Finetuning Playlist](https://www.youtube.com/playlist?list=PLZoTAELRMXVN9VbAx5I2VvloTtYmlApe3)
- Data Bricks: [Application Through Production](https://www.youtube.com/playlist?list=PLTPXxbhUt-YWSR8wtILixhZLF9qB_1yZm)
- [DataTrek Youtube Playlist for LLms](https://www.youtube.com/playlist?list=PL89V0TQq5GLofVxfT3D9hVK96ODiCghuM)

## RAG
### Articles
- [RAG Article](https://stackoverflow.blog/2023/10/18/retrieval-augmented-generation-keeping-llms-relevant-and-current/)

### YouTube Playlists
- [RAG From Scratch by Langchain](https://www.youtube.com/watch?v=wd7TZ4w1mSw&list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)

## Fine-tuning
### Courses
- [Weights and Bias LLM Courses](https://www.wandb.courses/pages/w-b-courses), [2](https://www.wandb.courses/courses/building-llm-powered-apps), [W&B Finetuning](https://www.wandb.courses/courses/training-fine-tuning-LLMs)
- [LLM Finetuning Courses By Deeplearning.ai](https://www.deeplearning.ai/short-courses/finetuning-large-language-models/)
- [RLHF Finetuning](https://www.deeplearning.ai/short-courses/reinforcement-learning-from-human-feedback/)

### Notebooks/Hands-on Practice
- [LLM Finetuning Notebooks](https://github.com/ashishpatel26/LLM-Finetuning)

## LLMOps
### Courses
- [LLMOps Course by Deeplearning.ai](https://www.deeplearning.ai/short-courses/llmops/)
- [Automated Testing for LLMOps Course by Deeplearning.ai](https://www.deeplearning.ai/short-courses/automated-testing-llmops/)

## Scaling Laws
Scaling laws refer to the relationship between the model's performance and factors such as the number of parameters, the size of the training dataset, the compute budget, and the network architecture. They were discovered after a lot of experiments and are described in the Chinchilla paper. These laws provide insights into how to allocate resources when training these models optimally.

The main elements characterizing a language model are:

- The number of parameters (N) reflects the model's capacity to learn from data. More parameters allow the model to capture complex patterns in the data.
- The size of the training dataset (D) is measured in the number of tokens (small pieces of text ranging from a few words to a single character).
- FLO

Ps (floating-point operations per second) measure the compute budget used for training.
