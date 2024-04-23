# Notes on Fine-Tuning and its Methods

- [1. What is LLM Fine-tuning?](#what-is-llm-fine-tuning)
- [2. Key Steps in LLM Fine-tuning](#key-steps-in-llm-fine-tuning)
- [3. Fine-tuning Methods](#fine-tuning-methods)
  - [a. Full Fine Tuning (Instruction fine-tuning)](#full-fine-tuning-instruction-fine-tuning)
  - [b. Parameter Efficient Fine-Tuning (PEFT)](#parameter-efficient-fine-tuning-peft)
- [4. LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA)](#lora-low-rank-adaptation-and-qlora-quantized-lora)
- [Reinforcement Learning from Human Feedback](#reinforcement-learning-from-human-feedback)
- [Advantages of Fine-Tuning](#advantages-of-fine-tuning)
- [Disadvantages of Fine-Tuning](#disadvantages-of-fine-tuning)

## 1. What is LLM Fine-tuning? {#what-is-llm-fine-tuning}

Fine-tuning LLM involves additional training of a pre-existing model with a smaller, domain-specific dataset to improve model performance for specific tasks. It reduces computational requirements and training costs while achieving high performance on specific tasks.

## 2. Key Steps in LLM Fine-tuning {#key-steps-in-llm-fine-tuning}

1. Select a pre-trained model.
2. Gather relevant dataset.
3. Preprocess dataset.
4. Fine-tune the model on the domain-specific dataset.
5. Task-specific adaptation: Adjust model parameters based on the new dataset.

## Notes on Parameter-Efficient Fine-Tuning (PEFT) for LLMs

**Problem:** Training Large Language Models (LLMs) requires significant computational resources due to:

* **High memory usage:** Model weights (hundreds of GB), optimizer states, gradients, activations, and temporary memory.
* **Full fine-tuning:** Updating all model parameters during training, creating large memory footprints and storage issues for multiple tasks.

**Solution:** PEFT reduces memory and storage requirements by training only a subset of parameters.

**Benefits:**

* **Reduced memory footprint:** Enables training on single GPUs.
* **Less prone to catastrophic forgetting:** Preserves original LLM knowledge.
* **Efficient adaptation for multiple tasks:** Smaller trained weights and easier swapping for different tasks.

**Three main PEFT approaches:**

1. **Selective methods:** Update a subset of existing LLM parameters (layers, components, parameter types). Mixed performance, complex trade-offs. (Not covered in this course.)
2. **Reparameterization methods:** Reduce trainable parameters by creating low-rank transformations of original weights. Example: LoRA (covered in next video).
3. **Additive methods:** Introduce new trainable components while freezing original LLM weights. Two main approaches:
    * **Adapter methods:** Add new layers to the model architecture (e.g., encoder/decoder).
    * **Soft prompt methods:** Manipulate the input to achieve better performance.
        * Trainable prompt embeddings.
        * Retrain fixed input embedding weights.

**Example:** Soft prompt tuning technique for specific tasks.

**Additional notes:**

* PEFT methods have trade-offs between parameter efficiency, memory efficiency, training speed, model quality, and inference costs.
* Consider these trade-offs when choosing a PEFT method for your task.

**Further learning:**

* Next video: LoRA method for reparameterization.


## Additional Notes and Information for PEFT:

**General:**

* LLMs require specialized hardware like TPUs for full fine-tuning, making PEFT crucial for wider accessibility.
* PEFT can potentially save energy and computational resources, contributing to green AI practices.
* Continual learning for LLMs becomes feasible with PEFT, as adapting to new tasks doesn't require retraining from scratch.

**Selective methods:**

* Can be effective for tasks requiring specific model components (e.g., updating decoder layers for text generation).
* Identifying optimal parameters for update can be challenging.

**Reparameterization methods:**

* LoRA reduces memory usage by a factor of 10x while maintaining comparable performance.
* Other methods like Kronecker Factored Transformers (KFT) offer further memory and computation gains.

**Additive methods:**

* Adapters are flexible and can be easily added to different model architectures.
* Soft prompts are lightweight and efficient but might require careful prompt engineering.

**Examples:**

* Fine-tuning a pre-trained LLM for question answering using adapter modules on top of the decoder.
* Prompt tuning a pre-trained LLM for sentiment analysis by adding trainable parameters to the input embeddings.

**Trade-offs:**

* Parameter efficiency vs. model quality: Simpler PEFT methods might lead to slight performance drops compared to full fine-tuning.
* Memory efficiency vs. training speed: Some methods like LoRA require additional computations during training.

**Further Resources:**

* Papers: "Parameter-Efficient Fine-Tuning of Large Language Models: A Comprehensive Introduction" by Google AI, "Towards Better Parameter-Efficient Fine-Tuning for Large Language Models: A Position Paper" by Stanford University.
* Blogs: "Parameter Efficient LLM Fine-Tuning" by Dataiku, "What is Parameter-Efficient Fine-Tuning (PEFT) of LLMs?" by Hopsworks.


Here are some notes on the content you provided:

* Training LLMs is computationally intensive and requires a lot of memory, especially when full fine-tuning is used.
* Parameter efficient fine-tuning (PEFT) methods update a small subset of parameters in the LLM, reducing the number of trained parameters and making the training process more manageable.
* PEFT can be performed on a single GPU, making it less prone to catastrophic forgetting problems.
* With PEFT, the number of trained parameters can be as small as megabytes, depending on the task.
* There are several PEFT methods, including selective methods, reparameterization methods, and additive methods.
* Selective methods fine-tune only a subset of the original LLM parameters, while reparameterization methods reduce the number of parameters to train by creating new low-rank transformations of the original network weights.
* Additive methods introduce new trainable components to the model, such as new layers or parameters.
* Soft prompt methods are a type of additive method that manipulate the input to achieve better performance.
* Prompt tuning is a specific soft prompt method that adds trainable parameters to the prompt embeddings.

## 3. Fine-tuning Methods {#fine-tuning-methods}
- [Articles 1](https://abvijaykumar.medium.com/fine-tuning-llm-parameter-efficient-fine-tuning-peft-lora-qlora-part-1-571a472612c4)
- [Article 2](https://dassum.medium.com/fine-tune-large-language-model-llm-on-a-custom-dataset-with-qlora-fb60abdeba07)

### a. Full Fine Tuning (Instruction fine-tuning) {#full-fine-tuning-instruction-fine-tuning}

- Updates all model weights by training on examples guiding responses to queries.
- Suitable for enhancing model performance across various tasks.
- Resource-intensive due to computational requirements similar to pre-training.

### b. Parameter Efficient Fine-Tuning (PEFT) {#parameter-efficient-fine-tuning-peft}

- Efficient form of instruction fine-tuning.
- Updates only a subset of parameters, freezing the rest to manage memory requirements.
- Addresses storage issues when fine-tuning for multiple tasks.
- Techniques like LoRA and QLoRA are used for PEFT.
- Balances retaining pre-trained knowledge with adapting to target tasks efficiently.

## 4. LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) {#lora-low-rank-adaptation-and-qlora-quantized-lora}

LoRA fine-tunes smaller matrices approximating the weight matrix of the pre-trained model, resulting in a considerably smaller adapter. QLoRA further reduces memory footprint by quantizing weights of LoRA adapters to lower precision (e.g., 4-bit). Both methods enable reusing the original LLM, reducing overall memory requirements for handling multiple tasks.

## Advantages of Fine-Tuning {#advantages-of-fine-tuning}

- Improves model performance for specific tasks.
- Reduces computational requirements and training costs.
- Allows leveraging existing knowledge encoded in pre-trained models.
- Enables customization and specialization for domain-specific tasks.

## Disadvantages of Fine-Tuning {#disadvantages-of-fine-tuning}

- Resource-intensive for full fine-tuning, demanding significant computational resources.
- Overfitting can occur if the fine-tuning dataset is not representative.
- Requires careful selection and preprocessing of domain-specific datasets.
- Fine-tuning may not always result in performance improvement, depending on the task and dataset quality.

## LoRA Finetuning Method

### Low-Rank Parameters

Modularly add parameters with lower-dimensional space, reducing the need to modify the entire network.

### Low-Rank Adaptation (LoRA)

Adds a layer of trainable parameters with substantially reduced rank to the frozen original network, enhancing adaptability for domain-specific tasks.

- **Mechanism:** LoRA introduces an additional layer of trainable parameters while maintaining the original parameters frozen. These trainable parameters possess a substantially reduced rank compared to the dimensions of the original network.
- **Training Process:** During training, the pre-trained parameters of the original model remain frozen. A new set of parameters is added to the network, utilizing low-rank weight vectors. These vectors have dimensions represented as dxr and rxd, where ‘d’ is the dimension of the original frozen network parameters vector, and ‘r’ signifies the chosen low-rank or lower dimension.
- **Advantages:** 
  - Simplifies and expedites the process of adapting the original models for domain-specific tasks.
  - Preserves modularity by refraining from altering the original weights, thereby avoiding catastrophic forgetting.

## QLoRA Finetuning Method

### Quantized Low-Ranking Adaptation (QLoRA)

Enhances efficiency by quantizing weight values to reduce memory demands and faster calculations.

#### 4-bit NF4 Quantization

Optimized data type reducing memory footprint. Involves normalization, quantization, and dequantization steps.

- **Normalization & Quantization:** Weights are adjusted to zero mean and constant unit variance, then mapped to a limited set of values. Loss of data occurs due to quantization from high-resolution (FP32) to low-resolution (int4) data types.
- **Dequantization:** Reverse process to retrieve original weight values.
- **Advantages:** Reduces memory footprint, albeit with some loss of data.

#### Double Quantization

Further reduces memory footprint by quantizing quantization constants.

- **Process:** Quantizes quantization constants to optimize memory usage. Grouping weights per block, then applying quantization on these constants.
- **Advantages:** Significantly reduces memory demands, improving overall efficiency.

#### Unified Memory Paging

Utilizes nVidia unified memory feature for seamless GPU->CPU page transfers to manage memory spikes.

- **Feature:** Utilizes GPU->CPU page transfers for memory management, especially during memory spikes.
- **Advantages:** Manages memory overflow/overrun issues efficiently.

### 6. Advantages

- Retains valuable pre-trained knowledge.
- Modular approach enhances adaptability.
- Reduction in memory demands and faster calculations.
- Avoids catastrophic forgetting.

### 7. Disadvantages

- Requires understanding of techniques and implementation complexities.
- Loss of data due to quantization.
- May not yield optimal results with lower rank values.
- GPU dependency for unified memory paging.

---

# [Reinforcement Learning from Human Feedback](https://www.labellerr.com/blog/reinforcement-learning-from-human-feedback/)

## 1. Reinforcement Learning from Human Feedback (RLHF) Overview

RLHF combines reinforcement learning with human input in Natural Language Processing (NLP). Historically, reinforcement learning was limited to gaming, but RLHF extends its application to NLP.

## 2. Operation of RLHF

- **Initial Phase:** Begins with a pre-trained model serving as a benchmark.
- **Human Feedback:** Human testers evaluate the model's performance, assigning quality ratings.
- **Reinforcement Learning:** Rewards generated from human feedback fine-tune the model through iterative refinement.

## 3. Steps in RLHF Training Process

1. Pretraining a Language Model (LM): LM trained on diverse internet text data.
2. Choosing a Base Language Model: Selection based on task, resources, and complexity.
3. Acquiring and Preprocessing Data: Data cleaning and normalization for training.
4. Language Model Training: Refinement of model parameters using training data.
5. Model Assessment: Evaluation of model performance on isolated datasets.
6. Preparing for RLHF: Supplementary data collection to orient model towards human preferences.

## 4. Developing a Reward Model through Training

- **Creating the Reward Model:** Establishing a mechanism to associate input text sequences with numerical rewards.
- **Data Compilation:** Assembly of dataset comprising prompts and corresponding rewards.
- **Model Learning:** Training the model to associate outputs with reward values.
- **Incorporating Human Feedback:** Iterative refinement through human input to align with preferences.

## 5. Techniques to Fine-Tune Model with RLHF

- **Applying the Reward Model:** Assessing model outputs based on rewards from human feedback.
- **Establishing the Feedback Loop:** Iterative process guiding model towards preferred outputs.
- **Quantifying Differences using KL Divergence:** Statistical technique to measure distinctions in output distributions.
- **Fine-tuning through Proximal Policy Optimization (PPO):** Balancing exploration and exploitation for effective learning.
- **Discouraging Inappropriate Outputs:** Low-reward outputs are less likely to be repeated, incentivizing better responses.

## 6. Challenges Associated with RLHF

- **Variability and Human Mistakes:** Feedback quality varies among users, experts needed for complex inquiries.
- **Question Phrasing:** Accuracy influenced by the wording of questions.
- **Bias in Training:** Susceptibility to biases, especially in complex questions.
- **Scalability:** Time and resource-intensive process due to human input dependency.

## Key Points for Quick Revision: Fine-Tuning Language Models

1. **Purpose of Fine-Tuning:**
   - LLMs are versatile for various language tasks, but your application may require a specific task.
   - Fine-tuning pre-trained models can enhance performance on the desired task.

2. **Example of Fine-Tuning:**
   - For instance, improving summarization using a dedicated dataset for that task.
   - Remarkably, good results often achieved with just 500-1,000 examples, despite the model's extensive pre-training.

3. **Catastrophic Forgetting:**
   - Potential downside: Full fine-tuning modifies original LLM weights.
   - This may lead to catastrophic forgetting, affecting performance on tasks the model previously excelled at.

4. **Illustrative Scenario:**
   - Before fine-tuning, the model correctly identified named entities (e.g., "Charlie" as a cat's name).
   - After fine-tuning, the model may forget this task, causing confusion and displaying behaviors related to the new task.

5. **Mitigating Catastrophic Forgetting:**
   - **Assessment:** Evaluate if catastrophic forgetting impacts your use case.
   - **Multitask Fine-Tuning:** Fine-tune on multiple tasks simultaneously, requiring more data and compute (50-100,000 examples).

6. **Parameter Efficient Fine-Tuning (PEFT):**
   - **Definition:** A technique preserving most pre-trained weights, training only task-specific adapter layers and parameters.
   - **Advantage:** Greater robustness to catastrophic forgetting.
   - **Active Research Area:** Ongoing exploration in the field.

7. **Upcoming Topic: Multitask Fine-Tuning:**
   - Requires fine-tuning on multiple tasks concurrently.
   - Demands more data and compute resources (50-100,000 examples).

8. **Conclusion:**
   - Different fine-tuning approaches suit different needs.
   - PEFT is promising for avoiding catastrophic forgetting and maintaining general capabilities.

*Note: Ensure to understand the concepts of fine-tuning, catastrophic forgetting, multitask fine-tuning, and PEFT for a comprehensive overview.*


## Concise Explanation

1. **Fine-Tuning:**
   - **Purpose:** Adapting a pre-trained language model (LLM) for a specific task.
   - **Example:** Enhancing summarization using a dedicated dataset.
   - **Result:** Good performance often achieved with a small number of task-specific examples.

2. **Catastrophic Forgetting:**
   - **Issue:** Occurs during fine-tuning when modifying LLM weights for a specific task.
   - **Consequence:** May degrade performance on tasks the model previously excelled at.
   - **Example:** Forgetting named entity recognition after fine-tuning for summarization.

3. **Multitask Fine-Tuning:**
   - **Objective:** Fine-tune on multiple tasks simultaneously.
   - **Requirement:** Larger dataset (50-100,000 examples) and more computing resources.
   - **Consideration:** Ensures the model maintains its generalized capabilities across tasks.

4. **Parameter Efficient Fine-Tuning (PEFT):**
   - **Approach:** Preserves most pre-trained LLM weights.
   - **Implementation:** Trains only a small number of task-specific adapter layers and parameters.
   - **Advantage:** Greater resistance to catastrophic forgetting.
   - **Current Status:** An active area of research.

In summary, fine-tuning customizes a model for a specific task, but it may lead to catastrophic forgetting. Multitask fine-tuning addresses this by training on multiple tasks simultaneously, while PEFT is a technique preserving pre-trained weights for better robustness. Understanding these concepts provides a comprehensive overview of optimizing language models for specific applications.
