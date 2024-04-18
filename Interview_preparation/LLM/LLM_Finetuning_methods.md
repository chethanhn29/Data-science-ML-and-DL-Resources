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

