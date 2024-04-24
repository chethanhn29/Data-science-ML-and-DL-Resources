## [LoRa, QLoRA Full Explanation](https://www.unite.ai/lora-qlora-and-qa-lora-efficient-adaptability-in-large-language-models-through-low-rank-matrix-factorization/)

- [Practical-tips-for-finetuning-llms](https://magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms)

Here's the Table of Contents with anchor links:

- [**Low-rank Adaptation (LoRA):**](#low-rank-adaptation-lora)

  - [**Overview:**](#overview)
  
  - [**Challenges in Full Parameter Fine-Tuning:**](#challenges-in-full-parameter-fine-tuning)
  
  - [**Benefits of LoRA:**](#benefits-of-lora)
  
  - [**Challenges Addressed by LoRA:**](#challenges-addressed-by-lora)
  
  - [**Core Principles of LoRA:**](#core-principles-of-lora)
  
    - [**Tracking Weight Changes:**](#tracking-weight-changes)
    
    - [**Matrix Decomposition:**](#matrix-decomposition)
    
    - [**Efficient Weight Updates:**](#efficient-weight-updates)

  - [**LoRA Fine-Tuning: A Step-by-Step Breakdown**](#lora-fine-tuning-a-step-by-step-breakdown)

  - [**Matrix Decomposition Process:**](#matrix-decomposition-process)
  
  - [**Performance:**](#performance)
  
  - [**Advantages of LoRA:**](#advantages-of-lora)
  
  - [**Disadvantages of LoRA:**](#disadvantages-of-lora)

- [**Additional notes:**](#additional-notes)

  - [**Further learning:**](#further-learning)

- [**Practical Example:**](#practical-example)

- [**Memory Efficiency:**](#memory-efficiency)

- [**Performance Comparison:**](#performance-comparison)

- [**Choosing LoRA Rank:**](#choosing-lora-rank)

- [**Quantized LoRa (CLoRA):**](#quantized-lora-clora)

- [**Simpler breakdown of QLoRA:**](#simpler-breakdown-of-qlora)

## Low-rank Adaptation (LoRA):

## Overview:
- LoRA is a parameter-efficient fine-tuning technique for Language Models (LLMs).
- Reduces training parameters by injecting smaller rank-decomposition matrices alongside original weights.
- Freezes original LLM weights, training only the smaller matrices.

![](https://media.licdn.com/dms/image/D4E12AQGdQChTCAJNRQ/article-cover_image-shrink_600_2000/0/1690809637997?e=2147483647&v=beta&t=SSYhObQXoZ5K7myUcCXEwbEzt77PxBtUli_9lhkz7Tw)

### Challenges in Full Parameter Fine-Tuning:
- Full parameter fine-tuning involves updating all weights of the base model, which can be resource-intensive.
- Updating billions of parameters can lead to hardware resource constraints and inefficiencies in downstream tasks.

### **Benefits of LoRA:**
- **Reduced memory footprint:** Trainable parameters decrease by 86% in an example, enabling single-GPU training.
- **Efficient adaptation for multiple tasks:** Swap out LoRA matrices for different tasks without retraining the entire model.
- **Less prone to catastrophic forgetting:** Preserves original LLM knowledge.

### **Challenges Addressed by LoRA:**
   - **Resource Constraints:** Updating all parameters during fine-tuning can strain computational resources.
   - **Efficiency:** Traditional fine-tuning methods may not be computationally efficient for large models.
   - **Complexity:** Managing and updating billions of parameters can be complex and resource-intensive.

### **Core Principles of LoRA:**
   - **Weight Change Tracking:** LoRA tracks changes in weights rather than updating all parameters directly.
   - **Matrix Decomposition:** Original weight matrices are decomposed into smaller matrices based on a specified rank.
   - **Efficient Updates:** By tracking weight changes using smaller matrices, LoRA optimizes the fine-tuning process.

    - **Tracking Weight Changes:**
    - During fine-tuning, LoRA tracks the changes made to the weights of the base LLM instead of updating all parameters directly.
    - This tracking process involves decomposing the original weight matrices into smaller matrices based on a specified rank.

    - **Matrix Decomposition:**
    - Original weight matrices of the LLM are decomposed into two smaller matrices.
    - Decomposition is based on a rank parameter, which determines the size of the resulting matrices.
    - Smaller matrices require fewer parameters to store weight changes, reducing memory overhead.

    - **Efficient Weight Updates:**
    - By tracking weight changes using smaller matrices, LoRA optimizes the fine-tuning process.
    - Only the changed weights need to be updated during fine-tuning, reducing computational complexity and resource requirements.


### **LoRA Fine-Tuning: A Step-by-Step Breakdown**

1. **Preparation:**
    * You have a pre-trained LLM (Large Language Model) with a massive weight matrix. 
    * You also have a specific task or dataset for which you want to fine-tune the LLM. 

2. **Rank Selection:**
    * Choose a rank (a small number, typically between 4 and 32) for the two low-rank matrices (A and B) . This rank determines the complexity of the updates captured by these matrices. 

3. **Matrix Decomposition (Optional):**
    * This step provides a deeper understanding of how LoRA works. You can skip it if the concept of matrix decomposition is unfamiliar. 
        * The original weight matrix (W₀) of the LLM is decomposed into two smaller matrices, A and B, using a chosen rank. 
        *  Multiply the two decomposition matrices (A and B) to create a matrix with the same dimensions as the original weights.
        * This decomposition essentially captures the most significant changes needed for the specific task within these smaller matrices. 

4. **Weight Update Tracking:**
    * Instead of directly updating all the weights (W₀) of the pre-trained LLM, LoRA uses matrices A and B to track the essential changes required for the task.
    * This is why A and B are called "adapter" matrices.

5. **Fine-Tuning the Adapter:**
    * Only matrices A and B are trained on the task-specific data. 
    * The original LLM's weights (W₀) remain frozen and unchanged during this process. 

6. **Combined Weights for Inference:**
    * During use for predictions, the original weights (W₀) and the update captured by the product of A and B are combined.
        * Essentially, W_updated = W₀ + A * B

![](https://substackcdn.com/image/fetch/f_auto,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F5dfbd169-eb7e-41e1-a050-556ccd6fb679_1600x672.png)

**Matrix Decomposition Process:**
   - **Rank Parameter:** Determines the size of the resulting smaller matrices.
   - **Decomposition Benefits:** Reduces memory overhead by requiring fewer parameters to store weight changes.
   - **Selective Weight Updates:** Only changed weights need to be updated during fine-tuning, reducing computational complexity.

**Performance:**
- LoRA achieves performance close to full fine-tuning (0.17 vs. 0.19 ROUGE 1 score increase in an example).
- Smaller trade-off in performance for a significant reduction in training parameters and computational resources.

**5. Advantages of LoRA:**
   - **Resource Efficiency:** Minimizes memory requirements for tracking weight changes.
   - **Computational Optimization:** Improves fine-tuning efficiency by selectively updating weights.
   - **Performance Enhancement:** Fine-tunes LLMs while maintaining or enhancing performance on downstream tasks.

### **Disadvantages of LoRA:**

1. **Rank Selection Complexity:** Choosing the right rank for matrix decomposition can be tricky, requiring experimentation and adding complexity to implementation.
  
2. **Loss of Precision:** Matrix decomposition may reduce precision compared to updating all parameters directly, potentially impacting the model's ability to capture subtle patterns.
  
3. **Limited Performance Improvement:** LoRA may not significantly boost performance beyond the pre-trained model's capabilities, especially in scenarios requiring extensive fine-tuning.
  
4. **Dependency on Initial Model Quality:** LoRA's effectiveness relies on the quality and relevance of the pre-trained model, limiting its utility if the base model lacks robustness.
  
5. **Complexity Overhead:** Implementing LoRA entails understanding complex principles like rank selection and matrix decomposition, which may pose challenges for researchers and practitioners.

**Additional notes:**
- LoRA focuses on self-attention layers for parameter efficiency (most LLM parameters reside there).
- Other components like feed-forward layers can also be used with LoRA.
- Choosing the optimal rank is an active research area.
- Principles of LoRA extend beyond LLMs to other model domains.

**Further learning:**
- Papers: "LoRA: Low-Rank Adaptation of Large Language Models" by Microsoft AI, "Parameter-Efficient Fine-Tuning of Large Language Models: A Comprehensive Introduction" by Google AI.

**Low-rank Adaptation (LoRA):**

**Overview:**
- LoRA is a parameter-efficient fine-tuning technique falling under re-parameterization.
- Transformer architecture involves tokenization, embedding, and self-attention/feedforward networks in encoder/decoder parts.
- Weights are learned during pre-training, and during full fine-tuning, all parameters are updated.
- LoRA reduces trainable parameters by freezing original model parameters and introducing low-rank matrices.

**LoRA Process:**
- Freeze original model weights; inject low-rank matrices with dimensions ensuring product matches original weights.
- Train smaller matrices via supervised learning, updating them instead of original weights.
- For inference, multiply low-rank matrices, add to frozen weights, and replace original weights in the model.
- LoRA fine-tuned model is task-specific, maintaining the same number of parameters as the original model.

**Practical Example:**
- Transformer weights in the Attention is All You Need paper: 512 by 64.
- LoRA rank = 8; two low-rank matrices have dimensions 8x64 and 512x8, totaling 4,608 trainable parameters (86% reduction).

**Memory Efficiency:**
- LoRA matrices have small memory footprint.
- Can fine-tune different sets for various tasks, switch at inference time, and avoid storing multiple full-size models.

**Performance Comparison:**
- ROUGE metric used for performance evaluation.
- FLAN-T5 base model, full fine-tuned model, and LoRA fine-tuned model compared for dialogue summarization.
- LoRA shows a boost in performance (ROUGE 1 score increased by 0.17), slightly below full fine-tuning, but with significantly fewer parameters.

**Choosing LoRA Rank:**
- Rank choice impacts trainable parameters and model performance.
- Microsoft researchers found a plateau in loss value for ranks > 16; ranks in the range of 4-32 offer a good trade-off between parameter reduction and performance.
- The rank parameter determines the size of the decomposed matrices.
- Higher ranks may allow the model to learn more complex patterns but require more parameters.
- Rank selection depends on the complexity of the task and available computational resources.

**Quantized LoRa (CLoRA):**
- CLoRA involves quantizing model parameters to further reduce memory requirements.
- Parameters are converted from higher precision formats (e.g., float16) to lower precision (e.g., 4-bit) for efficient storage and computation.
- CLoRA combines quantization with low-rank adaptation to achieve even greater efficiency in fine-tuning large language models.

**Here’s a simpler breakdown of QLoRA:**

- Initial Quantization: First, the Large Language Model (LLM) is quantized down to 4 bits, significantly reducing the memory footprint.
- LoRA Training: Then, LoRA training is performed, but in the standard 32-bit precision (FP32).
