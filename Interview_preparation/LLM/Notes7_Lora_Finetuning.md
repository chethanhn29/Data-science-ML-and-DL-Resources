**Low-rank Adaptation (LoRA):**

**Overview:**
- LoRA is a parameter-efficient fine-tuning technique for Language Models (LLMs).
- Reduces training parameters by injecting smaller rank-decomposition matrices alongside original weights.
- Freezes original LLM weights, training only the smaller matrices.

**Benefits of LoRA:**
- **Reduced memory footprint:** Trainable parameters decrease by 86% in an example, enabling single-GPU training.
- **Efficient adaptation for multiple tasks:** Swap out LoRA matrices for different tasks without retraining the entire model.
- **Less prone to catastrophic forgetting:** Preserves original LLM knowledge.

**How LoRA works:**
1. **Rank selection:** Choose the rank of the decomposition matrices (e.g., 4-32 for good performance).
2. **Matrix multiplication:** Multiply the two decomposition matrices (A and B) to create a matrix with the same dimensions as the original weights.
3. **Weight update:** Add the product matrix to the original frozen weights.
4. **Inference:** Use the updated weights for inference on the specific task.

**Performance:**
- LoRA achieves performance close to full fine-tuning (0.17 vs. 0.19 ROUGE 1 score increase in an example).
- Smaller trade-off in performance for a significant reduction in training parameters and computational resources.

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

**Conclusion:**
- LoRA is a powerful fine-tuning method, applicable not only to language models but also in other domains.
- Principles behind LoRA are valuable, and ongoing research may refine best practices for choosing the rank.
- Next exploration involves the final path method, focusing on training input text without altering the language model.
