## Table of Contents

1. [Introduction to Quantization in LLM](#introduction-to-quantization-in-llm)
2. [Types of Quantization in LLM](#types-of-quantization-in-llm)
   - [Post Training Quantization (PTQ)](#post-training-quantization-ptq)
   - [Quantization Aware Training (QAT)](#quantization-aware-training-qat)
3. [Advantages of Quantization in LLM](#advantages-of-quantization-in-llm)
4. [Disadvantages of Quantization in LLM](#disadvantages-of-quantization-in-llm)
5. [Applications of Quantization in LLM](#applications-of-quantization-in-llm)
6. [How Each Method Works](#how-each-method-works)
7. [Symmetric and asymmetric quantization](#symmetric-and-asymmetric-quantization)
8. [Scale Factor and Zero Point](#scale-factor-and-zero-point)
9. [How Does Quantization Work?](#how-does-quantization-work)
  - [Mapping to Lower-Precision Formats](#mapping-to-lower-precision-formats)
  - [Affine Quantization Scheme](#affine-quantization-scheme)
  - [Calibration and Outlier Handling](#calibration-and-outlier-handling)
  - [Quantization in Blocks](#quantization-in-blocks)
  - [Dequantization at Inference](#dequantization-at-inference)
10. [Different Techniques for LLM Quantization](#different-techniques-for-llm-quantization)
  - [QLoRA (Quantized Low-Rank Adaptation)](#qlora-quantized-low-rank-adaptation)
    - [NF4 (4-bit NormalFloat)](#nf4-4-bit-normalfloat)
    - [Double Quantization (DQ)](#double-quantization-dq)
  - [PRILoRA (Pruned and Rank-Increasing Low-Rank Adaptation)](#prilora-pruned-and-rank-increasing-low-rank-adaptation)

![Screenshot (83)](https://github.com/chethanhn29/Data-science-ML-and-DL-Resources/assets/110838853/f41bc2e0-dd3f-4e16-8547-285d2e869ea7)

### Introduction to Quantization in LLM
- [Krish Video Tutorial](https://www.youtube.com/watch?v=6S59Y0ckTm4&t=8s)
**Quantization is a model compression technique that converts the weights and activations within an LLM from a high-precision data representation to a lower-precision data representation. For example, it can convert data from a 32-bit floating-point number (FP32) to an 8-bit or 4-bit integer (INT4 or INT8). This reduction occurs in the model's parameters, specifically in the weights of the neural layers, and in the activation values that flow through the model's layers.**
- 
Quantization involves converting from a higher memory format to a lower memory format. It's crucial for compressing large models, making them suitable for deployment on devices with limited resources like mobile phones, edge devices, or smartwatches.

  
1. **Introduction to Quantization in LLM:**Quantization involves converting from a higher memory format to a lower memory format.
It's crucial for compressing large models, making them suitable for deployment on devices with limited resources like mobile phones, edge devices, or smartwatches.
   - **Definition:** Quantization in LLMs involves reducing the precision of numerical values used in the model's parameters (weights and activations) from higher precision formats, such as 32-bit floating-point (FP32), to lower precision formats, such as 16-bit floating-point (FP16) or 8-bit integers (int8).
   - **Importance:** LLMs, like other deep learning models, often contain millions or even billions of parameters, resulting in significant memory requirements. Quantization addresses this issue by compressing the model's parameters, making it feasible for deployment on resource-constrained devices like mobile phones or edge devices.

### Types of Quantization in LLM
   a. **Post Training Quantization (PTQ):** PTQ involves converting pre-trained LLMs into quantized models after the training phase.

      - **Process:** PTQ involves converting pre-trained LLMs into quantized models after the training phase.
      - **Steps:** It typically includes calibrating the model's weights and activations to lower precision formats, such as int8, while minimizing the loss of accuracy.
      - **Suitability:** PTQ is suitable for scenarios where sacrificing a small amount of accuracy is acceptable, as it offers a straightforward approach to converting existing models into quantized versions.

   b. **Quantization Aware Training (QAT):** QAT incorporates quantization considerations during the model training phase.
      - **Process:** QAT incorporates quantization considerations during the model training phase.
      - **Steps:** During training, the model is optimized with quantization constraints, ensuring that the parameters are compatible with lower precision formats while maintaining high accuracy.
      - **Suitability:** QAT is preferred when maintaining high accuracy is critical, as it allows for fine-tuning the model's parameters with quantization constraints, minimizing the loss of accuracy associated with quantization.
### Advantages of Quantization in LLM
- Reduced Memory Footprint
- Faster Inference
- Deployment Flexibility
   - **Reduced Memory Footprint:** Quantization significantly reduces the memory requirements of LLMs, enabling efficient deployment on devices with limited resources.
   - **Faster Inference:** Lower precision formats allow for faster inference, resulting in improved responsiveness and efficiency, particularly in real-time applications.
   - **Deployment Flexibility:** Quantized LLMs can be deployed across various platforms, including mobile devices, edge computing environments, and embedded systems, enabling a wide range of applications without sacrificing performance.

### Disadvantages of Quantization in LLM
- Accuracy Loss
- Calibration Overhead
- Fine-Tuning Complexity
   - **Accuracy Loss:** Quantization may lead to a loss of model accuracy due to reduced precision, especially in complex tasks or highly accurate models.
   - **Calibration Overhead:** PTQ requires additional calibration steps to adjust for precision loss, which can be computationally expensive and time-consuming.
   - **Fine-Tuning Complexity:** QAT involves retraining the model with additional training data to mitigate accuracy loss, adding complexity to the training process and requiring additional computational resources.

### Applications of Quantization in LLM
- Mobile Applications
- Edge Computing
- Embedded Systems
   - **Mobile Applications:** Quantized LLMs are well-suited for integrating natural language processing capabilities into mobile apps, enabling on-device inference with minimal resource usage.
   - **Edge Computing:** Deploying quantized LLMs on edge devices allows for real-time language processing without relying on cloud servers, improving privacy and reducing latency.
   - **Embedded Systems:** Quantized LLMs can be integrated into embedded systems for tasks like speech recognition, text generation, and chatbots, enabling intelligent interactions in IoT devices and wearables.

6. **How Each Method Works:**
   - **Post Training Quantization:** Converts pre-trained LLMs by quantizing weights and activations, followed by calibration to adjust for precision loss. This process prepares the model for deployment without requiring additional training.
   - **Quantization Aware Training:** Incorporates quantization constraints during model training, optimizing the model's parameters for lower precision formats while minimizing accuracy loss through fine-tuning with additional training data.

7. **Relevant Information:**
   - Techniques like CLAA (Calibration-less Activation Adjustment) and Laura (Layer-Wise Reduced Activation) can further optimize quantization for specific LLM architectures and tasks, enhancing efficiency and accuracy.
   - The choice between PTQ and QAT depends on the specific requirements of the application, with PTQ offering simplicity and efficiency at the cost of potential accuracy loss, while QAT provides better accuracy through fine-tuning but requires additional computational resources and training complexity.

These detailed explanations cover various aspects of quantization in LLMs, providing insights into its significance, implementation methods, advantages, disadvantages, applications, and relevant techniques. They offer a comprehensive understanding of how quantization optimizes LLMs for efficient deployment in real-world scenarios while balancing accuracy and resource constraints.

### Scale Factor and Zero Point:
- Scale factor: Determines the scaling of values during quantization.
- Zero point: Shifts the range of values to align with the desired format (e.g., unsigned int8).

The modes of quantization refer to different approaches used to apply quantization techniques to deep learning models. There are primarily two modes of quantization: post-training quantization (PTQ) and quantization-aware training (QAT). Let's delve into each mode and understand how they work:

#### Post Training Quantization (PTQ)
   - **Process:** PTQ involves converting pre-trained deep learning models into quantized versions after the training phase.
   - **Steps:**
     1. **Model Training:** Initially, the deep learning model is trained using standard techniques, typically using high-precision floating-point representations (e.g., 32-bit floating-point).
     2. **Quantization:** After training, the model's weights and possibly activations are quantized to lower-precision formats, such as 8-bit integers or 16-bit floating-point.
     3. **Calibration:** In PTQ, calibration steps may be necessary to fine-tune the quantized model to minimize the loss of accuracy caused by precision reduction. Calibration involves adjusting scaling factors or thresholds to ensure that the quantized model performs adequately.
     4. **Deployment:** Once quantization and calibration are complete, the quantized model is ready for deployment on target hardware, such as mobile devices or edge devices, where memory and computational resources are limited.

   - **How It Works:**
     - PTQ works by applying quantization to an already trained deep learning model, effectively reducing the precision of its parameters without retraining the model itself. This approach is relatively straightforward and does not require modifications to the training process.
     - However, PTQ may result in some loss of accuracy due to the conversion of high-precision values to lower precision. Calibration helps mitigate this accuracy loss by fine-tuning the quantized model's parameters.

#### Quantization Aware Training (QAT)
   - **Process:** QAT integrates quantization considerations directly into the training process of the deep learning model.
   - **Steps:**
     1. **Model Training with Quantization Constraints:** During the training phase, the model is optimized with quantization constraints, ensuring that the parameters are compatible with lower-precision formats.
     2. **Fine-tuning:** Additional training iterations may be performed with quantization-aware objectives to further refine the model's parameters while minimizing accuracy loss.
     3. **Quantization:** After training, the model is already optimized for quantization, and the conversion to lower-precision formats (e.g., int8 or FP16) can be performed without additional calibration steps.
     4. **Deployment:** The quantized model resulting from QAT is ready for deployment on target hardware, maintaining a balance between accuracy and resource efficiency.

   - **How It Works:**
     - QAT modifies the training process to account for the effects of quantization on model performance. By incorporating quantization constraints and objectives into the optimization process, QAT ensures that the model's parameters are trained to be quantization-friendly.
     - This approach allows the model to learn representations that are robust to lower precision, minimizing the accuracy loss typically associated with post-training quantization.
     - QAT typically requires more computational resources and training time compared to PTQ but often yields quantized models with higher accuracy.

In summary, PTQ involves quantizing pre-trained models after training, while QAT integrates quantization considerations into the training process itself. Both approaches aim to reduce the memory footprint and computational requirements of deep learning models for deployment on resource-constrained devices, with QAT often yielding quantized models with higher accuracy due to its optimization during training.

## Symmetric and asymmetric quantization 
Symmetric and asymmetric quantization  are two approaches used to convert the weights and activations of deep learning models from high-precision floating-point representations to lower-precision formats, such as integers. Let's explore each approach in detail:

1. **Symmetric Quantization:**
   - **Overview:** Symmetric quantization aims to evenly distribute the range of values between the minimum and maximum values of the original data.
   - **Process:**
     1. **Range Determination:** Identify the minimum and maximum values of the data (e.g., weights, activations) to be quantized.
     2. **Scaling:** Compute a scale factor based on the desired quantization range (e.g., 8-bit integer range from 0 to 255).
     3. **Quantization:** Divide the entire range of values into discrete intervals based on the scale factor, mapping each original value to the nearest quantized value within the defined range.
   - **Example:** If the original data range is from -10 to 10, symmetric quantization would map this range to the quantized range of 0 to 255 by evenly distributing the values within this range.
   - **Advantages:**
     - Simple to implement and understand.
     - Suitable for data distributions that are approximately symmetric.
   - **Disadvantages:**
     - May lead to suboptimal quantization for asymmetric data distributions.
     - Can result in information loss if the data distribution is highly skewed.

2. **Asymmetric Quantization:**
   - **Overview:** Asymmetric quantization allows for the mapping of data values to a quantized range that is not necessarily centered around zero.
   - **Process:**
     1. **Range Determination:** Identify the minimum and maximum values of the data to be quantized.
     2. **Scaling:** Compute a scale factor based on the desired quantization range.
     3. **Quantization:** Map the original data values to the quantized range, which may be asymmetrically distributed around zero.
   - **Example:** In asymmetric quantization, if the original data range is from -20 to 20, the quantized range may be mapped from 0 to 255, with the minimum value of -20 mapped to 0 and the maximum value of 20 mapped to 255.
   - **Advantages:**
     - Allows for more flexibility in mapping data values to quantized ranges, accommodating asymmetric data distributions.
     - Can provide better accuracy preservation for skewed data distributions compared to symmetric quantization.
   - **Disadvantages:**
     - Requires additional consideration and computation compared to symmetric quantization.
     - May introduce complexity in implementation due to the asymmetric nature of the quantization range.
    



## How Does Quantization Work?

### Mapping to Lower-Precision Formats

The quantization process involves mapping weights stored in high-precision values to lower-precision data types. For example, mapping a 64-bit or 32-bit float to a 16-bit float is relatively straightforward, but quantizing a 32-bit float value to a 4-bit integer is more complex due to the limited representation of integers.

### Affine Quantization Scheme

The affine quantization scheme is commonly used, represented by the formula:

```
x_q = round(x / S + Z)
```

Where:
- `x_q` is the quantized integer value corresponding to the floating-point value `x`.
- `S` is an FP32 scaling factor.
- `Z` is the zero-point, representing the INT4 value corresponding to 0 in the FP32 space.
- `round` refers to rounding the result to the nearest integer.

### Calibration and Outlier Handling

Before quantization, the model is calibrated using a smaller dataset to determine the range `[min, max]` of FP32 values. Outliers may have a disproportionate impact on scaling, so techniques like clipping and quantizing in blocks are used to handle them effectively.

### Quantization in Blocks

Quantization in blocks involves dividing weights into groups and quantizing each block individually. This mitigates the impact of outliers and increases precision, but also increases the number of scaling factors that must be stored.

### Dequantization at Inference

During inference, the quantized weights and activations are dequantized to perform necessary computations with higher precision data types. This ensures that the model's accuracy is maintained during forward and backward propagation.

## Different Techniques for LLM Quantization

### QLoRA (Quantized Low-Rank Adaptation)

QLoRA reduces the memory requirements of LLMs by quantizing weights to 4-bit. It utilizes the NF4 (4-bit NormalFloat) data type and Double Quantization (DQ) for additional memory savings.

#### NF4 (4-bit NormalFloat)

NF4 normalizes each weight to a value between -1 and 1 for a more accurate representation of lower precision weight values compared to conventional 4-bit floats.

#### Double Quantization (DQ)

DQ quantizes the scaling factors for each block of weights, reducing memory requirements further.

### PRILoRA (Pruned and Rank-Increasing Low-Rank Adaptation)

PRILoRA increases efficiency by linearly increasing the rank for each layer and performing importance-based A-weight pruning. This reduces the time and memory requirements of fine-tuning an LLM while maintaining performance.


In summary, symmetric quantization aims to evenly distribute the data values within a specified quantization range, while asymmetric quantization allows for more flexibility by accommodating data distributions that are not centered around zero. The choice between symmetric and asymmetric quantization depends on the characteristics of the data and the specific requirements of the application.
