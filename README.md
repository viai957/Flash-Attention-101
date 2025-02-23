# FlashAttention From Scratch 🚀

## **Overview**
This repository is a deep dive into **FlashAttention**, implemented from first principles using **Triton**, **CUDA Kernels**, and **PyTorch**. The goal is to provide an in-depth understanding of the FlashAttention mechanism by breaking it down step-by-step and implementing it from scratch. 

Key highlights:
- Mathematical derivation with **handwritten notes** 📜
- **Jupyter Notebook scratchpad** with explanations at each step
- **Triton and CUDA kernel implementations** ⚡
- **Benchmarking**: Comparing naive PyTorch vs. Triton implementations 🔥
- **Google Colab Notebook** for easy experimentation 🎯


## **Work Under Progress** 🛠️
- [ ] **Sliding Window Attention** for improved efficiency in long sequences.
- [ ] **MoBA (Mixture of Block Attention)** to enhance block-wise computations for better performance.

---

## **Folder Structure** 📂

```markdown
/FlashAttention-From-Scratch
│── cuda/                         # CUDA kernel implementations for FlashAttention
│── notes/                        # Handwritten notes for derivation and explanation
│── triton/                       # Triton-based implementation of FlashAttention
│── FlashAttention_Benchmark.ipynb  # Jupyter notebook comparing naive PyTorch vs Triton FlashAttention
│── flash-attention-notes.ipynb    # Notebook containing detailed notes and derivations
│── scratch-pad.ipynb              # Jupyter scratchpad for experimenting with implementations
│── triton-scratch-pad.ipynb       # Scratchpad for Triton-specific implementations and testing
```

---

## **Understanding FlashAttention** 🧠
### **1. What is FlashAttention?**
FlashAttention is an **optimized attention mechanism** that significantly reduces the memory overhead of self-attention by:
- **Computing attention in blocks** (reducing memory footprint)
- **Leveraging hardware-efficient optimizations** (such as fused operations)
- **Reducing redundant computation** while maintaining accuracy

This makes FlashAttention particularly effective for **large transformer models**, as it enables scaling without running into memory bottlenecks.

### **2. Mathematical Derivation** ✏️
The repository includes **handwritten notes** with a step-by-step breakdown of the derivation. Key components covered:
- Standard **Self-Attention** formulation
- How FlashAttention **optimizes memory usage**
- **Rewriting the attention mechanism** in a more hardware-efficient way

You can find these in the `notes/` folder and the **flash-attention-notes.ipynb** notebook.

---

## **Implementation Details** ⚙️
### **1. PyTorch Naive Implementation**
The first step is implementing self-attention **naively** in PyTorch. This serves as a baseline to:
- Understand standard attention computations
- Compare performance later

This is included in the `scratch-pad.ipynb` notebook.

### **2. Triton Implementation** ⚡
**Triton** is used to optimize the attention computation with:
- **Parallelized operations** to reduce bottlenecks
- **Optimized memory access** to improve efficiency
- **Fused computation** to minimize redundant operations

This is covered in `triton/` and `triton-scratch-pad.ipynb`.

### **3. CUDA Kernel Implementation** 🎯
For an even deeper understanding, **custom CUDA kernels** are written to:
- Gain fine-grained control over memory layout
- Optimize performance beyond what Triton provides
- Experiment with GPU-specific optimizations

This is covered in the `cuda/` directory.

---

## **Benchmarking & Performance Analysis** 📊
The **FlashAttention_Benchmark.ipynb** notebook compares:
- **Naive PyTorch Implementation**
- **Triton Optimized FlashAttention**

Metrics analyzed:
- **Memory usage** 📉
- **Execution time** ⏱️
- **Scalability** across different input sizes 📈

### **Google Colab for Easy Testing** 🖥️
Run the benchmark notebook in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1cIjXMNYWoC4hjuurjtvcxtsUBYPbvTJO?usp=sharing)

---

## **Setup & Installation** 🛠️
### **1. Clone the Repository**
```sh
git clone https://github.com/viai957/Flash-Attention-101
cd flashattention-from-scratch
```

### **2. Install Dependencies**
```sh
pip install torch triton jupyter
```

### **3. Run Jupyter Notebook**
```sh
jupyter notebook
```
Then open the relevant notebook (**flash-attention-notes.ipynb**, **scratch-pad.ipynb**, etc.) to get started.

---

## **Contributing** 🤝
If you'd like to contribute:
1. Fork the repository
2. Create a new branch
3. Submit a pull request

Feel free to open **issues** for discussions or questions!

---

## **Acknowledgements & References** 📚
- **FlashAttention Paper**: [https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **Triton Documentation**: [https://triton-lang.org](https://triton-lang.org)
- **CUDA Programming Guide**: [https://developer.nvidia.com/cuda-zone](https://developer.nvidia.com/cuda-zone)

---

## **License** 📜
This project is licensed under the MIT License. See `LICENSE` for details.

---

### 🚀 **Let's build efficient attention together!** 🔥
