<p align="center">
  <img src="./img/mmtok.png" alt="MMTok" width="360">
</p>

<div align="center">

# Multimodal Coverage Maximization for Efficient Inference of VLMs

[![ICLR 2026](https://img.shields.io/badge/Conference-ICLR%202026-blue)](https://openreview.net/)
[![arXiv](https://img.shields.io/badge/arXiv-2508.18264-red?logo=arxiv)](https://arxiv.org/abs/2508.18264)
[![HF](https://img.shields.io/badge/HuggingFace-Daily%20Paper-yellow?logo=huggingface)](https://huggingface.co/papers/2508.18264)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/Ironieser/MMTok)<br>
[![Project](https://img.shields.io/badge/Project-Homepage-orange?logo=google-chrome)](https://cv.ironieser.cc/projects/mmtok.html)
[![Blog](https://img.shields.io/badge/Blog-Post-green?logo=blogger)](https://cv.ironieser.cc/blog.html?post=mmtok-multimodal-coverage-maximization)
[![Zhihu](https://img.shields.io/badge/Zhihu-专栏-red?logo=zhihu)](https://zhuanlan.zhihu.com/p/1944299907109348831)

**🎉 Accepted to ICLR 2026** &nbsp;|&nbsp; **🚀 Featured in 🤗 Hugging Face Daily Papers**

Welcome to star🌟 this repo or cite✨ the paper if you find it interesting😊
</div>

---

## 🚀 Overview

MMTok is a novel **multimodal approach** for efficient vision-language model (VLM) inference. Unlike existing methods that rely solely on vision or text information, MMTok leverages **both** to select informative vision tokens through **coverage maximization**.

> 💡 **Learn More:** To get an intuitive understanding of MMTok without reading the full paper, check out our **[Project Homepage](https://cv.ironieser.cc/projects/mmtok.html)**, or or read the accessible guide available in **[Blog (English)](https://cv.ironieser.cc/blog.html?post=mmtok-multimodal-coverage-maximization)** and **[知乎专栏 (Chinese Blog)](https://zhuanlan.zhihu.com/p/1944299907109348831)**.

![Performance Results](img/perf.jpg)

## ✨ Key Highlights

- ⚡ **State-of-the-Art Speedup:** **1.87× faster** inference on LLaVA-NeXT-13B (H100) while retaining **98.7%** performance.
- 🎯 **Extreme Compression:** Achieves **87.7% F1** with only **4 vision tokens** on the POPE dataset (LLaVA-1.5-7B).
- 🧠 **Multimodal Coverage:** The first framework to formulate vision token subset selection as a coverage maximization problem, ensuring selected tokens are both semantically relevant to the text query and informationally rich.
- 📈 **Broad Compatibility:** Consistent improvements across multiple VLM architectures (LLaVA-1.5, LLaVA-NeXT, Qwen2.5-VL) and model sizes.



## 📅 Roadmap / TODO
- [ ] **Batch Inference**: Support `batch_size > 1` for high-throughput scenarios.
- [ ] **Video-VLM**: Extend MMTok to video understanding models (e.g., Video-LLaVA).
- [x] **ICLR 2026 Camera-ready**: Final version of the paper and code.
- [x] **Core Framework**: Support for LLaVA-1.5, LLaVA-NeXT, and Qwen2.5-VL.

## 🏗️ Architecture

![MMTok Architecture](img/framework.jpg)

*Our framework selects informative vision tokens by jointly considering both vision and text information.*

**How It Works:**
1. **Multimodal Encoding**: Extract features from both vision and text tokens.
2. **Coverage Computation**: Compute scores measuring how well vision tokens cover text semantics and preserve visual information.
3. **Token Selection**: Select the optimal subset of vision tokens that maximizes multimodal coverage.

## TODO

[] Support Batchsize > 1
[] Support Video-VLM


## 📦 Installation & Usage

Full installation flow (create env → PyTorch → LLaVA → MMTok → optional lmms-eval) is available in **[install.md](install.md)**. Tested on **H100** and **A6000**.

```bash
# Clone and install MMTok
git clone https://github.com/Ironieser/MMTok.git
cd MMTok
pip install -e .
```

```python
# Import mmtok before process_images so the patched version is bound.
from mmtok import mmtok
from llava.mm_utils import process_images  # or: process_images = llava.mm_utils.process_images

# After loading model and tokenizer (e.g., LLaVA)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
model = mmtok(model, language_tokenizer=tokenizer, target_vision_tokens=256)
# Then use process_images(...) and model.generate(...) as with standard LLaVA
```

**Standalone examples:**

```bash
# LLaVA-NeXT
python example/llava_mmtok_example.py

# Qwen2.5-VL
python example/qwen_mmtok_example.py
```

See **[example/llava_mmtok_example.py](example/llava_mmtok_example.py)** and **[example/qwen_mmtok_example.py](example/qwen_mmtok_example.py)** for full end-to-end examples.

**With lmms-eval:** Use the MMTok model adapters for benchmarking. For each, add the script to lmms-eval `models/`, register the model in your eval config, then run eval:
- **LLaVA MMTok:** [example/lmms_eval_llava_mmtok.py](example/lmms_eval_llava_mmtok.py)
- **Qwen MMTok:** [example/lmms_eval_qwen_mmtok.py](example/lmms_eval_qwen_mmtok.py)

Setup details are in **[install.md](install.md)** (optional lmms-eval section).

## 📈 Multi-turn Conversation & Answer Drift with #Tokens

![MMTok Visualization](img/vis.jpg)

## 👥 Authors

- **Sixun Dong** 💼 (Arizona State University)
- **Juhua Hu** (University of Washington)
- **Mian Zhang** 💼 (UT Dallas)
- **Ming Yin** 💼 (Duke University)
- **Yanjie Fu** (Arizona State University)
- **Qi Qian** ✉️ (Zoom Communications) — *Corresponding Author*

*💼 Work done during internship at Zoom.*

## 📚 🌟 Citation

😊 If you find MMTok or this repo useful in your research, please star🌟 and cite✨:

```bibtex
@inproceedings{dong2026mmtok,
  title={{MMT}ok: Multimodal Coverage Maximization for Efficient Inference of {VLM}s},
  author={Sixun Dong and Juhua Hu and Mian Zhang and Ming Yin and Yanjie Fu and Qi Qian},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=GvPdSWZT31}
}
```

## 🙏 Acknowledgments

This project was developed during Sixun Dong's internship at Zoom. We thank **Zoom Communications** for providing internship opportunities, computational resources, and research support. Special thanks to **Yebowen Hu** and **Kaiqiang Song** for their invaluable discussions, feedback, and resource scheduling assistance.

We are grateful to the Hugging Face team for featuring our work in [Daily Papers](https://huggingface.co/papers/2508.18264), and to the ICLR 2026 reviewers for their constructive feedback.

We also acknowledge the foundational open-source projects we built upon: **[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)**, **[LLaVA](https://github.com/haotian-liu/LLaVA)**, **[Qwen-VL](https://github.com/QwenLM/Qwen3-VL)**, **[DivPrune](https://github.com/vbdi/divprune)**, and **[VisionZip](https://github.com/dvlab-research/VisionZip)**.

---

<div align="center">

[⭐ Star us on GitHub](https://github.com/Ironieser/MMTok) &nbsp;|&nbsp; [📄 Read the Paper](https://arxiv.org/abs/2508.18264) &nbsp;|&nbsp; [🏠 Homepage](https://cv.ironieser.cc/projects/mmtok.html) &nbsp;|&nbsp; [📝 Blog](https://cv.ironieser.cc/blog.html?post=mmtok-multimodal-coverage-maximization) &nbsp;|&nbsp; [📰 知乎专栏](https://zhuanlan.zhihu.com/p/1944299907109348831)

</div>
