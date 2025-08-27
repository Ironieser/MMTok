<!-- HERO (centered) -->

<!-- HERO (left-aligned) -->
<p align="center">
  <img src="./img/logo.png" alt="MMTok Logo" width="60">
  <img src="./img/mmtok_title.png" alt="MMTok Title" width="200">
</p>

# **Multimodal Coverage Maximization for Efficient Inference of VLMs**

<p align="center">
  <a href="https://arxiv.org/abs/2508.18264">
    <img src="https://img.shields.io/badge/Paper-arXiv%2025'08-orange?logo=arxiv&logoColor=white" alt="Paper"/>
  </a>
  <a href="https://github.com/Ironieser/MMTok">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code"/>
  </a>
  <a href="https://project.ironieser.cc/mmtok">
    <img src="https://img.shields.io/badge/Homepage-Live-blue?logo=google-chrome&logoColor=white" alt="Homepage"/>
  </a>
</p>



## 🚀 Overview

MMTok is a novel multimodal approach for efficient vision-language model (VLM) inference. By leveraging both vision and text tokens to select informative vision tokens through coverage maximization, our method achieves significant speedup while maintaining high performance.

## ✨ Key Features

- **1.87× Speedup** on H100 GPU (higher on other GPUs)
- **95%+ Performance Retained** at highest prune ratio on LLaVA-1.5 & Next (7B & 13B)
- **87.7% F1** with only **4 tokens** on POPE dataset
- **Multimodal Coverage Maximization** - leverages both vision and text information

## 🎯 Method

Our key insight is that **multimodal information is complementary for vision token selection**. We formulate the subset selection problem as a **coverage maximum problem**, where a subset of vision tokens is optimized to cover both text tokens and the original set of vision tokens simultaneously. This approach leverages both vision and text information to select the most informative vision tokens for efficient VLM inference.

## 📊 Results

| Model | Dataset | Speedup | Performance Retained |
|-------|---------|---------|---------------------|
| LLaVA-Next-13B | POPE | 1.87× | 98.7% |
| LLaVA-1.5-7B | POPE | - | 87.7% (4 tokens) |

## 🏗️ Architecture

![MMTok Architecture](img/mmtok.jpg)

Our multimodal framework combines vision and text information for efficient token selection.

## 📈 Performance Comparison

![Performance Results](img/combined_plots.png)

Comprehensive comparison with state-of-the-art methods across multiple models and datasets.

## 🚧 Coming Soon

- [x] Paper on arXiv
- [ ] Code implementation
- [ ] Detailed blog post
- [ ] Chinese technical blog (知乎)

## 👥 Authors

- **Sixun Dong*** (Arizona State University)
- **Juhua Hu** (University of Washington)
- **Mian Zhang*** (UT Dallas)
- **Ming Yin*** (Duke University)
- **Yanjie Fu** (Arizona State University)
- **Qi Qian** (Zoom Communications) - Corresponding Author
  
*Work done during internship at Zoom.
## 📚 Citation

```bibtex
@misc{dong2025mmtok,
      title={MMTok: Multimodal Coverage Maximization for Efficient Inference of VLMs}, 
      author={Sixun Dong and Juhua Hu and Mian Zhang and Ming Yin and Yanjie Fu and Qi Qian},
      year={2025},
      eprint={2508.18264},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2508.18264}, 
}
```

## 🏠 Homepage

Visit our project homepage: **[project.ironieser.cc/mmtok](https://project.ironieser.cc/mmtok)**

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.


## 🙏 Acknowledgments

We thank Zoom Communications for providing internship opportunities and research support. Special thanks go to our collaborators for their constructive feedback and support. In particular, Yebowen Hu offered valuable discussions and feedback, while Kaiqiang Song contributed many insightful discussions, extensive assistance with computational resource scheduling, and helpful exchanges that enriched our learning. We also acknowledge the support from Zoomies.

---

**MMTok: Efficient Vision Token Pruning for VLMs** 🎯
