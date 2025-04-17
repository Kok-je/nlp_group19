# 🎉 Citation Intent Classification with Model Distillation 🎉

**✨ Shrinking Giants & Making Citations Smarter! ✨**

Ever wondered if a tiny model could outsmart a giant LLM? **We proved it can!** 🚀 This project distills the power of massive language models (like GPT-4 and LLaMA3) into a lightweight **T5-base model** (223M params) to classify citation intent—because research should be **fast, efficient, and fun**!

## 🔥 Why You'll Love This

- **⚡ Lightning-Fast**: Ditch the 70B-parameter monsters—our distilled model is small but mighty!
- **🎯 Accurate AF**: Hits ~0.7 F1 on the [SciCite dataset](https://github.com/allenai/scicite) (just like the big guys).
- **🤯 Surprise Finding**: Adding *reasoning* during distillation **didn't help** (plot twist! 🍿).
- **📦 Plug-and-Play**: Easy-to-run code for training, testing, and playing with citation magic.

---
## 📊 What's Inside?

```plaintext
├── Cleaned Work/     # Main project work directory
│   ├── utils/                    # Utility functions and helpers, e.g. Charts, Data Augmentor, Model Cards, Post and Preprocessing
│   ├── teacher rationale generation/  # Teacher model rationale generation
│   ├── student training data/    # Data for student model training
│   ├── results (student, teacher outputs)/  # Model outputs and results
│   ├── phase1_v2/               # Phase 1 implementation
│   ├── data/                    # Dataset and processed data
│   ├── titan-job.slurm         # Slurm job script for Titan
│   ├── a100-job.slurm          # Slurm job script for A100
│   └── requirements.txt         # Python dependencies
├── Archive/          # Archived files and previous work
├── pyproject.toml    # Python project configuration
├── poetry.lock       # Poetry dependency lock file
├── .gitignore        # Git ignore rules
└── README.md         # You're here! 👋
```

## 🔍 Key Findings
✅ **Off-the-shelf LLMs perform well**  
   - Achieved F1 ~0.7 on SciCite dataset  
   - Model size ≠ performance (e.g., LLaMA3 1B outperformed Mistral 7B)  

✅ **Distilled model succeeds**  
   - T5-base (223M params) matched giant LLMs (70B+ params)  
   - Dramatically reduced compute requirements  

❌ **Rationales didn't help** *(Surprise!)*  
   - Unlike [prior work](https://arxiv.org/abs/2305.02301), reasoning distillation:  
     - Didn't improve SciCite performance  
     - May not work for nuanced, domain-specific tasks  

## 👏 Credits
Team Superstars: Cheng Jin Hao, Chung Yunseong, Ethan Wei, Evelyn Lai, Kok Joon Eu Shawn, Nikhil Sultania
Mentor: Zhang Bowen

**📚 Key Reference**:  
Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., et al. (2023). *"Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes"*.  
[arXiv:2305.02301](https://arxiv.org/abs/2305.02301)