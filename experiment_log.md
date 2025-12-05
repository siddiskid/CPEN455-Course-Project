# CPEN455 Project Experiment Log

## Project Overview
- **Task**: Spam detection using LLMs (SmolLM2-135M-Instruct)
- **Method**: Bayesian Inverse Classification
- **Start Date**: December 5, 2025
- **Platform**: Lightning.ai Studios

---

## Phase 1: Setup & Environment Verification

### Step 1: Environment Setup

**Date**: December 5, 2025

**Status**: COMPLETE

**Commands Run**:
```bash
cd /teamspace/studios/this_studio/CPEN455-Course-Project
git clone https://github.com/DSL-Lab/CPEN455-Project-2025W1-Autograder.git autograder
uv sync
```

**Environment Details**:
- Python Version: CPython 3.13.7
- Virtual Environment: `.venv` created in project root
- Package Manager: UV

**Packages Installed** (67 total, key ones):
- torch==2.8.0+cu128
- huggingface-hub==0.35.3
- wandb==0.22.2
- pandas==2.3.3
- numpy==2.3.3
- einops==0.8.1
- safetensors==0.6.2
- bidict==0.23.1

**CUDA/GPU Verification**:
```python
>>> torch.cuda.is_available()
True
>>> torch.cuda.get_device_name(0)
'Tesla T4'
```

**Datasets Verified**:
| File | Size | Description |
|------|------|-------------|
| `train_val_subset.csv` | 27,871 bytes | 20 labeled samples for training/validation |
| `test_subset.csv` | 1,982,181 bytes | ~1000 samples for grading (labels hidden) |
| `test_subset_random_labels.csv` | 1,981,181 bytes | Same as above but with random labels for local testing |

**Model Configuration** (from `.env`):
```
MODEL_CACHE_DIR=./cache/huggingface/transformers
MODEL_CHECKPOINT=HuggingFaceTB/SmolLM2-135M-Instruct
PROJECT_NAME=CPEN455-Project-2025W1
```

**Model Download**:
- Files downloaded: `config.json`, `tokenizer.json`, `tokenizer_config.json`, `model.safetensors` (269MB)
- Cache location: `./cache/huggingface/transformers/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/12fd25f77366fa6b3b4b768ec3050bf629380bac/`

**Model Architecture** (from config):
- **Model**: SmolLM2-135M-Instruct (LlamaForCausalLM)
- **Type**: Decoder-only transformer
- **Parameters**: 162,826,560 (~163M)
- **Layers**: 30 (`num_hidden_layers`)
- **Hidden Size**: 576 (`hidden_size`)
- **Attention Heads**: 9 (`num_attention_heads`)
- **Key-Value Heads**: 3 (`num_key_value_heads`) - uses Grouped Query Attention (GQA)
- **Head Dimension**: 576 / 9 = 64
- **Weight Tying**: Yes (`lm_head.weight` created from `embed_tokens.weight`)

**Notes**:
- Used HTTPS for cloning autograder (worked on first try)
- GPU Tesla T4 available with CUDA 12.8

---

## Phase 2: Baseline Experiments

### Step 2: Chatbot Example

**Date**: December 5, 2025

**Status**: COMPLETE

**Command**: `uv run -m examples.chatbot_example`

**Device Used**: CUDA (Tesla T4)

**Model Loading Logs**:
```
Using device: cuda
Found 1 snapshots in cache
Loading model weights from: ./cache/.../model.safetensors
Creating lm_head.weight from embed_tokens.weight
Missing keys: []
Unexpected keys: []
```

**Chat Template Format**:
```
<|im_start|>system
You are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
```

**Generation Parameters** (from `chatbot_example.py`):
- `max_new_tokens`: 500
- `temperature`: 0.2
- `top_p`: 0.9
- `do_sample`: True

**Prompts Tested**:

| # | Prompt | Model Response Summary | Notes |
|---|--------|------------------------|-------|
| 1 | "What is gravity?" (default) | Explained gravity as fundamental force, Einstein's general relativity, spacetime curvature, gravitational fields. Mentioned Earth's surface gravity. | Coherent, informative, ~200 words |
| 2 | "How do I write a Python function to sort a list?" | Provided complete bubble sort implementation with code, example usage, and output demonstration | Good code generation, correct syntax |
| 3 | "What is the difference between spam and ham emails?" | Explained spam as "sent without indication of sender", ham as "sent with clear intention". Mentioned spam filters, phishing. Some terminology confusion ("Hammels") | Understands concepts but imprecise language |
| 4 | "Explain machine learning in one sentence." | (Part of batch test) | Tested for conciseness |

**Full Response for Gravity Prompt**:
```
Gravity is a fundamental force of nature that attracts objects with mass towards 
each other. It is a result of the interaction between mass, energy, and space 
itself. According to Einstein's theory of general relativity, gravity is not a 
force that acts between objects, but rather a curvature of spacetime caused by 
the presence of mass and energy.

Imagine spacetime as a fabric that surrounds all objects, including light and 
matter. Massive objects warp the fabric of spacetime, creating a gravitational 
field that pulls objects towards them. This field is what we experience as gravity.

The strength of gravity depends on the mass of the object and the distance between
it and the center of the Earth. The stronger the gravitational field, the stronger
the pull of gravity. On Earth, the gravitational pull is strongest at the surface,
where objects are closer together.

Gravity is a universal force that acts across the entire universe, from the 
smallest subatomic particles to the largest objects in the cosmos. It is a 
fundamental aspect of the universe, and its effects are felt everywhere in 
the universe.
```

**Full Response for Spam/Ham Question**:
```
The main difference between spam and ham emails is the intent behind the message.

Spam emails are sent to a specific group of recipients without any indication of 
who sent them or what they are trying to achieve. They often contain irrelevant 
or misleading information, such as spam filters, spam emails, or spam messages 
from unknown sources. Spam emails are often used to trick recipients into opening 
a malicious email, such as a phishing email, or to send unsolicited messages.

Hammels, on the other hand, are emails that are sent to a specific group of 
recipients, often with a clear intention of sending a message to them. Hammels 
are often used to send unsolicited messages, such as messages from unknown 
sources, or to send messages to specific groups of people. Hammels are often 
used to send messages that are intended to be personal, confidential, or to 
promote a particular cause.
```

**Observations**:
- Model generates coherent, well-structured responses
- Good general knowledge about physics, programming, and email concepts
- Uses chat template format with special tokens (`<|im_start|>`, `<|im_end|>`)
- Some terminology issues (invented word "Hammels" for ham emails)
- Generation is deterministic-ish with low temperature (0.2)
- Model understands spam detection concept but explanation is somewhat confused
- This explains why direct prompting for classification may be unreliable

---

### Step 3: Bayes Inverse Zero-Shot

**Date**: December 5, 2025

**Status**: COMPLETE

**Command**: `bash examples/bayes_inverse_zero_shot.sh`

**Script Contents**:
```bash
uv run -m examples.bayes_inverse \
--method zero_shot \
--max_seq_len 512 \
--batch_size 1
```

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Method | `zero_shot` |
| Max Sequence Length | 512 |
| Batch Size | 1 |
| User Prompt | (none/empty) |
| Learning Rate | N/A (no training) |
| Num Iterations | 1 (evaluation only) |

**Dataset Statistics**:
```python
# From train_val_subset.csv
Spam/Ham
0    10  # Ham
1    10  # Spam
Total: 20 samples (perfectly balanced)
```

**WandB Run**: `bayes-inverse-zero_shot_msl512`

**Output Files Generated**:
- `bayes_inverse_probs/train_n_val_dataset_probs.csv` (923 bytes)
- `bayes_inverse_probs/test_dataset_probs.csv` (45,237 bytes)

**Results**:
| Metric | Value |
|--------|-------|
| Training Accuracy | N/A (no training) |
| Validation Accuracy | **50.00%** |
| Test Accuracy (random labels) | 49.30% |

**Prediction Analysis**:
```python
# All 20 samples predicted as spam
Predictions distribution:
pred
1    20  # All spam
Name: count, dtype: int64

All predicting spam: True
```

**Sample Predictions** (from `train_n_val_dataset_probs.csv`):
| data_index | prob_ham | prob_spam | Prediction |
|------------|----------|-----------|------------|
| 8403 | 0.000253 | 0.999747 | Spam |
| 3377 | 2.01e-06 | 0.999998 | Spam |
| 9734 | 3.99e-09 | 1.000000 | Spam |
| 13253 | 0.000148 | 0.999852 | Spam |
| 12946 | 0.000113 | 0.999887 | Spam |
| 27081 | 1.85e-06 | 0.999998 | Spam |
| 20291 | 5.70e-07 | 0.999999 | Spam |
| 2902 | 5.54e-08 | 1.000000 | Spam |

**Key Observation - Extreme Spam Bias**:
The model assigns near-100% spam probability to ALL emails:
- Minimum spam prob: ~0.9997 (99.97%)
- Maximum spam prob: 1.0 (100%)
- Mean spam prob: ~0.9999

**Confusion Matrix** (on train_val set):
|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actual Ham** | 0 | 10 |
| **Actual Spam** | 0 | 10 |

- True Positives (Spam→Spam): 10
- False Positives (Ham→Spam): 10
- True Negatives (Ham→Ham): 0
- False Negatives (Spam→Ham): 0
- **Ham Recall: 0%** (missed all ham)
- **Spam Recall: 100%** (caught all spam)

**Why This Happens - Technical Analysis**:
The Bayesian inverse method computes:
$$P(label|email) = \frac{P(email|label)}{\sum_{labels} P(email|label)}$$

For zero-shot:
1. The prompt template is: `"The following email is labeled as {label}.\nSubject: {subject}\nMessage: {message}"`
2. Model computes log probability of this entire sequence for label="spam" and label="ham"
3. Since "spam" is likely a more common word/concept in pre-training data, P(sequence|spam) >> P(sequence|ham)
4. This creates extreme bias toward spam predictions

**Prompt Template Used** (from `utils/prompt_template.py`):
```python
PROMPT_TEMPLATE = (
    "{user_prompt}\n"
    "The following email is labeled as {label}.\n"
    "Subject: {subject}\n"
    "Message: {message}"
)
```

**Analysis**:
- Zero-shot baseline is completely useless for classification
- Model has strong prior bias that email content is more likely under "spam" label
- This establishes the worst-case baseline that training must improve upon
- 50% accuracy = random guessing (since dataset is balanced)

---

### Step 4: Bayes Inverse Naive Prompting

**Date**: December 5, 2025

**Status**: COMPLETE

**Command**: `bash examples/bayes_inverse_naive_prompting.sh`

**Script Contents**:
```bash
uv run -m examples.bayes_inverse \
--method naive_prompting \
--max_seq_len 512 \
--batch_size 1 \
--user_prompt "You are an email classifier for a corporate dataset.
Output exactly one lowercase token: spam or ham.
Ham: direct, personal/operational notes (replies, questions, scheduling, task-specific), short and focused, addressed to a specific person/team.
Spam if either:
A) Automated/system notice: repetitive subject/body; log/error text (e.g., 'general sql error', 'parsing file', 'start date : … ; hourahead hour : … ;').
B) Impersonal memo: very long, broad distribution/forward chain; corporate/legal/financial content; contains a long boilerplate legal disclaimer/footer.
Rule: If any spam cue matches → spam; else → ham.
Return only the label with no explanation."
```

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Method | `naive_prompting` |
| Max Sequence Length | 512 |
| Batch Size | 1 |
| User Prompt | Detailed classification instructions (see above) |
| Learning Rate | N/A (no training) |
| Num Iterations | 1 (evaluation only) |

**WandB Run**: `bayes-inverse-naive_prompting_msl512`
- Run ID: `yny7fhvf`
- Link: https://wandb.ai/siddarth_ch-university-of-british-columbia/CPEN455-Project-2025W1/runs/yny7fhvf

**Results**:
| Metric | Value |
|--------|-------|
| Training Accuracy | N/A (no training) |
| Validation Accuracy | **50.00%** |
| Test Accuracy (random labels) | ~50% expected |

**Key Observation - Complete Uncertainty (0.5/0.5)**:
The model assigns **exactly 0.5 probability to BOTH classes** for ALL emails:

**Sample Predictions** (ALL identical):
| data_index | prob_ham | prob_spam | Prediction |
|------------|----------|-----------|------------|
| 8403 | 0.5 | 0.5 | Ham (tie→0) |
| 3377 | 0.5 | 0.5 | Ham (tie→0) |
| 9734 | 0.5 | 0.5 | Ham (tie→0) |
| 13253 | 0.5 | 0.5 | Ham (tie→0) |
| ... | 0.5 | 0.5 | Ham (tie→0) |

**Prediction Distribution**:
```python
pred
0    20  # All ham (due to tie-breaking)
Name: count, dtype: int64

All 0.5/0.5 ties: True
```

**Confusion Matrix** (on train_val set):
|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actual Ham** | 10 | 0 |
| **Actual Spam** | 10 | 0 |

- True Positives (Spam→Spam): 0
- False Positives (Ham→Spam): 0
- True Negatives (Ham→Ham): 10
- False Negatives (Spam→Ham): 10
- **Ham Recall: 100%** (caught all ham)
- **Spam Recall: 0%** (missed all spam)

**Why This Happens - Technical Analysis**:
1. The detailed prompt is prepended to both ham and spam versions
2. With the long, complex prompt, the model's log probabilities become nearly identical
3. P(sequence|"spam") ≈ P(sequence|"ham") → softmax gives exactly 0.5/0.5
4. When probabilities are tied, `argmax` returns index 0 (ham)

**Comparison: Zero-Shot vs Naive Prompting**:
| Metric | Zero-Shot | Naive Prompting |
|--------|-----------|-----------------|
| Val Accuracy | 50% | 50% |
| Prediction Bias | All Spam | All Ham |
| prob_spam range | 0.9997-1.0 | 0.5 (exactly) |
| prob_ham range | 0-0.0003 | 0.5 (exactly) |
| Ham Recall | 0% | 100% |
| Spam Recall | 100% | 0% |
| Behavior | Extreme spam bias | Complete uncertainty |

**Analysis**:
- The detailed prompt **neutralizes** the model's prediction capability
- Instead of helping classification, it creates maximum entropy (uncertainty)
- This is the **opposite problem** of zero-shot: instead of overconfident wrong, it's uncertain
- The prompt may be too complex or contradictory for a 135M parameter model
- Both baselines achieve 50% accuracy but with completely opposite error patterns
- Neither baseline is useful - we need fine-tuning

---

### Step 5: Bayes Inverse Full Fine-tune

**Date**: December 5, 2025

**Status**: COMPLETE

**Command**: `bash examples/bayes_inverse_full_finetune.sh`

**Script Contents**:
```bash
# This config spends about 3 minutes to run on a Macmini with M4 Chip, 16GB RAM
uv run -m examples.bayes_inverse \
--method full_finetune \
--max_seq_len 256 \
--batch_size 8 \
--num_iterations 80
```

**Configuration**:
| Parameter | Value |
|-----------|-------|
| Method | `full_finetune` |
| Max Sequence Length | 256 |
| Batch Size | 8 |
| Num Iterations | 80 |
| Learning Rate | 1e-5 (default) |
| Optimizer | AdamW |
| Device | CUDA (Tesla T4) |

**WandB Run**: `bayes-inverse-full_finetune_msl256_ni80_bs8`
- Run ID: `y42nnfpy`
- Link: https://wandb.ai/siddarth_ch-university-of-british-columbia/CPEN455-Project-2025W1/runs/y42nnfpy

**Training Time**: ~34 seconds (80 iterations at 2.32 it/s on Tesla T4)

**Results**:
| Metric | Value |
|--------|-------|
| Validation Accuracy | **85.00%** |
| Test Accuracy (random labels) | 50.40% (expected ~50% for random) |
| Training Time | 34 seconds |

**Confusion Matrix** (on train_val set, 20 samples):
|  | Predicted Ham | Predicted Spam |
|--|---------------|----------------|
| **Actual Ham** | 8 | 2 |
| **Actual Spam** | 1 | 9 |

- True Positives (Spam→Spam): 9
- False Positives (Ham→Spam): 2
- True Negatives (Ham→Ham): 8
- False Negatives (Spam→Ham): 1
- **Ham Recall: 80%** (8/10)
- **Spam Recall: 90%** (9/10)
- **Precision (Spam): 81.8%** (9/11)
- **Precision (Ham): 88.9%** (8/9)

**Prediction Distribution**:
```python
pred
1    11  # Spam
0     9  # Ham
Name: count, dtype: int64

# Test set distribution
pred
1    566  # Spam (56.6%)
0    434  # Ham (43.4%)
```

**Probability Statistics** (after fine-tuning):
| Metric | prob_ham | prob_spam |
|--------|----------|-----------|
| Min | 0.000000 | 0.000000 |
| Max | 1.000000 | 1.000000 |
| Mean | 0.4315 | 0.5685 |

**Sample Predictions** (showing confident discrimination):
| data_index | prob_ham | prob_spam | Actual | Pred | Correct |
|------------|----------|-----------|--------|------|---------|
| 8403 | 1.0000 | 2.6e-08 | Ham | Ham | Yes |
| 3377 | 0.9977 | 0.0023 | Ham | Ham | Yes |
| 9734 | 1.1e-09 | 1.0000 | Spam | Spam | Yes |
| 13253 | 0.0648 | 0.9352 | Ham | Spam | **No** |
| 12946 | 0.9993 | 0.0007 | Ham | Ham | Yes |
| 27081 | 1.3e-07 | 1.0000 | Spam | Spam | Yes |
| 16282 | 0.5654 | 0.4346 | Spam | Ham | **No** |
| 13384 | 0.0010 | 0.9990 | Ham | Spam | **No** |

**Misclassified Samples Analysis**:
| Index | Actual | Predicted | prob_ham | prob_spam | Notes |
|-------|--------|-----------|----------|-----------|-------|
| 13253 | Ham | Spam | 0.0648 | 0.9352 | High confidence wrong |
| 16282 | Spam | Ham | 0.5654 | 0.4346 | Low confidence, borderline |
| 13384 | Ham | Spam | 0.0010 | 0.9990 | Very high confidence wrong |

**Key Observations**:
1. **Massive improvement**: 50% → 85% accuracy (+35 percentage points)
2. **Model now discriminates**: Probabilities span full 0-1 range (vs all 0.5 or all 0.99+)
3. **Confident predictions**: Most predictions are near 0 or 1 probability
4. **Balanced errors**: 2 false positives, 1 false negative (no severe bias)
5. **Fast training**: Only 34 seconds on GPU vs expected 3 min on Mac M4

**Comparison Across All Methods**:
| Method | Val Accuracy | Ham Recall | Spam Recall | Prediction Bias |
|--------|--------------|------------|-------------|-----------------|
| Zero-Shot | 50% | 0% | 100% | All Spam |
| Naive Prompting | 50% | 100% | 0% | All Ham (0.5/0.5 tie) |
| **Full Fine-tune** | **85%** | **80%** | **90%** | **Balanced** |

**Analysis**:
- Fine-tuning is essential for this task - baselines are useless
- The model learns to distinguish spam patterns from ham patterns
- 3 errors out of 20 samples (85% accuracy)
- Test set predicts 56.6% spam, suggesting reasonable generalization
- Ready to submit to Kaggle to see real test accuracy

---

## Phase 3: Improvements

### Kaggle Submission #1 - Full Fine-tune (80 iterations)

**Date**: December 5, 2025

**Status**: COMPLETE - SUCCESS!

**Model Used**: `examples/ckpts/final_model.ckpt`
- Training: Full fine-tune, 80 iterations
- Batch size: 8
- Max seq length: 256
- Learning rate: 1e-5

**Validation Accuracy**: 95.00% (19/20 correct)

**Kaggle Results**:
| Metric | Value |
|--------|-------|
| **Public Leaderboard Score** | **91.01%** |
| Submission File | `kaggle_submission.csv` |

**Prediction Distribution**:
```
Spam: 641 (64.1%)
Ham: 359 (35.9%)
```

**Thresholds Achieved**:
- [x] 80% accuracy (5% of grade) - PASSED
- [x] 85% accuracy (5% of grade) - PASSED
- [x] 90%+ accuracy - EXCEEDED EXPECTATIONS!

**Analysis**:
- Model generalizes extremely well from 20 training samples to 1000 test samples
- Validation accuracy (95%) closely predicts test accuracy (91%)
- The slight drop (95% → 91%) is expected generalization gap
- 91% is an excellent result for a 135M parameter model with minimal training data

---

## Phase 4: Final Results
(To be filled)

---

## Appendix: Technical Details

### Bayesian Inverse Classification Formula

$$
P(Y_{label}|X_{\leq i}) = \frac{P(X_{\leq i}, Y_{label})}{\sum_{Y'} P(X_{\leq i}, Y')}
$$

Where:
- $X_{\leq i}$: Input sequence (email content)
- $Y_{label}$: Label (spam=1 or ham=0)
- The model computes joint log probabilities and normalizes via softmax

**Implementation** (from `bayes_inverse.py`):
```python
# Compute log prob for both labels
prompts_ham = [get_prompt(..., label="ham") for ...]
prompts_spam = [get_prompt(..., label="spam") for ...]
prompts = prompts_ham + prompts_spam

seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)
seq_log_prob = rearrange(seq_log_prob, '(c b) -> b c', c=2)  # [batch, 2]
probs = F.softmax(seq_log_prob, dim=-1)  # Normalize to probabilities
labels_pred = torch.argmax(probs, dim=-1)  # 0=ham, 1=spam
```

### Label Mapping
```python
ENRON_LABEL_INDEX_MAP = bidict({
    'ham': 0,
    'spam': 1
})
```

### Prompt Template
```python
PROMPT_TEMPLATE = (
    "{user_prompt}\n"
    "The following email is labeled as {label}.\n"
    "Subject: {subject}\n"
    "Message: {message}"
)
```

### Model Architecture Summary
| Component | Details |
|-----------|---------|
| Type | Decoder-only Transformer (LLaMA-style) |
| Embedding | 49152 vocab → 576 dim |
| Layers | 30 × LlamaDecoderLayer |
| Attention | Grouped Query Attention (9 Q heads, 3 KV heads) |
| MLP | SwiGLU (576 → 1536 → 576) |
| Norm | RMSNorm (eps=1e-5) |
| Position | RoPE (theta=100000) |
| Output | Linear (576 → 49152) tied with embedding |

---

## Key Findings Summary
(To be compiled at the end)
