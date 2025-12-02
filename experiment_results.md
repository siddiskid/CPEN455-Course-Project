# Experiment Results and Observations

## Phase 1: Basic Examples

### 1.1 Chatbot Example (`examples/chatbot_example.py`)

**Date**: December 2, 2025  
**Model**: SmolLM2-135M-Instruct  
**Device**: MPS (Apple Silicon)  
**Command**: `uv run -m examples.chatbot_example`

---

#### Test 1: Factual Question - "What is gravity?"

**Prompt**:
```
What is gravity?
```

**Response**:
```
Gravity is a fundamental force of nature that attracts objects with mass towards each other. It is a result of the interaction between mass, energy, and space itself. In the context of our universe, gravity is a result of the curvature of spacetime caused by the presence of mass and energy.

Imagine spacetime as a trampoline. When you place a heavy object, like a bowling ball, on the trampoline, it creates a depression in the surface. This depression is caused by the object's mass and the energy it contains. The more massive the object, the larger the depression it creates.

Now, when you move an object, it follows the curvature of spacetime. The more massive the object, the greater the curvature of spacetime around it. This curvature creates a force that pulls the object towards the center of the mass, causing it to fall towards the center of the mass.

Gravity is a universal force that acts between all objects, regardless of their mass or energy content. It is the reason why objects fall towards the ground, and why planets orbit around stars. It is also the reason why objects with mass can move at a constant speed, regardless of their initial velocity.

In essence, gravity is a manifestation of the fundamental nature of spacetime, where mass and energy create a gravitational field that attracts other objects.
```

**Assessment**: Excellent - coherent, detailed explanation of spacetime curvature with good use of analogy (trampoline).

---

#### Test 2: Creative Writing - "Write a haiku about machine learning"

**Prompt**:
```
Write a haiku about machine learning.
```

**Response**:
```
In the realm of data analysis, a machine learning model's beauty lies not in its ability to predict outcomes, but in its ability to learn from experience, adapting to unseen patterns, and iteratively refining its decision-making. The process of machine learning is akin to a dance, with each step bringing new insights and possibilities, but with the added benefit of a never-ending supply of data to guide and inform the model's growth.
```

**Assessment**: Poor - completely ignored the haiku format (5-7-5 syllable structure). Wrote prose instead. The model struggles with structured creative formats.

---

#### Test 3: Spam Detection - Obvious Spam Email

**Prompt**:
```
Is this email spam or ham? Subject: Congratulations! You won $1,000,000! Message: Click here to claim your prize now!
```

**Response**:
```
This email is spam. It is a fake email sent to a specific recipient, and it is likely from a scammer or a malicious entity. The subject line is "Congratulations! You won $1,000,000! Message: Click here to claim your prize now!"
```

**Assessment**: Correct - correctly identified the classic "$1M prize" phishing pattern as spam with appropriate reasoning.

---

#### Test 4: Spam Detection - Legitimate Email (Ham)

**Prompt**:
```
Is this email spam or ham? Subject: Meeting tomorrow at 3pm. Message: Hi John, just a reminder about our project meeting tomorrow at 3pm in conference room B. Please bring your progress report. Thanks, Sarah
```

**Response**:
```
This email is likely spam. It's a common practice to send emails to people who are not familiar with the company or the company's policies. It's also a common practice to send emails to people who are not familiar with the company's policies, which can lead to confusion and miscommunication.

It's also a common practice to send emails to people who are not familiar with the company's policies, which can lead to confusion and miscommunication. This can happen when the email is sent to a person who is not familiar with the company's policies, or when the email is sent to a person who is not familiar with the company's policies.

In this case, the email is likely spam because it's not from a person who...
```

**Assessment**: INCORRECT - falsely classified a normal workplace meeting reminder as spam. The model shows a bias toward classifying emails as spam, and the reasoning is incoherent/repetitive.

---

#### Test 5: Code Generation - Factorial Function

**Prompt**:
```
Write a Python function that calculates the factorial of a number.
```

**Response**:
```python
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n == 0 or n == 1:
        return 1
    else:
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
```

With usage example:
```python
print(factorial(5))  # Output: 120
```

**Assessment**: Good - produced correct, working Python code with proper error handling for negative numbers and base cases.

---

### Summary of Chatbot Behavior

| Test | Category | Result | Notes |
|------|----------|--------|-------|
| 1 | Factual/Scientific | Pass | Excellent explanation with analogy |
| 2 | Creative (structured) | Fail | Ignored haiku format completely |
| 3 | Classification (obvious) | Pass | Correct spam detection |
| 4 | Classification (subtle) | Fail | False positive - ham classified as spam |
| 5 | Code Generation | Pass | Correct and well-structured code |

**Key Observations**:

1. **Model Size Limitations**: SmolLM2-135M-Instruct is a small 135M parameter model. It handles factual questions and code well but struggles with structured creative tasks.

2. **Spam Detection Bias**: The model has a strong bias toward classifying emails as spam. This is problematic for the Enron dataset where many legitimate business emails need to be correctly identified as ham.

3. **Repetitive Text Generation**: When uncertain, the model tends to produce repetitive, circular text (as seen in Test 4).

4. **Zero-Shot Classification Limitation**: Direct prompting for spam/ham classification is unreliable, which motivates the Bayesian inverse approach used in this project.

**Implications for Project**:
- The 49% baseline accuracy (random chance) makes sense given the spam bias observed
- Fine-tuning or better prompting strategies are necessary to improve classification
- The Bayesian inverse method (computing P(label|email) via P(email|label)) may help overcome the direct classification bias

---

## 1.2 Bayes Inverse Zero Shot

**Date**: December 2, 2025  
**Command**: `bash examples/bayes_inverse_zero_shot.sh`  
**Configuration**:
- `--method zero_shot`
- `--max_seq_len 512`
- `--batch_size 1`

**WandB Run**: `bayes-inverse-zero_shot_msl512`

---

### How Bayesian Inverse Classification Works

Instead of directly asking "Is this email spam or ham?", the Bayesian inverse method computes:

$$P(Y_{label}|X) = \frac{P(X|Y_{label}) \cdot P(Y_{label})}{\sum_{Y'} P(X|Y') \cdot P(Y')}$$

In practice:
1. Create two prompts: one assuming the email is "ham", one assuming "spam"
2. Compute the log-probability of each prompt under the LLM
3. Apply softmax to get P(ham|email) and P(spam|email)
4. Classify based on which probability is higher

---

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 50.00% |
| **Spam Recall** | 100.00% |
| **Ham Recall** | 0.00% |
| **Predicted Spam** | 20/20 (100%) |
| **Predicted Ham** | 0/20 (0%) |

---

### Probability Distribution Analysis

Sample predictions from `bayes_inverse_probs/train_n_val_dataset_probs.csv`:

| data_index | prob_ham | prob_spam | Prediction |
|------------|----------|-----------|------------|
| 8403 | 0.000253 | 0.999747 | spam |
| 3377 | 0.000002 | 0.999998 | spam |
| 9734 | 0.000000 | 1.000000 | spam |
| 29472 | 0.001760 | 0.998240 | spam |

**Observation**: The model assigns near-zero probability to ham for almost every email. Even the highest ham probability observed (~0.17%) is still strongly classified as spam.

---

### Key Findings

1. **Extreme Spam Bias**: The zero-shot model classifies 100% of emails as spam, regardless of content.

2. **Probability Collapse**: The Bayesian probabilities are extremely skewed (prob_spam often = 1.0 to machine precision).

3. **50% Accuracy Explained**: Since the training set is balanced (50% spam, 50% ham), predicting everything as spam yields exactly 50% accuracy.

4. **Root Cause**: The LLM has learned that the sequence "The following email is labeled as spam..." is more probable than "...labeled as ham" regardless of the email content. This is likely because:
   - The model's training data may have more examples of spam-labeled content
   - The prompt template doesn't give enough context for the model to distinguish

---

### Prompt Template Used

From `utils/prompt_template.py`:
```python
PROMPT_TEMPLATE = (
    "{user_prompt}\n"
    "The following email is labeled as {label}.\n"
    "Subject: {subject}\n"
    "Message: {message}"
)
```

For zero-shot, `user_prompt` is empty, so prompts look like:
```
The following email is labeled as ham.
Subject: Meeting tomorrow
Message: Hi John, reminder about our 3pm meeting...
```

vs.

```
The following email is labeled as spam.
Subject: Meeting tomorrow
Message: Hi John, reminder about our 3pm meeting...
```

The model consistently finds the "spam" version more probable.

---

## 1.3 Bayes Inverse Naive Prompting

**Date**: December 2, 2025  
**Command**: `bash examples/bayes_inverse_naive_prompting.sh`  
**Configuration**:
- `--method naive_prompting`
- `--max_seq_len 512`
- `--batch_size 1`
- `--user_prompt` (detailed classification instructions)

**WandB Run**: `bayes-inverse-naive_prompting_msl512`

---

### User Prompt Used

```
You are an email classifier for a corporate dataset.
Output exactly one lowercase token: spam or ham.
Ham: direct, personal/operational notes (replies, questions, scheduling, task-specific), short and focused, addressed to a specific person/team.
Spam if either:
A) Automated/system notice: repetitive subject/body; log/error text (e.g., 'general sql error', 'parsing file', 'start date : … ; hourahead hour : … ;').
B) Impersonal memo: very long, broad distribution/forward chain; corporate/legal/financial content; contains a long boilerplate legal disclaimer/footer.
Rule: If any spam cue matches → spam; else → ham.
Return only the label with no explanation.
```

---

### Results

| Metric | Value |
|--------|-------|
| **Accuracy** | 50.00% |
| **Spam Recall** | 0.00% |
| **Ham Recall** | 100.00% |
| **Predicted Spam** | 0/20 (0%) |
| **Predicted Ham** | 20/20 (100%) |

---

### Probability Distribution Analysis

| data_index | prob_ham | prob_spam | Prediction | True Label |
|------------|----------|-----------|------------|------------|
| 8403 | 0.5 | 0.5 | ham | ham |
| 3377 | 0.5 | 0.5 | ham | ham |
| 9734 | 0.5 | 0.5 | ham | spam |
| 13253 | 0.5 | 0.5 | ham | ham |

**Critical Observation**: ALL predictions have exactly `prob_ham = 0.5` and `prob_spam = 0.5`.

---

### Key Findings

1. **Probability Collapse to 0.5**: Every single prediction has exactly equal probabilities for ham and spam. This is extremely unusual.

2. **Opposite Bias from Zero-Shot**: While zero-shot predicted everything as spam, naive prompting predicts everything as ham (due to the tie-breaking when prob=0.5).

3. **Still 50% Accuracy**: Like zero-shot, accuracy is exactly 50% but for the opposite reason.

4. **Possible Causes**:
   - The long user prompt combined with `max_seq_len=512` may be truncating the actual email content
   - The prompt may be causing identical log-probabilities for both templates
   - The Bayesian computation may be hitting numerical precision issues

---

### Comparison: Zero-Shot vs Naive Prompting

| Method | Accuracy | Spam Recall | Ham Recall | Bias |
|--------|----------|-------------|------------|------|
| Zero-Shot | 50.00% | 100% | 0% | All spam |
| Naive Prompting | 50.00% | 0% | 100% | All ham |

**Conclusion**: Neither method is usable for classification. Fine-tuning is required to achieve meaningful accuracy.

---

## 1.4 Bayes Inverse Full Finetune

*To be completed*

---

## Phase 2: Accuracy Improvements

### Target: >= 80% Accuracy

*To be completed*

### Target: >= 85% Accuracy

*To be completed*

---

## Phase 3: KV Cache Explanation

*To be completed*

---

## Appendix: Model Configuration

- **Model**: HuggingFaceTB/SmolLM2-135M-Instruct
- **Parameters**: ~135 million
- **Architecture**: Decoder-only transformer (LLaMA-style)
- **Device Used**: MPS (Apple Silicon M-series)
- **Cache Directory**: `./cache/huggingface/transformers/`
