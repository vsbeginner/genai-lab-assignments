# Laboratory Assignment 03: Autoregressive Language Modeling with GPT-2

**Course Code**: ETMMGA304 — Generative AI and Its Applications  
**Academic Program**: Master of Computer Applications (MCA)  
**Institution**: K.R. Mangalam University  
**Pedagogical Taxonomy**: BTX 3 (Apply), BTX 4 (Analyze) & BTX 5 (Evaluate)  

---

## 1. Experimental Overview & Objectives

The primary objective of this laboratory assignment is to analyze the mechanics of **Autoregressive Causal Language Modeling** utilizing OpenAI's **GPT-2** (124M parameters, `openai-community/gpt2`). The experiment is divided into two empirical investigations:
1. **Decoding Strategy & Hyperparameter Ablation**: Evaluating the behavior, degenerative repetition, and diversity of Greedy Search versus Stochastic Nucleus ($p=0.9$) Sampling across varying temperature scales ($T \in \{0.6, 1.0\}$) and random seeds ($42, 43$).
2. **Parameter-Efficient Fine-Tuning (PEFT)**: Investigating parameter freezing and downstream adaptation by fine-tuning strictly the final Transformer block (`transformer.h[-1]`) on structured instructional prompt-response pairs with prompt-token loss masking.

```
Input Prompt ──► BPE Tokenizer ──► 12-Layer Transformer Decoder ──► Next-Token Logits ──► Temperature / Top-p ──► Generated Output
                                   [Blocks 0-10 Frozen]
                                   [Block 11 Unfrozen (PEFT)]
```

---

## 2. Theoretical Foundations & Mathematical Formulations

### 2.1 Causal Autoregressive Language Modeling
Given an input token sequence $W = (w_1, w_2, \dots, w_n)$, a causal language model factorizes the joint probability via the chain rule of probability:

$$P(W) = \prod_{t=1}^{n} P\left(w_t \,\middle|\, w_1, w_2, \dots, w_{t-1}\right)$$

During generation, at each forward time-step $t$, the network outputs an unnormalized logit vector $z_t \in \mathbb{R}^{|\mathcal{V}|}$ over vocabulary $\mathcal{V}$ ($|\mathcal{V}| = 50,257$).

### 2.2 Decoding Strategies

1. **Greedy Search**: Selects the mode of the distribution at every step:
   $$\hat{w}_t = \arg\max_{w \in \mathcal{V}} P(w \mid w_{<t})$$
   *Failure Mode*: Greedy decoding is deterministic and computationally myopic; it frequently gets trapped in infinite repetitive loops and produces generic, degenerate text.

2. **Temperature Rescaling**: Logits are divided by temperature parameter $T > 0$ prior to the softmax transformation:
   $$P(w_i \mid w_{<t}) = \frac{\exp(z_i / T)}{\sum_{j=1}^{|\mathcal{V}|} \exp(z_j / T)}$$
   - $T < 1.0$ (e.g., $T=0.6$): Sharpens the distribution, suppressing low-probability tokens to promote grammatical coherence and reduce hallucinations.
   - $T > 1.0$ (e.g., $T=1.0$): Preserves the raw logit variance, promoting lexical diversity and creative novelty at the risk of semantic drift and syntactic breakdown.

3. **Nucleus (Top-p) Sampling**: Dynamically truncates the sampling pool to the smallest subset of tokens $V^{(p)} \subset \mathcal{V}$ whose cumulative probability mass exceeds threshold $p \in (0, 1]$:
   $$\sum_{w \in V^{(p)}} P(w \mid w_{<t}) \ge p$$
   Probabilities are renormalized over $V^{(p)}$. Setting $p=0.9$ effectively trims the long tail of unreliable low-probability tokens.

### 2.3 Parameter-Efficient Fine-Tuning (PEFT) & Prompt Masking
To demonstrate low-resource downstream adaptation without updating all 124M parameters:
- **Layer Freezing**: Parameters in word token embeddings, positional embeddings, and Transformer blocks $0$ through $10$ are frozen (`requires_grad = False`).
- **Active Adapter**: Only Transformer block $11$ (`transformer.h[-1]`) is unfrozen, updating only **7,087,872 parameters** (just **5.69%** of the total 124,439,808 parameters).
- **Prompt Token Masking**: Standard causal language modeling trains on the entire sequence. However, in conditional instruction following, the model should only be penalized for generating the target advice. Prompt tokens (`Topic: ...\nStudy tip:`) are mapped to target label index `-100`, which PyTorch's `CrossEntropyLoss` ignores during backpropagation:
  $$\mathcal{L} = -\sum_{t = |W_{\text{prompt}}| + 1}^{|W|} \log P\left(w_t \,\middle|\, w_{<t}\right)$$

---

## 3. Experimental Deliverables & File Manifest

| Artifact | Description |
|:---|:---|
| `lab3_gpt2.py` | Complete Python script executing decoding comparisons, PEFT training, and held-out scoring. |
| `lab_outputs/outputs.json` | Complete machine-readable traces of all prompt completions, temperature grids, loss records, and held-out comparisons. |
| `lab_outputs/study_tip_gpt2/` | Directory containing fine-tuned model weights and tokenizer configurations. |

To reproduce locally:
```bash
cd lab3_gpt2
python lab3_gpt2.py
```

---

## 4. Empirical Results & Performance Analysis

### 4.1 Part 1: Decoding Ablation Study

Evaluated under `max_new_tokens = 45`. Scoring uses a structured 3-factor rubric ($0$: Poor, $1$: Partial, $2$: Strong) across Coherence (syntactic flow), Relevance (prompt adherence), and Creativity (novelty/specificity).

#### Prompt 1: *"A story about a robot begins like this:"*

| Decoding Strategy & Seed | Coherence (0-2) | Relevance (0-2) | Creativity (0-2) | Supporting Output Excerpt & Qualitative Critique |
|:---|:---:|:---:|:---:|:---|
| **Greedy Search** | 1 | 1 | 0 | *"A robot is a robot that is programmed to do something. It is programmed to do something. It is programmed..."*<br>Degenerates into immediate pathological repetition; defines an entity rather than narrating a story. |
| **$T=0.6$, Seed 42** | 1 | 2 | 1 | *"I was in my basement, and a guy came in, and he said, 'Hey, we're going to be building a robot. Let's go...' "*<br>Establishes a genuine conversational narrative opening, but begins repeating phrases toward the token limit. |
| **$T=0.6$, Seed 43** | 1 | 1 | 0 | *"A robot is a computer program that... A robot is not a computer program. A robot is not a computer program."*<br>Contradicts itself immediately and enters a 2-cycle loop. |
| **$T=1.0$, Seed 42** | 1 | 1 | 1 | *"It was supposed to be something big, but it never happened. It was called a toy, but no one liked it..."*<br>Poetic prose, but drifts entirely away from mentioning or describing a robot. |
| **$T=1.0$, Seed 43** | 1 | 2 | 2 | *"A robotic girl runs into a strange, alien world called 'Rome.' Rome's 'Sculpture of Chaos' is where she's found..."*<br>Highly imaginative and visually evocative, though syntax and quotation structures exhibit mild drift. |

#### Prompt 2: *"In a school library, a tiny robot found a lost book. It"*

| Decoding Strategy & Seed | Coherence (0-2) | Relevance (0-2) | Creativity (0-2) | Supporting Output Excerpt & Qualitative Critique |
|:---|:---:|:---:|:---:|:---|
| **Greedy Search** | 1 | 1 | 0 | *"...was a book that had been lost for years. 'I was like, Oh my God, this is so weird,' said the boy, who asked not to be identified."*<br>Abruptly shifts into a journalistic interview style; drops the robot entity entirely. |
| **$T=0.6$, Seed 42** | 1 | 2 | 1 | *"...was the only book in the library, and it was the only one that had been lost in the war. The robot was not the only one who had lost his book."*<br>Strong thematic retention; mild sentence repetition. |
| **$T=0.6$, Seed 43** | 1 | 2 | 1 | *"...was a book with a name: 'The Book of the Dead.' The book was the book of the dead... A few days later, the robot went missing."*<br>Maintains narrative flow; repeats key noun phrases. |
| **$T=1.0$, Seed 42** | 1 | 2 | 1 | *"...was the only book in its collection... A dozen books stood on the shelf, each a different class of books."*<br>Retains library imagery well; semantic substance is somewhat generic. |
| **$T=1.0$, Seed 43** | 1 | 1 | 2 | *"...knew that the book belonged to a man named David, who had been captured by aliens with a key. He thought that he was going to be able to rescue..."*<br>Highly novel plot twist, but completely abandons the school library setting for sci-fi tropes. |

#### Findings on Text Generation Dynamics:
1. **Greedy Search Degeneracy**: Deterministic argmax selection reliably triggers repetitive loops when generating multi-sentence continuations with small base models.
2. **Temperature vs Hallucination Trade-off**: At $T=0.6$, top-p sampling maintains topical focus but frequently exhibits phrase repetition. Raising temperature to $T=1.0$ resolves repetition and introduces imaginative concepts, but substantially increases the risk of topic drift (both Seed 43 runs drifted into aliens).
3. **Stochastic Seed Sensitivity**: For identical prompts and temperature settings, changing the seed from 42 to 43 produced completely different narratives. This proves that single-sample generation is statistically insufficient for evaluating LLM quality.

---

### 4.2 Part 2: Parameter-Efficient Fine-Tuning Analysis

The model was adapted on 6 hand-crafted instructional pairs across 3 epochs (18 optimizer steps) with AdamW ($\text{lr}=2 \times 10^{-5}$):

| Epoch | Final Step Loss | Mean Epoch Loss | Training Dynamics |
|:---:|:---:|:---:|:---|
| 1 | 4.759 | 5.159 | Initial gradient updates on Transformer block 11. |
| 2 | 4.328 | 4.876 | Noticeable loss reduction as formatting patterns are absorbed. |
| 3 | 4.724 | 5.178 | Loss plateau; fluctuations due to tiny batch size ($N=1$) and sample variance. |

#### Held-Out Topic Benchmark Evaluation (Greedy Decoding)
Scored on a 0-4 point scale: **Topic Relevance** (0-2) + **Pedagogical Actionability** (0-2):

| Benchmark Topic | Base Model Score (/4) | Tuned Model Score (/4) | Empirical Comparison & Output Analysis |
|:---|:---:|:---:|:---|
| **Recursion** | 1 | 1 | Both models output: *"Use the recursion keyword to define a new type of recursion..."* followed by fragmented C++ pseudocode (`Recursion(int i)` vs `Recursion(int x, int y)`). Neither provides actionable student study advice. |
| **Matrices** | 1 | 1 | **Identical Completion**: *"Use the matrices to create a matrix of the same size as the input..."* Repeats circular tautologies; no measurable difference. |
| **Binary Search**| 0 | 0 | **Identical Hallucination**: Both models confuse the algorithmic search concept with a web UI search box: *"Use the search box to find the search term you want to search for."* |

#### Analytical Synthesis:
1. **Pipeline Demonstration vs Weight Adaptation**: The experiment proves the technical validity of the PyTorch PEFT pipeline (successful layer unfreezing, loss masking, gradient backpropagation, and weight serialization).
2. **Data & Step Volume Limitation**: 6 examples trained over 18 optimization steps with $\text{lr}=2 \times 10^{-5}$ is numerically insufficient to induce true semantic concept learning in a 124M parameter model.
3. **Base Model Inductive Bias**: The base model already possessed a prior bias toward imperative phrasing when prompted with `Topic: ...\nStudy tip:`. The small fine-tune reinforced the template syntax without altering internal factual knowledge. This motivates expanding the dataset to 40+ examples and 10 epochs, as implemented in our [Capstone Project](../capstone_study_tip_generator).

---

## 5. Viva-Voce Examination Guide (Technical Q&A)

### Q1: How does greedy decoding select the next token, and why does it lead to repetitive degeneration?
> **Answer**: At each step $t$, greedy decoding selects the token with the highest conditional probability: $\hat{w}_t = \arg\max_{w \in \mathcal{V}} P(w \mid w_{<t})$. Because this selection is completely deterministic and greedy, the model cannot look ahead to evaluate the probability of complete sequences. If the model assigns slightly higher probability to a phrase it recently output, that phrase feeds back into the causal context window for step $t+1$, further inflating the probability of repeating the phrase. This positive feedback loop locks the model into an unbreakable cyclic attractor (e.g., *"It is programmed to do something. It is programmed to do something..."*).

### Q2: What is the mathematical effect of the temperature parameter ($T$) during stochastic sampling?
> **Answer**: The temperature parameter acts as a variance scaling factor on the unnormalized logit vector $z$: $\tilde{z}_i = z_i / T$.
> - When $T < 1$, the differences between logits are amplified. As $T \to 0$, the highest logit dominates and the softmax distribution approaches a Dirac delta function (collapsing into greedy search).
> - When $T > 1$, the differences between logits are compressed. As $T \to \infty$, the logits become equivalent and the softmax distribution approaches a uniform distribution $\mathcal{U}(1, |\mathcal{V}|)$, maximizing entropy and introducing unpredictable token choices.

### Q3: Which specific tokens contribute to the loss function during fine-tuning, and how is this enforced in PyTorch?
> **Answer**: Only the target response tokens (the study tip text) and the final end-of-text token (`<|endoftext|>`) contribute to the loss. The prompt tokens (`Topic: ...\nStudy tip:`) must be excluded from loss calculation because the model is not being trained to predict the prompt. This is enforced by assigning a target label of `-100` to all prompt token positions: `labels[:len(prefix)] = -100`. In PyTorch, `nn.CrossEntropyLoss` has its default `ignore_index` parameter set to `-100`, which masks out those positions during both loss summation and gradient computation.

### Q4: Why are held-out evaluation topics strictly excluded from the training dataset?
> **Answer**: Excluded held-out topics are mandatory to evaluate out-of-distribution **generalization** versus mere **memorization**. If the evaluation topics (e.g., recursion, matrices) were present in the training set, a low evaluation loss or fluent response would merely indicate that the network memorized specific token sequences. By evaluating on unseen topics, we measure whether the model learned the underlying meta-task (synthesizing concise, actionable advice for arbitrary concepts) or merely recalled its training corpus.
