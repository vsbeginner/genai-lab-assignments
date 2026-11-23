# Capstone Project: Domain-Adapted Study Tip Generator
### Generative AI Creative Solution for Pedagogical Scaffolding

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vsbeginner-genai-study-tip.streamlit.app)
[![Live Demo](https://img.shields.io/badge/Streamlit-Live%20Demo-FF4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white)](https://vsbeginner-genai-study-tip.streamlit.app)

**Course Code**: ETMMGA304 — Generative AI and Its Applications  
**Academic Program**: Master of Computer Applications (MCA)  
**Institution**: K.R. Mangalam University  
**Candidate**: Vinayak Sharma ([@vsbeginner](https://github.com/vsbeginner))  
**Live Interactive Studio**: [https://vsbeginner-genai-study-tip.streamlit.app](https://vsbeginner-genai-study-tip.streamlit.app)  
**Pedagogical Taxonomy**: BTX 4 (Analyze), BTX 5 (Evaluate) & BTX 6 (Create)  

---

## Executive Summary

The objective of this Capstone Project is to architect, adapt, rigorously evaluate, and deploy an end-to-end **Generative AI Creative Solution** addressing a real-world educational challenge: **synthesizing concise, actionable, and pedagogical study tips for beginner computer science and mathematics students**.

Beginning students exploring abstract technical topics (e.g., *recursion, SQL joins, hash tables*) are frequently overwhelmed by dense textbook definitions, encyclopedic explanations, and verbose forum threads. They require immediate, single-sentence actionable scaffolding—a tangible exercise or visualization task to initiate conceptual comprehension.

Building upon the Parameter-Efficient Fine-Tuning (PEFT) methodology established in **Lab 3**, this capstone implements a complete 5-stage engineering lifecycle:
$$\text{Define} \longrightarrow \text{Prepare Data} \longrightarrow \text{Adapt Model} \longrightarrow \text{Evaluate Benchmarks} \longrightarrow \text{Demonstrate Application}$$

```
                                      CAPSTONE ENGINEERING LIFECYCLE
  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
  │  1. DEFINE   │ ──►  │  2. PREPARE  │ ──►  │   3. ADAPT   │ ──►  │ 4. EVALUATE  │ ──►  │5. DEMONSTRATE│
  │ Problem &    │      │ Curate 40    │      │ PEFT GPT-2   │      │ 12 Held-Out  │      │ Streamlit UI │
  │ SMART Goals  │      │ Pairs + Split│      │ Block 11     │      │ Multi-Metric │      │ & AppTest CI │
  └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘      └──────────────┘
```

---

## 1. Stage 1: Define — Problem Statement & Success Criteria

### 1.1 Problem Definition
When queried on technical subjects, pre-trained base foundation models (such as zero-shot GPT-2 124M) exhibit three consistent failure modes:
1. **Unconstrained Generation**: Dumping code snippets, website boilerplate, or academic citations rather than pedagogic advice.
2. **Repetitive Looping**: Becoming trapped in deterministic sentence loops under greedy decoding.
3. **Passive Tautology**: Re-stating the definition of the term (e.g., *"A hash table is a table that is used for hashes"*).

**Target Solution**: An adapted causal language model that generates exactly **one concise, imperative instruction** telling the learner a physical or mental action to perform (e.g., *"Trace a loop by writing the counter after each step"*).

### 1.2 SMART Success Criteria (Pre-Registered Before Adaptation)
1. **Structural Conciseness**: The output must consist of a single actionable imperative sentence ($\le 45$ tokens) that terminates cleanly.
2. **Topical Relevance**: The advice must directly address the subject concept.
3. **Format Readability**: The generation must be free of code dumps, repetitive loops, URL artifacts, and news boilerplate.
4. **Empirical Benchmark Superiority**: The adapted model must achieve a higher mean overall score (out of 6.0) than the base model across at least 10 held-out evaluation topics.

---

## 2. Stage 2: Prepare — Dataset Engineering & Split Integrity

| Parameter | Specification & Engineering Details |
|:---|:---|
| **Curation Methodology** | Hand-authored domain pairs crafted specifically for beginner algorithmic, mathematical, and software engineering pedagogy. No automated web-scraping or third-party datasets were used, ensuring clean data provenance. |
| **Training Split (`data/train.json`)** | **40 instructional pairs** covering fundamental programming paradigms (loops, conditionals, functions), data structures (trees, arrays, lists, dictionaries), mathematics (algebra, permutations, probability, limits), and systems concepts (IP addresses, processes, normalization). |
| **Held-Out Benchmark Split (`data/heldout.json`)** | **12 unseen topics**: `recursion`, `matrices`, `binary search`, `linked lists`, `SQL joins`, `stacks and queues`, `hash tables`, `Big O notation`, `Git branches`, `derivatives`, `standard deviation`, `object-oriented programming`. |
| **Mathematical Zero-Leakage Guarantee** | The training pipeline programmatically enforces that the set intersection between normalized training topics and held-out evaluation topics is strictly empty: $\{t_{\text{train}}\} \cap \{t_{\text{heldout}}\} = \emptyset$. |
| **Formatting Template** | Standardized prompt-response syntax: `Topic: <topic>\nStudy tip: <tip><|endoftext|>`. |

---

## 3. Stage 3: Adapt — Parameter-Efficient Fine-Tuning (PEFT)

### 3.1 Adaptation Methodology
- **Base Architecture**: `openai-community/gpt2` (12-layer Transformer Decoder, 124,439,808 parameters).
- **Freezing Scheme**: Blocks $0$ through $10$ along with token and positional embeddings are frozen.
- **Active Subsystem**: Only Transformer Block 11 (`transformer.h[-1]`) is unfrozen, updating only **7,087,872 parameters (5.69%)**, conserving GPU VRAM and preventing catastrophic forgetting of base language syntax.
- **Prompt Token Masking**: Prompt prefix tokens are masked with label `-100` so that cross-entropy loss gradients are exclusively computed over the response tokens.
- **Hyperparameters**:
  - Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.999$, Weight Decay $= 0.01$)
  - Learning Rate: $\eta = 1 \times 10^{-4}$ (scaled from Lab 3's $2 \times 10^{-5}$ to accelerate single-block adaptation)
  - Epochs: 10 epochs (400 optimization steps, reshuffled per epoch)
  - Batch Size: 1 (online stochastic gradient updates)

### 3.2 Training Loss Trajectory

```
  Mean Loss
   4.5 ┼  4.399
   4.0 ┼      3.771
   3.5 ┼          3.441
   3.0 ┼              3.069
   2.5 ┼                  2.725
   2.0 ┼                      2.397
   1.5 ┼                          1.998   1.811
   1.0 ┼                                      1.483   1.227
   0.5 ┼────────────────────────────────────────────────────
       01     02     03     04     05     06     07     08     09     10    Epoch
```

The mean training loss declined smoothly from **4.399** at Epoch 1 down to **1.227** at Epoch 10, confirming steady convergence of the adapted block.

---

## 4. Stage 4: Evaluate — Systematic Benchmark & Ablation

Both the zero-shot base GPT-2 and the domain-adapted model were evaluated on the exact same 12 held-out topics using deterministic greedy decoding (`max_new_tokens = 45`).

Scoring utilized an objective 3-factor rubric ($0$: Poor, $1$: Partial, $2$: Strong):
- **Relevance** ($0-2$): Does the completion directly address the queried concept?
- **Readability** ($0-2$): Is the output grammatical, concise, free of loops, code artifacts, or boilerplate?
- **Usefulness** ($0-2$): Does it provide actionable, pedagogically valid guidance for a beginner?
- **Total Score**: Sum across all three criteria (maximum: **6.0 points**).

### 4.1 Comparative Evaluation Record (All 12 Benchmark Topics)

| Topic | Model | Relevance (0-2) | Readability (0-2) | Usefulness (0-2) | Total (/6) | Qualitative Analysis & Notes |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| **recursion** | Base | 1 | 1 | 0 | 2 | Names the topic, then slides into fragmented C++ code (`Recursion(int i)`). Non-actionable. |
| | **Adapted** | **1** | **2** | **1** | **4** | Clean imperative sentence (*"Draw a small tree with a root and test it with three children"*). Grammatical and structured; transfers tree visualization to recursion. |
| **matrices** | Base | 1 | 1 | 0 | 2 | Vague, circular sentence; enters a repetition loop. |
| | **Adapted** | **0** | **1** | **0** | **1** | *"Draw the points as you see fit."* Syntactically clean but completely disconnected from matrices. |
| **binary search** | Base | 0 | 1 | 0 | 1 | Confuses binary search with a website search box; repeats itself. |
| | **Adapted** | **1** | **1** | **0** | **2** | *"Look for two values before looking at the expression result."* Hints at searching; slightly muddled. |
| **linked lists** | Base | 0 | 1 | 0 | 1 | Hallucinates news article boilerplate (*"This article was originally published on The Conversation..."*). |
| | **Adapted** | **1** | **1** | **0** | **2** | Mentions lists and first/last elements, but phrasing is somewhat convoluted. |
| **SQL joins** | Base | 1 | 1 | 0 | 2 | Mentions table joining, but loops repeatedly on the prompt phrase. |
| | **Adapted** | **2** | **2** | **1** | **5** | **Exemplary Output**: *"List the three columns you need to join the table."* Concise, on-topic, and actionable. |
| **stacks and queues**| Base | 1 | 1 | 0 | 2 | References queues, then repeats *"The data is stored in a queue"*. |
| | **Adapted** | **0** | **2** | **0** | **2** | Fluent English (*"Fill in the numbers for two variables and print the result"*), but unrelated to stacks/queues. |
| **hash tables** | Base | 1 | 1 | 0 | 2 | Circular definition (*"The hash table is a set of tables that are used to store..."*). |
| | **Adapted** | **2** | **2** | **1** | **5** | **Strong Output**: *"Store three names with the same key before you open a hash table."* Understands key-value association. |
| **Big O notation** | Base | 0 | 0 | 0 | 0 | Pathological failure: dumps a string of characters (*"1.1.1.1.1.1..."*). |
| | **Adapted** | **0** | **1** | **0** | **1** | *"Draw the root letter of each cell you can mark."* Non-sensical content, but terminates cleanly. |
| **Git branches** | Base | 2 | 1 | 1 | 4 | Understands topic, but repeats the `"Study tip:"` header three times. |
| | **Adapted** | **1** | **1** | **0** | **2** | Circular phrasing (*"Build a branch with one or two branches, then open it in a new tab"*). |
| **derivatives** | Base | 0 | 1 | 0 | 1 | Severe domain confusion: discusses the financial derivatives market and quotes a URL. |
| | **Adapted** | **0** | **0** | **0** | **0** | **Total Failure**: *"Swap one swap for one swap."* Fluent gibberish (see Section 4.3). |
| **standard deviation**| Base | 1 | 1 | 0 | 2 | Quotes CDC survey health statistics. |
| | **Adapted** | **1** | **2** | **1** | **4** | *"Sort the numbers first and compare the scores."* Valid first step for variance, but misses spread formula. |
| **object-oriented** | Base | 1 | 0 | 0 | 1 | Infinite repetition loop (*"create a class that will be used to create a class..."*). |
| | **Adapted** | **0** | **0** | **0** | **0** | Rambles about GitHub issues. |

---

### 4.2 Aggregate Statistical Benchmark

Aggregated across all 12 held-out evaluation topics:

| Model Architecture | Mean Relevance (/2.0) | Mean Readability (/2.0) | Mean Usefulness (/2.0) | **Mean Overall Score (/6.0)** |
|:---|:---:|:---:|:---:|:---:|
| **Zero-Shot Base GPT-2 (124M)** | 0.75 | 0.83 | 0.08 | **1.67** |
| **Domain-Adapted GPT-2 (Block 11)** | **0.75** | **1.25** (+50.6%) | **0.33** (+312%) | **2.33** (+39.5%) |

- **Win / Tie / Loss Ratio**: The adapted model achieved higher scores on **7 topics**, tied on **1 topic**, and scored lower on **4 topics**.

#### Key Architectural Findings:
1. **Formatting Acquisition vs Factual Learning**: Notice that **Mean Relevance remained identical at 0.75** for both models. Adapting 5.7% of weights on 40 examples did not inject new semantic knowledge into GPT-2. Instead, the performance gain stems almost entirely from **Readability ($0.83 \to 1.25$)** and **Usefulness ($0.08 \to 0.33$)**. The adapted model learned the *structural discipline* of outputting a single imperative command and terminating cleanly, eliminating the repetition loops and code dumps that plagued the base model.
2. **Domain Proximity Effect**: The adapted model performed best on topics adjacent to the training distribution (e.g., `SQL joins` scored 5/6 due to training examples on `SQL SELECT`; `hash tables` scored 5/6 due to training on `dictionaries`). On completely unrelated topics (`derivatives`, `Big O`), the model struggled.

---

### 4.3 Rigorous Failure Mode Analysis (Required Deep-Dive)

- **Failure Case**: Topic: **`derivatives`**
  - **Base Model**: *"If you're looking for a way to make a profit, you might want to look at the derivatives market. Source: http://..."*
  - **Adapted Model**: *"Swap one swap for one swap."*
- **Root Cause Diagnosis**:
  - The training dataset contained no calculus or advanced mathematical analysis pairs (only basic limits and algebra).
  - When prompted with an out-of-distribution keyword, the adapted Transformer block failed to activate relevant factual attention heads.
  - Instead, the model fell back onto high-frequency tokens from the sorting training example (*"Sort five values by hand and record each swap"*), stringing the word "swap" into a grammatically plausible but semantically empty loop.
- **Pedagogical Hazard**: Because the sentence *"Swap one swap for one swap"* is concise and grammatically structured, an automated pipeline might mistake it for valid text. A student reading this would waste time trying to decipher a non-existent concept.
- **Remediation Protocol for Instructors**:
  1. **Automated Keyword Overlap Filtering**: Enforce an automated lexical verification rule; if the generated tip shares zero vocabulary or stem overlap with the prompt or its known synonyms, immediately flag the tip for rejection.
  2. **Active Curriculum Expansion**: Route all rejected outputs to an instructor queue. The instructor provides the correct tip (*"Differentiate a polynomial step-by-step using the power rule"*), which is added to the training corpus for subsequent fine-tuning iterations.

---

## 5. Stage 5: Demonstrate — Production Streamlit Demonstration

### 5.1 Web Application Architecture (`app.py`)
The demonstration interface is built using Streamlit, featuring:
- **Public Live Deployment**: [https://vsbeginner-genai-study-tip.streamlit.app](https://vsbeginner-genai-study-tip.streamlit.app)
- **Executive Recruiter Studio**: Tabbed interface featuring Live Generation Studio, Held-Out Benchmark Scorecard, Architectural Deep-Dive, and Viva-Voce Technical Q&A.
- **Resilient Dual-Mode Engine**: Operates with zero setup latency using an intelligent pre-computed benchmark lookup engine (all 52 training and held-out pairs), and seamlessly mounts local fine-tuned PyTorch checkpoint (`study_tip_gpt2`) if available.
- **Fast Execution & Interactive Presets**: Interactive selector with benchmark quick-picks (`recursion`, `SQL joins`, `binary search`, etc.) and side-by-side empirical comparison cards (Base GPT-2 vs Adapted PEFT).
- In-memory model caching via `@st.cache_data` for sub-second responses.
- Maximum input character bounding (120 characters) to prevent buffer overflows and prompt injection.
- Form validation preventing blank or whitespace submissions.
- Prominent pedagogical disclosure notices and PII warnings.

#### Launching the Application
- 🌐 **Live Cloud Demo**: [https://vsbeginner-genai-study-tip.streamlit.app](https://vsbeginner-genai-study-tip.streamlit.app)
- 💻 **Local Launch**:
```bash
# Navigate to capstone directory
cd capstone_study_tip_generator

# Launch Streamlit web app locally
python -m streamlit run app.py

# Or execute headless CI/CD test harness
python test_app.py
```

### 5.2 Automated Headless UI Test Harness (`test_app.py`)
To ensure enterprise CI/CD reliability, the interface was headlessly tested using Streamlit's official `AppTest` framework without requiring a browser:

| Test Assertion | Input Vector | Expected UI Behavior | Verification Status |
|:---|:---|:---|:---:|
| **Empty Submission** | `""` | Warning: *"Enter a study topic first."* (No model inference) | **PASSED** |
| **Whitespace Edge Case** | `"   "` | Warning: *"Enter a study topic first."* (Input cleaned & caught) | **PASSED** |
| **Held-Out Topic: Recursion** | `"recursion"` | Outputs: *"Draw a small tree with a root and test it with three children."* | **PASSED** |
| **Held-Out Topic: SQL Joins** | `"SQL joins"` | Outputs: *"List the three columns you need to join the table."* | **PASSED** |

Telemetry is persisted in `lab_outputs/demo_checks.json`.

---

## 6. Model Governance, Ethics & Deployment Safeguards

1. **Human-in-the-Loop Requirement**: At an average usefulness score of **0.33/2.0**, this model cannot be deployed as an unmoderated, autonomous student tutor. It is strictly configured as an **instructor-assistive co-pilot**: drafting suggested tips that educators review, edit, and approve.
2. **Local Inference & Data Privacy**: All inference occurs locally on-device. No user input or synthesized outputs are transmitted to external cloud APIs or third-party tracking services.
3. **Synthetic Content Disclaimers**: Every output generated by the application displays an indelible disclosure caption reminding users that the content is AI-generated and subject to human review.

---

## 7. Production Roadmap & Architectural Scaling

1. **Low-Rank Adaptation (LoRA / QLoRA)**: Migrate from full-layer tuning of Block 11 to rank-8 LoRA adapters across query/key/value projections in all 12 blocks, enabling broader semantic adaptation without increasing parameter count.
2. **Retrieval-Augmented Generation (RAG)**: Integrate a vector database (e.g., ChromaDB with `sentence-transformers`) containing open educational textbook snippets. Conditioning GPT-2 on retrieved textbook passages will eliminate out-of-domain hallucinations like the derivatives failure.
3. **Multi-Rater Agreement & LLM-as-a-Judge**: Expand evaluation to 3 independent academic raters and compute Cohen's Kappa ($\kappa$) to establish inter-rater statistical reliability.

---

## 8. Viva-Voce Examination Guide (Technical Q&A)

### Q1: Why was parameter-efficient fine-tuning (PEFT) on a single block chosen over training all 124 million parameters?
> **Answer**: Full fine-tuning of 124M parameters on a tiny dataset of 40 examples would cause catastrophic overfitting and catastrophic forgetting—the network would memorize the 40 sentences word-for-word while destroying its pre-trained general linguistic fluency. Furthermore, full fine-tuning requires tracking optimizer states for all weights (consuming substantial GPU VRAM). Freezing 94.3% of the network and training only Transformer Block 11 (`transformer.h[-1]`) acts as a strong regularizer: it forces the pre-trained semantic representations to remain stable while allowing the final block to adapt its projection heads to the desired imperative output format.

### Q2: What quantitative evidence proves that adaptation improved the model, and where does the model still fall short?
> **Answer**: 
> - **Evidence of Improvement**: The adapted model achieved a mean total score of **2.33 / 6.0** compared to **1.67 / 6.0** for the base model across 12 held-out topics, winning on 7 out of 12 topics. Specifically, **Readability increased from 0.83 to 1.25** (+50.6%) and **Usefulness increased from 0.08 to 0.33** (+312%), demonstrating successful acquisition of the concise single-sentence instructional format.
> - **Evidence of Shortcomings**: **Mean Relevance remained static at 0.75 / 2.0** for both models, and the adapted model scored lower than the base model on 4 topics (`matrices`, `Git branches`, `derivatives`, `object-oriented programming`). This demonstrates that while the model successfully learned *how to format* a study tip, 40 examples were insufficient to teach it *what factual advice* to give for concepts outside its immediate training domain.

### Q3: What mandatory governance protocol must be enforced before exposing this tool to students?
> **Answer**: An **Instructor-in-the-Loop verification protocol** must be enforced. Under this protocol:
> 1. The application serves strictly as an authoring assistant for curriculum developers and teachers.
> 2. No tip is delivered to a student until a human subject-matter expert verifies that the tip is factually correct, pedagogically actionable, and free of conceptual errors.
> 3. The user interface must maintain permanent synthetic media disclosures, preventing learners from mistaking AI hallucinations for vetted educational material.
