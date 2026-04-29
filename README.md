# AI Productivity — Analyzing the Impact of AI on Operational Margins

**Course:** Machine Learning — A.A. 2025/26  
**Team Members:**
- [Student Name 1] — ID: [XXXXXXX]
- [Student Name 2] — ID: [XXXXXXX]
- [Student Name 3 / Captain] — ID: [XXXXXXX]

**GitHub Repository:** [link to repo]

---

## [Section 1] Introduction

This project investigates the financial and operational impact of AI adoption in a digital agency context. The central question is deceptively simple: *when a company starts using AI tools to complete tasks faster, does it actually earn more?*

The dataset covers 3,248 tasks across multiple teams, task types, and pricing models. Each task includes information about hours worked, rework, quality scores, revenue, cost, and AI usage intensity. The unit of analysis is the single task or deliverable.

The phenomenon we are studying is what we call the **AI Productivity Paradox**: companies that adopt AI produce more in less time, yet their margins often stay flat or decline. This happens because efficiency gains are offset by hidden costs — unstable quality, increased rework, and pricing models that no longer reflect the value produced.

Our analysis is structured around four research questions:
1. **RQ1** — Where is value created? Which tasks and contexts benefit most from AI?
2. **RQ2** — Where are losses incurred? What drives loss-making tasks?
3. **RQ3** — Does AI improve quality, or just speed?
4. **RQ4** — Is there a threshold beyond which AI becomes harmful to margin?

We also address three advanced questions on speed-quality trade-offs, rework thresholds, and pricing model sustainability.

> **Note — Partial Submission:** RQ1 and the full data pipeline (cleaning, feature engineering, EDA) are complete and finalized. RQ2, RQ3, RQ4, and the Advanced Questions contain preliminary results that are still being refined. Section 4 and Section 5 will be updated in the final submission.

---

## [Section 2] Methods

### 2.1 Dataset and Preprocessing

The dataset is a single CSV file (`ai_productivity_dataset_final.csv`) containing task-level records. It is intentionally imperfect — missing values, label inconsistencies, and logical contradictions are all present, mimicking real operational data.

The preprocessing pipeline follows these steps:

1. **Categorical cleaning** — Column values in `team`, `task_type`, and related fields were standardized: lowercasing, whitespace stripping, typo correction (e.g., `contennt` → `content`, `desgn` → `design`), and merging of low-frequency subcategories into parent categories.
2. **Date conversion** — Columns `created_at`, `delivered_at`, and `updated_at` were converted from strings to `datetime` objects to enable time-based feature engineering.
3. **Missing value treatment** — Numerical columns with missing values were imputed using the **median** (MCAR assumption). The `delivered_at` column had a distinct missingness pattern and was left without imputation, as the cause was not fully diagnosed.
4. **Inconsistency resolution** — Logical inconsistencies were identified and corrected: negative `billable_hours` (17 cases, set to 0), mismatches between `ai_assisted` flag and `ai_usage_pct > 0` (685 cases, flag corrected), and date ordering violations.

### 2.2 Feature Engineering

We created several derived variables to operationalize the research questions:

**AI intensity grouping:**
- `ai_group`: `ai_usage_pct` binned into five ordered categories (0–15%, 15–30%, 30–50%, 50–75%, 75–100%). This allowed pattern detection that was not visible in raw continuous form.
- `ai_complexity_interaction`: product of `ai_usage_pct` and `task_complexity_score`, capturing whether AI impact depends on task difficulty.

**Cost and efficiency metrics:**
- `rework_ratio`: share of total hours spent on corrections (`rework_hours / hours_spent`)
- `cost_ratio`: total cost relative to revenue (`cost / revenue`); values above 1 indicate a loss-making task
- `efficiency`: ratio of billable hours to hours spent

**Error and revision metrics:**
- `error_rate`: number of errors per hour spent
- `revisions_rate`: number of revisions per hour spent

**Profitability metrics:**
- `profit_margin`: retained revenue fraction (`profit / revenue`)
- `is_loss`: binary flag for tasks where `profit < 0`
- `profit_per_hour`: profit generated per hour of work

**Quality and speed indices (for RQ3):**
- `quality_index`: composite of normalized `outcome_score` and inverted `rework_ratio`
- `speed_index`: based on `sla_ratio = delivery_time / sla_days`; lower values mean faster delivery relative to deadline

### 2.3 Analytical Approach

This is a **descriptive and exploratory analysis**, not a supervised learning problem. There is no model to train or tune. The goal is to understand mechanisms — why margins change, when losses occur, for whom AI creates or destroys value.

The main analytical tools are:

- **Grouped comparisons** across `ai_group`, `task_type`, `team`, `pricing_model`, and `client_tier`
- **Pearson correlations** to measure the linear relationship between AI usage and derived metrics
- **Polynomial regression (degree 2)** on `ai_usage_pct` vs `profit_margin` to detect non-linear patterns and potential inflection points
- **Loss rate analysis**: proportion of `is_loss == True` by group

All visualizations are generated directly from the notebook code.

### 2.4 Environment

The analysis was conducted in Python 3. Key libraries:

```
numpy
pandas
matplotlib
seaborn
scipy
missingno
```

To recreate the environment:

```bash
pip install numpy pandas matplotlib seaborn scipy missingno
```

A `requirements.txt` / `environment.yml` file is included in the repository root.

---

## [Section 3] Experimental Design

Because this is an exploratory analysis rather than a predictive modeling project, the concept of "experiment" is interpreted as a structured investigation of a specific research question, where we define a comparison group, a metric, and an expected direction based on the business context.

### Experiment 1 — Where is value created? (RQ1)

**Purpose:** Determine whether higher AI usage is associated with better financial performance at the task level.

**Baseline:** The null hypothesis is that `ai_group` has no effect on `profit_margin`. If AI were irrelevant, we would expect the metric to be flat across all usage groups.

**Evaluation metrics:** Mean and median `profit_margin` by `ai_group`; loss rate (`is_loss`) by `ai_group`. We also check whether the pattern holds within specific `task_type` and `team` combinations to test for heterogeneity.

### Experiment 2 — Where are losses incurred? (RQ2)

**Purpose:** Identify which structural factors (task type, team, seniority, client tier, pricing model) are most associated with loss-making tasks.

**Baseline:** A uniform loss rate across all categories. Any group that deviates significantly from the overall loss rate (~22%) is flagged as a risk factor or a protective factor.

**Evaluation metrics:** `loss_rate` by `task_type`, `team`, `employee_seniority`, `client_tier`; `cost_ratio` by `ai_group`; `error_rate` and `revisions_rate` by `ai_group`.

### Experiment 3 — AI → quality or just speed? (RQ3)

**Purpose:** Decompose the effect of AI into two separate dimensions — output quality and delivery speed — and measure each independently.

**Baseline:** The implicit assumption that "AI makes work better and faster." We test whether this is actually supported by the data.

**Evaluation metrics:** Pearson correlation between `ai_usage_pct` and `quality_index`; Pearson correlation between `ai_usage_pct` and `speed_index`. We consider a correlation meaningful only if |r| > 0.1 and p < 0.05.

### Experiment 4 — When does AI become harmful? (RQ4)

**Purpose:** Detect whether there is a usage threshold beyond which AI adoption starts to damage profit margin.

**Baseline:** A linear relationship between `ai_usage_pct` and `profit_margin`, where more AI is always better. If a threshold exists, a degree-2 polynomial regression should show a negative quadratic coefficient.

**Evaluation metrics:** Polynomial regression coefficients and the fitted curve shape; mean `profit_margin` by `ai_group` and `task_type` combination to detect task-specific reversals.

---

## [Section 4] Results

> **Note:** The results below reflect the current state of the analysis. RQ1 is fully concluded. RQ2, RQ3, and RQ4 contain preliminary findings that may be revised.

### RQ1 — Where is value created?

AI adoption is clearly associated with higher financial performance. Profit margin increases nearly monotonically with AI usage intensity: tasks in the 75–100% AI group have a median profit margin of **0.48**, compared to **0.22** for tasks in the 0–15% group. The loss rate follows the opposite pattern, dropping from **30%** at low adoption to **12%** at high adoption.

One consistent anomaly across all analyses is the **15–30% AI group**, which performs worse than the 0–15% group across multiple metrics. This "partial adoption trap" suggests that introducing AI without fully integrating it adds overhead without delivering efficiency gains.

Value creation is not uniform across task types. Article and design tasks respond well to AI; ticket and release tasks remain risky regardless of AI intensity.

| AI Group | Mean Profit Margin | Loss Rate |
|---|---|---|
| 0–15% | ~0.22 | ~30% |
| 15–30% | ~0.18 | ~32% |
| 30–50% | ~0.23 | ~25% |
| 50–75% | ~0.31 | ~18% |
| 75–100% | ~0.48 | ~12% |

*Table 1 — Profit margin and loss rate by AI usage group. Values are approximate and will be updated with final figures in the complete submission.*

### RQ2 — Where are losses incurred? *(preliminary)*

Loss-making tasks are not defined by high rework or poor quality — they are defined by a structural **pricing mismatch**. Loss tasks have a `cost_ratio` averaging 1.6x their revenue, but they have slightly lower rework and similar or better output quality compared to profitable tasks.

The highest-risk combinations are:
- **Task type:** tickets and releases
- **Team:** seo and content
- **Seniority:** senior employees (loss rate ~45%), likely because their higher hourly cost is not priced into deliverables
- **Client tier:** low-tier clients

### RQ3 — AI → quality or just speed? *(preliminary)*

Neither. AI does not meaningfully improve quality or speed.

- `quality_index` vs `ai_usage_pct`: r = −0.025, p = 0.157 → no relationship
- `speed_index` vs `ai_usage_pct`: r = 0.054, p = 0.002 → statistically significant but negligible (AI explains < 0.3% of speed variance)

AI creates value exclusively through cost efficiency — reducing waste and cost overruns — not through better outputs or faster delivery.

### RQ4 — When does it become negative? *(preliminary)*

At the aggregate level, AI never becomes harmful to margin. The polynomial regression (degree 2) shows both coefficients positive, meaning the curve accelerates upward and never turns downward.

The only exception: **article tasks at 75–100% AI usage** show a mean profit margin of −0.15. This is the only task-type and AI-intensity combination where heavy AI adoption is associated with negative outcomes, likely because AI-generated text requires more revision cycles.

The real risk threshold is not "too much AI" — it is **partial adoption (15–30%)**, consistently the worst-performing group across all metrics.

![Placeholder — Profit Margin by AI Group and Task Type](images/profit_margin_ai_group_task_type.png)
*Figure 1 — Mean profit margin by AI group and task type (generated from notebook code).*

---

## [Section 5] Conclusions

> **Note:** This is a partial conclusions section. It will be fully rewritten once RQ2–RQ4 analyses are finalized.

### Summary of findings so far

The data tells a consistent story across all four research questions. AI creates real financial value in digital agency work, but the mechanism is not what most people expect. AI does not make work better (quality_index is flat across all groups) and does not make work measurably faster (speed_index correlation r = 0.054). Instead, AI reduces the frequency and severity of cost overruns — it makes the cost structure more predictable, which directly improves profit margins.

The most important structural finding is about pricing, not AI. Loss-making tasks look operationally similar to profitable ones: they have less rework, similar quality, and similar error rates. The only difference is that their costs exceed their revenue. This points to a pricing problem — tasks are being billed at rates that do not cover their actual cost, regardless of how much AI is used. The hourly pricing model is the clearest manifestation of this: it has the highest loss rate (33.3%) and the lowest profit per hour (median ~45), and AI cannot fix it.

### Open questions and next steps

This analysis has several limitations that future work should address.

First, causality cannot be established. We observe that high AI usage correlates with higher margins, but we cannot rule out selection effects — more profitable task types or more experienced teams may simply be more likely to adopt AI heavily. A controlled experiment or a difference-in-differences approach with time-series data would be needed to make causal claims.

Second, the `delivered_at` missingness was not fully resolved. If missing delivery dates are systematically associated with specific task outcomes, this could introduce bias in the speed-related analyses (RQ3).

Third, the analysis is entirely aggregate — it treats the 3,248 tasks as independent observations. A more granular approach would account for team-level and project-level effects using mixed models or hierarchical regression.

Finally, several variables that would be needed for a complete picture are absent from the dataset: client satisfaction scores beyond `outcome_score`, employee AI training levels, task brief quality at submission, and actual time-to-value for the client. These would allow a richer model of when and for whom AI adoption translates into sustainable financial performance.
