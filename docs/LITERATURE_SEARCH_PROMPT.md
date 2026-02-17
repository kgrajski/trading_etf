# Literature Search Prompt

**Instructions:** Copy the text below into Gemini Deep Research or ChatGPT with browsing.
The goal is a structured survey of prior work relevant to our project, with a clear
distinction between industrial-scale players, academic research, and smaller-scale /
retail practitioners.

---

## Prompt

I am building a weekly ETF trading system augmented by an agentic AI analyst. I need a
structured literature review to understand what prior work exists, what has been shown to
work, and where the open problems are. Please search academic papers (arXiv, SSRN, Google
Scholar), industry publications, blog posts, and open-source projects.

### My System (Context)

- **Universe:** ~2,200 U.S.-listed ETFs
- **Strategy:** Mean-reversion — buy ETFs that dropped significantly (bottom 5%, >2%
  weekly loss), targeting a 1–3 week bounce with defined stop-loss and profit targets
- **Quantitative pipeline:** Weekly feature engineering (returns, volatility, beta, ATR,
  Bollinger Bands), backtesting engine with parameter optimization (stop loss, profit
  target, max hold period, position sizing), regime detection (bull/bear via SPY vs SMA50)
- **AI analyst layer:** A LangGraph-based multi-node workflow that:
  1. Searches financial news (Tavily API) for each trade candidate
  2. Performs thematic analysis across candidates (grouping by macro drivers)
  3. Produces per-symbol qualitative assessments (GREEN / YELLOW / RED flags)
  4. Runs a reflection/review step where a senior model critiques the junior model's work
- **Current challenges:**
  - LLM analysis tends to be generic, sometimes hallucinated, when news is weak
  - Neighboring ETFs covering the same sector get inconsistent analysis
  - No integration of quantitative features (beta, RSI, alpha residual) into the AI analysis
  - Considering a multi-agent architecture: Quant Analyst, News Analyst, Synthesizer, Reviewer
- **Tech stack:** Python, LangGraph/LangChain, GPT-4o/GPT-4o-mini, Tavily, Plotly, MLflow, Alpaca API

### What I Need You To Research

#### 1. Mean-Reversion in ETFs
- What does the academic literature say about short-term (1–3 week) mean-reversion
  strategies in ETFs specifically? What are typical Sharpe ratios, win rates, and decay
  characteristics?
- How does this compare to mean-reversion in individual equities?
- Key factors that predict successful mean-reversion trades (volume, volatility regime,
  sector, market cap)
- Does regime detection (bull/bear) meaningfully improve mean-reversion returns?

#### 2. LLM/NLP for Trading Signals
- Survey of using LLMs (GPT-4, Gemini, Claude, FinGPT, BloombergGPT) for generating
  trading signals or sentiment scores
- What is the evidence that LLM-derived sentiment adds alpha over quantitative-only
  approaches?
- Specific papers on news-driven alpha for short-horizon strategies (1–3 weeks)
- What are the failure modes? (hallucination, recency bias, generic outputs, stale
  training data)
- FinBERT vs general-purpose LLMs — which is better for financial sentiment?

#### 3. Multi-Agent Systems for Financial Analysis
- Papers or projects using multi-agent LLM architectures for financial analysis or trading
- Specific architectures: analyst/reviewer patterns, specialist agents (quant + news),
  debate/critique patterns
- LangGraph, AutoGen, CrewAI, or similar frameworks applied to finance
- Evidence of quality improvement from reflection/self-critique patterns in financial
  domain

#### 4. Combining Quantitative and Qualitative Signals
- How do practitioners combine quant signals (technical indicators, factor models) with
  NLP/LLM-derived qualitative signals?
- Alpha/beta decomposition as a way to separate market-wide from idiosyncratic moves —
  any research on feeding this to LLMs?
- The "beta vs alpha" problem: when an ETF drops because the market dropped, vs when
  something specific happened to that sector
- Feature injection into LLM prompts — structured data + unstructured analysis

#### 5. Industrial vs Academic vs Retail Landscape
- **Industrial scale** (Renaissance, Two Sigma, Citadel, DE Shaw, Goldman Sachs):
  What can we infer about their use of NLP/AI? Job postings, conference talks, patent
  filings, public interviews. How do they use NLP — as primary signals or supplementary?
- **Academic:** Key research groups working on LLM+finance. Which universities/labs are
  publishing the most relevant work?
- **Smaller-scale / retail:** Open-source projects, blog posts, YouTube tutorials that
  combine LLMs with systematic trading. What is the quality level? Are people reporting
  real results or just demos?
- **Honest assessment:** Is an individual/small team likely to find genuine alpha here,
  or is this primarily a learning exercise that institutional players have already
  arbitraged away?

#### 6. Open Problems and Frontier
- What are the unsolved/hard problems in this space?
- Where is the research heading (2024–2026 trends)?
- What would a "state of the art" system for this use case look like today?
- Are there commercial products (not hedge funds, but tools/platforms) that do what I'm
  building?

### Output Format Requested

Please structure your response as:

1. **Executive Summary** (1 paragraph): What's the bottom line?
2. **Per-section findings** (the 6 sections above), each with:
   - Key papers/sources (with links/citations)
   - Main findings relevant to my project
   - Specific takeaways for my system
3. **Gap Analysis**: What is my project doing that others haven't? What am I missing
   that others do?
4. **Recommended Reading List**: Top 10 papers/resources I should read in priority order
5. **Honest Assessment**: Given what the literature shows, what is the realistic value
   of what I'm building?

Focus on work from 2022–2026. Older work is fine for foundational concepts (mean-reversion,
factor models) but I especially want recent LLM-specific research.
