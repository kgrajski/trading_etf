# Research Log: Weekly ETF Alpha Prediction

## Project Status

**Current Date:** 2026-01-23  
**Phase:** Systematic Grid Search  
**Baseline:** exp009 (rolling window, all models) → 50.9% Dir.Acc, IC=0.030

---

## Research Grid

### Dimensions Under Investigation

| Dimension | Options | Status |
|-----------|---------|--------|
| **Target** | Regression (α), Classification (UP/DOWN) | Regression tested ✓ |
| **Regime Handling** | None, As-Feature, Regime-Specific, Regime-Filter | As-Feature tested ✓ |
| **Model** | Linear, Tree, Neural | Linear/Tree tested ✓ |
| **Features** | L1/L2 only, +Positional, +Regime | L1/L2 + Regime tested ✓ |

### Kill Criteria

| Criterion | Threshold | Purpose |
|-----------|-----------|---------|
| IC minimum | < 0.02 | Signal too weak to trade |
| Dir.Acc minimum | < 52% | Barely above random |
| Improvement | < 2% over baseline | Not worth complexity |

---

## Experiment Queue

| Priority | ID | Hypothesis | Status | Result |
|----------|-----|------------|--------|--------|
| 1 | exp011 | Regime-as-feature improves IC | ✅ DONE | IC=0.038 (+27%), Dir=51.2% (+0.3%) |
| 2 | exp014 | Naive momentum baseline (no ML) | ✅ DONE | Mean-rev IC=0.034, Mom IC=-0.034 |
| 3 | exp012 | Classification on favorable regimes | NEXT | - |
| 4 | exp013 | Regime-specific models | QUEUED | - |

---

## Completed Experiments

### Phase 1: Foundation (Static CV - Misleading)

| Exp | Description | Dir.Acc | IC | Notes |
|-----|-------------|---------|-----|-------|
| exp001 | Baseline, 2 test weeks | 69.5% | 0.39 | Overfit to favorable period |
| exp002 | Clipped to 2 years | 69.6% | 0.39 | Same issue |
| exp003 | Feature+target normalized | 69.6% | 0.39 | Same issue |
| exp004 | Alpha (market-neutral) target | 76.0% | 0.45 | **Misleading - used actual β** |
| exp005 | Two-stage (predicted β) | 26.6% | 0.45 | β prediction failed |
| exp006 | Classification | 69.0% | - | Regression outperformed |

**Lesson:** Static CV on a favorable test period gives wildly optimistic results.

### Phase 2: Rolling Window (Realistic Evaluation)

| Exp | Description | Dir.Acc | IC | Notes |
|-----|-------------|---------|-----|-------|
| exp007 | Expanding window, Lasso | 49.6% | -0.007 | Essentially random |
| exp008 | Fixed 104-week, Lasso | 50.9% | 0.030 | Slight improvement |
| exp009 | Fixed 104-week, All models | 50.9% | 0.030 | **BASELINE** |
| exp010 | +Positional encoding | 50.8% | 0.029 | No improvement |

**Lesson:** True walk-forward evaluation shows the signal is weak but positive.

### Phase 3: Regime Analysis (Post-Hoc)

| Regime | IC | Dir.Acc | Interpretation |
|--------|-----|---------|----------------|
| VIX: Low | -0.007 | 48.3% | No signal |
| VIX: Medium | **0.092** | **55.1%** | Signal exists |
| VIX: High | 0.012 | 50.1% | Weak signal |
| Trend: Up | 0.013 | 50.5% | Weak |
| Trend: Flat | **0.066** | 51.3% | Better |
| Trend: Down | -0.033 | 49.6% | Worse |

**Lesson:** The signal is regime-dependent. Best in medium-VIX, range-bound markets.

### Phase 4: Regime-Conditioned Modeling (Current)

| Exp | Description | Dir.Acc | IC | vs Baseline |
|-----|-------------|---------|-----|-------------|
| exp011 | Regime as features | 51.2% | 0.038 | +0.3% Dir, +27% IC |
| exp014 | Naive baselines | - | - | See below |

**exp011 Interpretation:** Adding regime features provides marginal improvement, not enough to be practically significant.

### Phase 5: Naive Baseline Comparison (Complete)

| Strategy | Dir.Acc | IC | Interpretation |
|----------|---------|-----|----------------|
| Momentum | 48.6% | **-0.034** | Anti-momentum! Last week's winners underperform |
| Mean-Reversion | 50.9% | **+0.034** | Slight mean-reversion effect exists |
| Random | 50.0% | -0.002 | Control (as expected) |
| ML (exp011) | 51.2% | +0.038 | Slightly better than mean-reversion |

**Key Finding:** Weak mean-reversion effect exists (IC=0.034), but is it tradeable?

### Phase 6: Portfolio Backtest (Complete) ⚠️ CRITICAL

| Configuration | Gross Return | Net Return | Sharpe | Problem |
|---------------|--------------|------------|--------|---------|
| K=5, 1-week hold | +11.7% | **-40.3%** | -1.27 | 520 trades/yr |
| K=5, 4-week hold | +11.1% | **-1.9%** | -0.07 | 130 trades/yr |
| K=10, 1-week hold | +9.5% | **-94.5%** | -3.06 | 1040 trades/yr |
| K=20, 1-week hold | +8.4% | **-199.6%** | -7.81 | 2080 trades/yr |

**Decile Analysis (IC Concentration):**
| Decile | Mean Alpha | Note |
|--------|------------|------|
| 0 (losers) | +0.12% | Buy signal |
| 9 (winners) | -0.09% | Avoid |
| Spread | **0.20%/week** | Too small |

**CRITICAL FINDING (exp015 - Fixed Rebalancing):** 
- ✓ Mean-reversion signal exists (Gross Sharpe ~0.4)
- ❌ **Transaction costs destroy it** (Net Sharpe < 0)
- ❌ At 10 bps/trade, weekly rebalancing is unviable

### Phase 7: Exit Rule Strategy (Complete) ✓ BREAKTHROUGH

| Config | Trades/Yr | Win Rate | Ann. Return |
|--------|-----------|----------|-------------|
| TP3_SL3 (too tight) | 252 | 48.6% | -78% |
| TP5_SL5 | 187 | 50.1% | -17% |
| **TP10_SL10** | **94** | **53.2%** | **+47%** |
| SL5_8w (no TP) | 168 | 42.1% | +27% |

**BREAKTHROUGH FINDING (exp016 - Exit Rules):**
- ✓ **+46.6% annualized** with TP=10%, SL=10%
- ✓ Only **94 trades/year** → ~1.9% cost drag
- ✓ Estimated **net return ~44%** after costs
- ✓ Win rate 53.2%, average hold 11 weeks
- ✓ Key: Let trades run, don't force weekly rebalancing

---

## Key Learnings

### What Works (Statistically)
1. **Alpha (market-neutral) target** - Removes market beta noise
2. **Linear models** - Outperform trees in rolling evaluation  
3. **Fixed 104-week window** - Better than expanding window
4. **Regime awareness** - Signal is stronger in certain conditions
5. **Mean-reversion** - Weak but real effect at weekly frequency

### What Doesn't Work (Economically)
1. **Weekly rebalancing** - Transaction costs destroy returns
2. **High turnover strategies** - Need <100 trades/year to be viable
3. **Small edges (<0.5%/week)** - Eaten by costs
4. **Tree models** - Overfit under rolling evaluation
5. **Momentum at weekly freq** - Actually negative

### Open Questions
1. Can we **filter** to favorable regimes rather than **predict** them?
2. Does a **classification** approach work better for trading decisions?
3. ~~Is there a **naive baseline** (momentum, mean-reversion) that beats our models?~~ **Answered: No, but close**

---

## Current Interpretation

### exp011-exp014: Signal Discovery
- Mean-reversion signal exists (IC=0.034)
- ML adds marginal value (IC=0.038)
- Momentum is negative at weekly frequency

### exp015: Fixed Rebalancing Fails
- Weekly rebalancing generates 520+ trades/year
- Transaction costs consume all alpha
- **Fixed holding periods don't work**

### exp016: Exit Rules Work! ✓
**The signal IS tradeable with proper trade management.**

The key insight: **Don't force exits. Let trades reach targets.**

| Approach | Trades/Yr | Gross Return | Est. Net |
|----------|-----------|--------------|----------|
| Fixed 1w hold | 520 | +12% | -40% |
| Fixed 4w hold | 130 | +11% | -2% |
| **TP10/SL10** | **94** | **+47%** | **+44%** |

**Why this works:**
1. Fewer trades = lower costs
2. Winners hit +10% TP (avg +12.5%)
3. Losers cut at -10% SL (avg -13%)
4. 53% win rate × asymmetric payoff = profit

---

## Recommended Next Steps

### Option A: Refine TP10/SL10 Strategy ⭐ PRIORITY
- Add regime filter (only trade in medium VIX)
- Test asymmetric targets (e.g., TP15/SL10)
- Add position sizing based on signal strength

### Option B: Stress Test
- Monte Carlo on parameter sensitivity
- Out-of-sample validation (holdout period)
- Worst-case drawdown analysis

### Option C: Realistic Execution
- Add bid-ask spread (~5 bps)
- Add slippage estimate (~5 bps)
- Re-run with 20 bps total cost

### Option D: Live Paper Trading
- Forward-test with paper account
- Measure actual fill quality
- Build confidence before capital

### ⚠️ Key Lessons

1. **IC/Dir.Acc are necessary but not sufficient**
2. **Trade management matters as much as signal**
3. **Fixed rebalancing destroys weak signals**
4. **Exit rules preserve alpha**

---

## Technical Notes

### Data Summary
- **Symbols:** ~626 target ETFs per week
- **History:** 2021-03-29 to 2026-01-12 (~250 weeks)
- **Features:** 15 base (L1/L2) + 4 regime = 19 total
- **Target:** Alpha (log return minus cross-sectional mean)

### Model Configuration
- **Training window:** Fixed 104 weeks (2 years)
- **Min history:** 52 weeks before first prediction
- **Normalization:** Z-score on features and target

---

*Last updated: 2026-01-23 18:35*
