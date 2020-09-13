# AB Testing
---------------
## Process of AB Testing
* Design
 * Understand Problem & Objectives
 * Come Up with Hypothesis
 * Design of Experiment
* Implement
 * Code change & Testing
 * Run Experiment & Monitor
* Measurement
 * Result Measurement
 * Data Analysis
 * Decision Making
---------------
## Design of Experiment
### Principles of Experiment Design
* **Independent** samples
* **Block** what you can control
* **Randomize** what you cannot control

### Assignment Unit
* Most common 50/50
* Sometimes not
  * Time sensitive

### A/A Testing
*  use A/B test framework to test two identical versions against each other. There should be no difference between the two groups.

### Exposure and Duration
* Minimum sample size
* Daily volume & Exposure
* Seasonality
-----------------
## Power Analysis and Sample Size
*Type I: Reject H0 when H0 is true
*Type II: Do not Reject HO when H0 is wrong
*Power: Reject H0 when H0 is wrong

### Power changes if factors increase
* Variance (decrease)
* Significance level (decrease)
* Size of the effect (increase)

### Sample Size Calculation
* Set the Power and Significance level
-------------------
## Design of Experiment
### Peeking
* Peeking increase false positive

https://www.evanmiller.org/how-not-to-run-an-ab-test.html

### Monitoring
* should NOT frequently check result
* Should NOT stop once result turns significant

### Problems and Solutions
1. What if it takes too long to get desired sample size?
* Increase Exposure
* Reduce variance to reduce required sample size
  * Blocking-run Experiment with sub-groups
  * Propensity Score Matching

  #### Propensity Score Matching
  1. Run a model to predict Y (CTR rate) with appropriate covariates Obtain propensity score: predicted y_hat
  2. Check that propensity score is balanced across test and control groups
  3. Match each test unit to one or more controls on propensity score:
    * Nearest neighbor matching
    * Matching with certain width
    * [Intern Project](https://datnerde.github.io/2019-08-22-balance/)

2. What if your data is highly skewed or statistics is hard to approximate with CLT?
* Tranformation
* [Winsorization](https://en.wikipedia.org/wiki/Winsorizing#:~:text=Winsorizing%20or%20winsorization%20is%20the,as%20clipping%20in%20signal%20processing.)/Capping
* Bootstrap
  * Pros
    * No assumptions no distribution
    * simple
    * can be used for all statistics
  * Cons
    * Computational expensive

  #### Bootstrap
  1. Randomly generate a sample of size n with replacement from the original data. n is the # of observations in original data
  2. Repeat step 1 many times
  3. Estimate statistics with sampling statistics of the generated samples

## Result Measurement
* Data Exploration
  * Check % of test/control with DOE
  * Mixed Assignment
  * Sanity check
* Set up the right Testing
  * Mostly use T-test
  * When variance is know is large, use Z-Test
  * Sample size small use non-parametric methods
  * For complicated statistics, Bootstrap to calculate p-value
  * [A Summary for all tests](https://stats.idre.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/)
* Decision Making
  * If results are neural
    * Slice/Dice on sub-groups
  * If results are positive/negative
    * check expectation
    * find causes
* Multiple Testing
  * Bonferroni Adjustment
  * Benjamini-Hochberg Adjustment
* Pre-bias Adjustment
  * Regression Adjustment
* Cohort Analysis
  * Measure Impact over time
* Multi-arm Bandit Problem

## Limitations of AB Testing
* Highly rely on your hypothesis
* Good for optimize small changes, Not good for innovative changes, long term strategies
* Other factors involved: e.g. learning effect, network effect
