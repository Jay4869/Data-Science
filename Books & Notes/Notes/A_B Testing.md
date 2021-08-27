# A/B Testing

## Four Principles
1. Assessing Risk
2. Benefits
3. Alternatives
4. Privacy 

User flow: Customer funnel
* Homepage visits
* Exploring the site
* Creat accounts
* Complete


## Experiment
1. Hypothesis: changing the "Start Now" button from orange to pink will incrase how many students explore Udacity courses.
2. Define Metrics: messure users pregressing to the second level of funnel
    * [ ] Total number of courses completed: taking too much time to collect results
    * [ ] Number of clicks: large population has more people clicks
    * [ ] Click-Through-Rate (CTR) = number of clicks / Number of pagereviews: messure the user ability
    * [x] Click-Through-Probability (CTP) = unique number of users who click for each time interval / unique users who visit page: messure the total impact
3. Build Metric Intuition
    * [x] Def 1: number of pageviews click for each time interval / total number of  pageviews
    * [x] Def 2: Cookie-Probability = # cookies click for each time interval / total number of cookies
4. Estimating the probility and confidence interval of event
5. Power Analysis to calculate the sample size: hypothesis testing, statistical significance, practical business significance (substantive), power.
6. Analyze the results: construct the confidence interval

## Statistical Distribution
As we know the random variables following the certain distribution, we can use the formula we have for sample standard error to estimate how variable we expect our overall probability and observe statistical significance (repeatability). However, from a businesss standpoint, it's practically significant, also statistically significant.

### Random Variables
Random variable is a variable whose possible values are outcomes of a random phenomenon.

* Continuous: any value within a range (uniform distribution, normal distribution)
* Discrete: certain values (binomial distribution, poisson distribution)

### Normal Distribution
The normal distribution is a probability function that describes how the values of a variable are distributed, also called as the Gaussian distribution. A probability distribution is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. Extreme values in both tails of the distribution are similarly unlikely.

* Bell curve
* Mean symmetric
* Area under curve (total probability) = 1
* Mean (position), SD (shape)
* Z-scores: map observations on the standard normal scale, also called standardization.

### Binomial Distribution
The discrete probability distribution of the number of success in sequence of N indepedent Bernoulli trials. Binomial distributions must also meet the following three criteria:

* The number of observations or trials is fixed. In other words, you can only figure out the probability of something happening if you do it a certain number of times.
* Each observation or trial is independent. In other words, none of your trials have an effect on the probability of the next trial.
* The probability of success (tails, heads, fail or pass) is exactly the same from one trial to another.
* Expectation: np, variance: npq

### Possion Distribution
The Poisson probability distribution is often used as a model of the number of arrivals at a facility within a given period of time.

## Statistical Inference
### Law of Large Numbers
An observed sample average from a large sample will be close to the true population average and that it will get closer the larger the sample.

### Central Limit Theorem
When we have a population with mean $\mu$ and standard deviation $\sigma$, and take sufficiently large random samples (n>=30) from the population with replacement, the distribution of sample means approximates a normal distribution $N(\mu, \frac{\sigma}{\sqrt{n}})$ as the sample size gets larger.

### Type I error ($\alpha$)
False positive: rejects a null hypothesis that is actually true. A % chance finding a difference but not a true difference. The larger value is, the less **reliable** test is. It's very common in the A/B test to show the observe difference between two groups but no difference in reality.

Ex: A person is not infected by COVID, and the test result is positive.

### Type II error ($\beta$)
False negative, is a failure to reject a null hypothesis that is actually false. A % chance not detecting a difference but difference actually exists. The larger value is, the less **ability** test is, which means we need to minimize the type II error of the test. It's very common in the A/B test to show we are not observing difference between two groups but there is truly difference in reality.

Ex: A person is infected by COVID, and the test result is negative.

### P Value
A conditional probability of extreme observed cases occurred when the null hypothesis is true. Simple words, the p-value is the evidence against a null hypothesis. The lower the p-value, the stronger evidence is, the more likely reject the null hypothesis.

### Confidence Interval
The purpose is because true parameter is unknow, and how can we estimate it based on sample observerations. A range of values tells us how often it would contain true parameter. The probability of true parameter falls between the range, called confident level ($1 - \alpha$). 

It's computed by sample mean and margin of error, and measures the degree of certainty using a sampling method. The wider interval, the more uncertainty about the sample result. Margin of Error (M): a function of both the proportion of probility and the size of the sample.

* $M = Z * SE$
* $CI = [\hat{x} - M, \hat{x} + M]$
* Less data -> wider $CI$
* Higher confidence level -> wider $CI$

### Statistics Power (sensitive)
The probability that the test correctly rejects the null hypothesis. (The probability of a true positive result, $1 - \beta$). The higher the statistical power, better the test is. The most application is calculating the mini sample size when we are designing a experiment.

![](https://i.imgur.com/jBOsV4R.png)

### Power Analysis
Decide how many sample size we need in orde to have enough power to conclude with high probability that the result is statistically significant. An trade-off is between power and sample size.
* The smaller the change that you want to detect, or increase confidence, you will need larger sample size.
* For small samples, $\alpha$ is low (unlikely to launch a bad experiemnt), but $\beta$ is high (likely to fail detect the difference).
* However, the larger samples reduce the standard deviation of distribution, so the power has increased.

To calculate the sample size:
* Minimum Detectable Effect (MDE): relative minimal improvement over the baseline. The samller MDE requires more data to detect a difference.
* Population Std: The higher standard deviation, the more data are needed to demonstrate a difference. 
* Significance level: % chance of finding a random difference
* Type-II-error: % chance that a significant difference is missed. 

$$
n = \frac{2\sigma^2(Z_\beta+Z_{\alpha/2})^2}{MDE^2}
$$

## Randomized Experiment
Randomized experiments allow to statistically estimate treatment effects on a particular outcome of interest. Randomization involves randomly allocating the experimental units across the treatment groups that no other independent variables exist.

### Purpose of Randomization
* Reduce experimental bias by controlling all lurking variables.
* Control the explanatory variable to establish a causal and effect relationship
* Produce ignorable designs, which are valuable in model-based statistical inference (Bayesian or likelihood-based)

Note: Ignorability is a feature of an experiment design whereby the method of data collection (and the nature of missing data) do not depend on the missing data.

### Running Time
The business time to run an experiment may be limited for operational reasons:
* Low Traffic: limited amount of traffic to an experiment
* Time: get results quickly due to operational pressures
* Value: won’t run the experiment unless you can prove that it provides a certain amount of value

To estimate how long a given experiment will need to run to achieve statistical significance:
* Traffic allocated for the experiment: what percentage of your traffic (or unique weekly visitors) will you allocate for this experiment
* Total sample size: estimate the sample size per variation, depending on the baseline conversion rate and the MDE
* Running time: divide total sample size by traffic allocated to the experiment
![Running Time](https://i.imgur.com/kInEdmg.png)

Another option is using a range of MDEs to get a feel for how long willing to invest into each experiment. The baseline, number of variations, number of unique visitors, and statistical significance are constant. Therefore, we can plot the time it takes to run this experiment as a function of the MDE, and then use MDEs to make informed business decisions.
![](https://i.imgur.com/mmcGMKV.png)


### Multiple Testing Problem
When we are testing multiple tests at the same time, the result will be bias due to false positive. The probability of at least one false positive is larger than our significant level.
* Bonferroni Correction
    * significant level / number of test
    * too conservative
* Control False Discovery Rate (FDR)
    * FDR = E[false postive / rejections]
    * only apply a large number of tests


### Novelty & Primacy Effect
The problem is occurred that after launch a new features to all users for weeks, the treatment effect quickly declined. The answer is the novelty effect is small that repeating usage effect is small, so we observe a declining.
* Primacy effect (change aversion): reluctant to change
* Novelty effect: willing the change 
* Both effects not last longer

The idea behind is rule out the possibility that run tests only on the first time users. However, if the test is alreayd running, we can compare the first time users to old users in the treatment group to estimate the novelty effect.


### Interference Between Variants
* Social Network effect
    * User behaviors are impacted by others
    * Effect can spillover the control group
    * Observed difference underestimates the treatment effect
* Two-sided Markets
    * Resources are shared/compete among control and treatment groups
    * Observed difference overestimates the treatment effect

Solution: isolate users, using different keys to do sampling.
* Geo-based Randomization
    * unique market will have a large variance 
* Time-based Randomization
    * select a random time to run the experiment
    * only work when treatment effect is in short time
* Cluster-based Randomization
    * people interact mostly within the cluster
    * assign clusters randomly
* Ego-network Randomization
    * A cluster is composed of an "ego" and "alters"
    * One-out network effect: user either has the feature or not


## Others
### Two Sample Test
We need to choose a standard error that gives us a good comparison of both: pooled standard error.

1. $\hat{P}_{pool} = \frac{X_{contr}+X_{exp}}{N_{contr}+{N_{exp}}}$
2. $SE_{pool}$
3. $\hat{d} = \hat{P}_{exp} - \hat{P}_{contr}$
4. $H_0: \hat{d}=0, \hat{d} \sim N(0, SE_{pool})$
5. $\hat{d} > |Z*SE_{pool}| \rightarrow$ Reject $H_0$ 

### Decision Making
1. Confidence interval is greater than the practical signifiance boundary.
2. Neutral case is no significant change from 0 since confidence interval includes 0, and not a practically significant change.
3. The result is statistically significant, but not practical significant
4. Not have enough power to draw the conclusion.


![](https://i.imgur.com/Aj3yhPM.png)


### Metric Design
Two main use cases
1. Invariant (sanity) checking: the metrics stay same across experiments
    * population
    * distribution
2. Evaluation
    * overall business
    * detailed business metrics, user behavior
3. Composite metric
    * objective function
    * overall evaluation criterion
    

### User Experience Research
conduct survey analysis or lab researches to show the correlations (actually caused what happened), not causation because not running a parallel experiment to get a temporal effect.

* User experience research: deeply study users, but small group of participants
* Focus groups: more participants but less deep to learn the participants, and run the risk of group think convergence on fewer opinions.
* Survey: useful for gathering metrics you can't directly measure self with large group of participants, but survey results may not be truth, so never compare the survey results directly. 

### Validate Data
Filtering data helps de-biase the results of experiments. Computing metrics on a bunch of disjoint sets (slicing data) to compare results whether expect or not.

### Summary Metrics
Some metrics are directly data messurement per unit, so we need Summarize all of these individual events into a signle summary metric.
1. sensitivity: able to detect a change
    * A/A test: measure how sensitivity of metrics if users saw the same thing (no changes)
    * Observational analysis of logs
2. robustness: not impact on outliers
3. Distribution of metrics

There are four categories of metrics commonly to use:
1. sum and counts
2. distribution metrics: mean, percentile
3. probability and rate
4. ratio

### Variability

| Type of Metrics | Distribution | Estimated Variance |
| -------- | -------- | -------- |
| Probability  | Binomial (normal)  | $\hat{p}(1-\hat{p})/N$  |
| Mean  | normal  | $\hat{\sigma}^2/N$  |
| Median/Percentile  | Depends  | Depends  |
| Count/difference  | normal  | $\hat{\sigma_x}^2 +\hat{\sigma_y}^2$  |
| rate  | poisson  | $\bar{x}$  |
| ratio  | Depends  | Depends  |


### Analytical Estimate vs. Empirical Estimate
1. if statistical distribution of metrics is normal, then we can compute a normal conference interval with estimated variance.
2. if not, compute a non-parametric conference interval to get a robust results.
3. Using A/A test to estimate the empirical variability of metrics. The purpose of A/A test is that any differences you measure are due to the underlying variability (mutiple A/A tests, boostrap)
    * compare the analytical results with A/A test to what you expect (sanity check)
    * estimate the variance empirically, then use the assumption of distribution to calculate the confidence interval.
    * directly estimate confidence interval.
    
Example: CTR = clicks / page views 
unit of analysis: page views
unit of diversion: page views (event-based), so analytical variability = empireical variability. However, unit of diversion: cookie or user_id (user-based), analytical variability > empirical variabilty

Reason: Independent assumption
* E vent-based diversion: every single event is a random draw, so independent assumption is valid
* User-based diversion: people have correlation between each others, so independent is not hold and cause large variability (standard error).


### Unit of Diversion
* Event: any changes users won't notice such as loading time, search results.
* Cookie: any changes to keep same with same devices such as button color and size.
* User-id: any changes users certainly notice, so it's based on single user.

### Cohort
Cohort usually means that a sub-set of population and only look users who entered your experiment on both sides shared the same experience (time, location), except receiving changes you made.

Cohort is hard to analyze and require more data. Typically, you only want to use them when you're looking for user stability. More generally, using cohort meassures a real effect on user behavior relateive to history.
* Looking for learning effects
* Examining user retention
* Increasing user activity/usage

### Size & Duration


## Learning Effect
Basically when you want to measure user learning, or effectively whether a user is adapting to a change or not. The key issue with trying to measure a learning effect is time. One risk is test size since learning effect is require a long period of time to measure. Another risk is uncertainity about what the effect of change is going to be. 

Solution:
* running through a small proportion of users for a longer period of time.
* Pre-period and post-period: A/A -> A/B -> A/A, the differences found from post-period can be attributed to user learning effect in the experiment period.