# How to choose the right statistical test

## Q1: Is there a difference between groups that are unpaired?
* Unpaired: there is no possibility of the values in one data set being related to or being influenced by the values in the other data sets
* Data Type: Numerical or Categorical data
### Numerical data
#### Follow the Parameters of the normal distribution curve?
* If yes, use parametric
* If no or not sure, use non-parametric
#### Multiple group comparison
* Determine difference when significant

<img src="assets/Pick the Right Test-f6ed9220.jpg"/>

## Q2: Is there a difference between groups which are paired?
* Pairing
  * Data sets are derived by repeated measurements (e.g. before-after measurements or multiple measurements across time) on the same set of subjects.
  * If subject groups are different but values in one group are in some way linked or related to values in the other group (e.g. twin studies, sibling studies, parent-offspring studies)

<img alt="Pick the Right Test-99ca4ebf.jpg" src="assets/Pick the Right Test-99ca4ebf.jpg" width="" height="" >

## Q3: Is there any association between variables?
<img alt="Pick the Right Test-54aed4ac.jpg" src="assets/Pick the Right Test-54aed4ac.jpg" width="" height="" >

## Q4: Is there agreement between data sets?

<img alt="Pick the Right Test-db11f0d0.jpg" src="assets/Pick the Right Test-db11f0d0.jpg" width="" height="" >

## Q5: Is there a difference between time-to-event trends or survival plots?

* This question is specific to survival analysis.
* A sizeable proportion of the original study subjects may not reach the endpoint in question by the time the study ends
* If there are two groups then the applicable tests are Cox-Mantel test, Gehan’s (generalized Wilcoxon) test or log-rank test.
* In case of more than two groups Peto and Peto’s test or log-rank test can be applied to look for significant difference between time-to-event trends.

## Miscellanies
* [χ2  TEST of PROPORTIONS IS IDENTICAL TO Z-TEST SQUARED](http://rinterested.github.io/statistics/chi_square_same_as_z_test.html)

* [Detailed Examples of Test](https://stats.idre.ucla.edu/spss/whatstat/what-statistical-analysis-should-i-usestatistical-analyses-using-spss/)
