
# sae.projection

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/sae.projection)](https://CRAN.R-project.org/package=sae.projection)
[![R-CMD-check](https://github.com/Alfrzlp/sae.projection/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/Alfrzlp/sae.projection/actions/workflows/R-CMD-check.yaml)

<!-- badges: end -->

## Author

Azka Ubaidillah, Ridson Al Farizal P

## Maintainer

Ridson Al Farizal P <ridsonap@bps.go.id>

## Description

The goal of **sae.projection** is to provide tools for performing
**Small Area Estimation (SAE)** using projection-based methods. This
approach is particularly effective for generating reliable domain-level
estimates in situations where sample sizes are too small for direct
estimation. By utilizing auxiliary information and statistical models,
the package improves the precision and accuracy of small area estimates.

## Installation

You can install the development version of sae.projection from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("Alfrzlp/sae.projection")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(sae.projection)
library(dplyr)
#> Warning: package 'dplyr' was built under R version 4.2.3
#> 
#> Attaching package: 'dplyr'
#> The following objects are masked from 'package:stats':
#> 
#>     filter, lag
#> The following objects are masked from 'package:base':
#> 
#>     intersect, setdiff, setequal, union
```

### Logistic Model

``` r
df_res <- projection_estimator(
   formula = unemployment ~ edu + sex + age + head,
   domain = c('prov', 'regency'),
   weight = 'weight',
   data_model = df_svy2,
   data_proj = df_svy1,
   model = 'logistic',
   cutoff = 'optimum'
)
#> ℹ Getting the optimum cut-off...
#> ✔ The optimum cut-off is 0.105987114406781 and 0.30650406504065
#> Warning in predict_result > cutoff: longer object length is not a multiple of
#> shorter object length
#> # A tibble: 119 × 6
#>     prov regency     n   yhat  var_yhat   rse
#>    <int>   <int> <int>  <dbl>     <dbl> <dbl>
#>  1    31       1   329 0.0633 0.000199  22.3 
#>  2    31      71  1229 0.0750 0.0000586 10.2 
#>  3    31      72  1368 0.0725 0.0000522  9.97
#>  4    31      73   846 0.0800 0.0000837 11.4 
#>  5    31      74  1265 0.109  0.0000714  7.74
#>  6    31      75  1126 0.0941 0.0000633  8.46
#>  7    32       1  1476 0.115  0.0000597  6.71
#>  8    32       2  1224 0.0899 0.0000585  8.50
#>  9    32       3  1248 0.0841 0.0000508  8.48
#> 10    32       4  1353 0.0854 0.0000557  8.73
#> # ℹ 109 more rows
```

### Linear Model

``` r
df_svy2_new <- df_svy2 %>% dplyr::filter(!is.na(income))

m2 <- projection_estimator(
  formula = income ~ edu + sex + age + head,
  domain = c('prov', 'regency'),
  weight = 'weight',
  data_model = df_svy2_new,
  data_proj = df_svy1,
  model = 'linear'
)
#> # A tibble: 119 × 6
#>     prov regency     n     yhat    var_yhat   rse
#>    <int>   <int> <int>    <dbl>       <dbl> <dbl>
#>  1    31       1   329 2579381. 4931305728.  2.72
#>  2    31      71  1229 3202174. 1814357196.  1.33
#>  3    31      72  1368 3158206. 1516683859.  1.23
#>  4    31      73   846 2958848. 2221792017.  1.59
#>  5    31      74  1265 2749878. 1210877276.  1.27
#>  6    31      75  1126 2965990. 1785753160.  1.42
#>  7    32       1  1476 2414944.  876222281.  1.23
#>  8    32       2  1224 2075254.  772421710.  1.34
#>  9    32       3  1248 1996901.  676199395.  1.30
#> 10    32       4  1353 2567675. 1073105001.  1.28
#> # ℹ 109 more rows
```
