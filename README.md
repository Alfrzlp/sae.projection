
# sae.projection

<!-- badges: start -->

[![CRAN
status](https://www.r-pkg.org/badges/version/sae.projection)](https://CRAN.R-project.org/package=sae.projection)
[![R-CMD-check](https://github.com/Alfrzlp/sae.projection/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/Alfrzlp/sae.projection/actions/workflows/R-CMD-check.yaml)

<!-- badges: end -->

## Author

Azka Ubaidillah, Ridson Al Farizal P, Silvi Ajeng Larasati, Amelia
Rahayu

## Maintainer

Ridson Al Farizal P <ridsonap@bps.go.id>

## Description

The **sae.projection** package provides a robust tool for *small area
estimation using a projection-based approach*. This method is
particularly beneficial in scenarios involving two surveys, the first
survey collects data solely on auxiliary variables, while the second,
typically smaller survey, collects both the variables of interest and
the auxiliary variables. The package constructs a working model to
predict the variables of interest for each sample in the first survey.
These predictions are then used to estimate relevant indicators for the
desired domains. This condition overcomes the problem of estimation in a
small area when only using the second survey data.

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
#> Loading required package: tidymodels
#> ── Attaching packages ────────────────────────────────────── tidymodels 1.2.0 ──
#> ✔ broom        1.0.7     ✔ recipes      1.1.0
#> ✔ dials        1.3.0     ✔ rsample      1.2.1
#> ✔ dplyr        1.1.4     ✔ tibble       3.2.1
#> ✔ ggplot2      3.5.1     ✔ tidyr        1.3.1
#> ✔ infer        1.0.7     ✔ tune         1.2.1
#> ✔ modeldata    1.4.0     ✔ workflows    1.1.4
#> ✔ parsnip      1.2.1     ✔ workflowsets 1.1.0
#> ✔ purrr        1.0.2     ✔ yardstick    1.3.1
#> Warning: package 'scales' was built under R version 4.4.3
#> ── Conflicts ───────────────────────────────────────── tidymodels_conflicts() ──
#> ✖ purrr::discard() masks scales::discard()
#> ✖ dplyr::filter()  masks stats::filter()
#> ✖ dplyr::lag()     masks stats::lag()
#> ✖ recipes::step()  masks stats::step()
#> • Search for functions across packages at https://www.tidymodels.org/find/
library(dplyr)
```

## Regression

### Data

``` r
df_svy22_income <- df_svy22 %>% filter(!is.na(income))
df_svy23_income <- df_svy23 %>% filter(!is.na(income))
```

### Linear Regression Model

``` r
lm_proj <- projection(
   income ~ age + sex + edu + disability,
   id = 'PSU', weight = 'WEIGHT', strata = 'STRATA',
   domain = c('PROV', 'REGENCY'),
   model = linear_reg(),
   data_model = df_svy22_income,
   data_proj = df_svy23_income,
   nest = TRUE
)

lm_proj$projection
#>    PROV REGENCY     ypr      var_ypr   rse_ypr
#> 1    40       1 2355859 112850851251 14.259463
#> 2    40       2 2155687  14070372903  5.502588
#> 3    40       3 2088804   9711460386  4.717855
#> 4    40       4 2188828  12511600884  5.110281
#> 5    40       5 2138723  22768231513  7.055213
#> 6    40       6 2061848  10210621315  4.900827
#> 7    40       7 2085265   9976295858  4.789866
#> 8    40       8 2028248   6536668678  3.986183
#> 9    40       9 2233891  12561471412  5.017163
#> 10   40      10 2997192 470694250001 22.890476
#> 11   40      11 2061056  10060073816  4.866433
#> 12   40      12 2201031   9581460318  4.447231
#> 13   40      13 2213420  32332172691  8.123695
#> 14   40      14 2200890  13896011417  5.356075
#> 15   40      15 2269106  15294166332  5.450146
#> 16   40      16 2174782  12143176114  5.066994
#> 17   40      17 1965337  10785195069  5.284172
#> 18   40      18 1837433   8405642799  4.989694
#> 19   40      19 2208518  24478147649  7.084160
#> 20   40      20 2143794  13056774386  5.330094
#> 21   40      21 2067066   7984452688  4.322831
#> 22   40      22 2038540   9802389037  4.856761
#> 23   40      23 2393420  86969130589 12.321503
#> 24   40      24 2240375  20747721026  6.429313
#> 25   40      25 2195086  14570971410  5.499113
#> 26   40      26 2187382  24209446221  7.113247
#> 27   40      27 2217716  64407458495 11.443585
#> 28   40      28 1986641   6921930259  4.187880
#> 29   40      29 1965301   5897858683  3.907672
#> 30   40      30 2256849  17114638305  5.796707
#> 31   40      31 2267756  24897436539  6.957945
#> 32   40      32 2063165   8403451069  4.443189
#> 33   40      33 1982999   7567172983  4.386765
#> 34   40      34 2244054  21199661590  6.488302
#> 35   40      35 2137450  38140066571  9.136813
#> 36   40      36 2024760  10698498794  5.108435
#> 37   40      37 2055319  18930473132  6.694245
#> 38   40      38 2954545 314127308067 18.969774
#> 39   40      39 1955815  13455721680  5.930971
#> 40   40      40 1998518  23669511858  7.698154
#> 41   40      41 2387539  63038731173 10.516066
#> 42   40      42 2007678   8297754980  4.537183
#> 43   40      43 1983971   5647857868  3.787971
#> 44   40      44 2187359  41458503574  9.308653
#> 45   40      45 1986850  14801246261  6.123279
#> 46   40      46 2369022  48861397759  9.330694
#> 47   40      47 2116854   9339080301  4.565214
#> 48   40      48 2174280  19690945724  6.453833
#> 49   40      49 2239364  41590806935  9.106972
#> 50   40      50 2118603  13019855312  5.385840
#> 51   40      51 1927953   6732676376  4.255959
#> 52   40      52 1970130   7315602106  4.341404
#> 53   40      53 1954391   7289520282  4.368556
#> 54   40      54 2137444  15690069691  5.860272
#> 55   40      55 2017267   6428930853  3.974715
#> 56   40      56 2351050  50853300343  9.591748
#> 57   40      57 2162501  25747523120  7.420129
#> 58   40      58 2674530 313400377469 20.931591
#> 59   40      59 2335474  17471370430  5.659635
#> 60   40      60 2120891   9692679744  4.641983
#> 61   40      61 2098821  13706060975  5.578031
#> 62   40      62 1971450   9299512082  4.891525
#> 63   40      63 2209349  32676097067  8.181835
#> 64   40      64 1941592   6738894142  4.228014
#> 65   40      65 2070572   7158546619  4.086223
#> 66   40      66 2089372  17862942632  6.396771
#> 67   40      67 1900122   7741749874  4.630608
#> 68   40      68 1941208   6187014231  4.051992
#> 69   40      69 2279011  33956910517  8.085701
#> 70   40      70 2142969  14310388261  5.582257
#> 71   40      71 2341867  30248638021  7.426611
#> 72   40      72 1968031   7531981310  4.409839
#> 73   40      73 2333145  31242426098  7.575837
#> 74   40      74 2191866  15913071184  5.755235
#> 75   40      75 2390506  52976205496  9.628322
#> 76   40      76 1937820   6596212084  4.191157
#> 77   40      77 2057303  20972153977  7.039198
#> 78   40      78 2269991  34769289337  8.214362
#> 79   40      79 2166598  17221385107  6.056974
```

### Random Forest Regression with Hyperparameter Tunning

``` r
rf_proj <- projection(
  income ~ age + sex + edu + disability,
  id = 'PSU', weight = 'WEIGHT', strata = 'STRATA',
  domain = c('PROV', 'REGENCY'),
  model = rand_forest(mtry = tune(), trees = tune(), min_n = tune()),
  data_model = df_svy22_income,
  data_proj = df_svy23_income,
  kfold = 3,
  grid = 20
)

rf_proj$projection
```

## Classification

### Data

``` r
df_svy22_neet <- df_svy22 %>% 
  filter(between(age, 15, 24))
df_svy23_neet <- df_svy23 %>% 
  filter(between(age, 15, 24))
```

### Logistic Regression

``` r
lr_proj <- projection(
  formula = neet ~ sex + edu + disability,
  id = 'PSU',
  weight = 'WEIGHT',
  strata = 'STRATA',
  domain = c('PROV', 'REGENCY'),
  model = logistic_reg(),
  data_model = df_svy22_neet,
  data_proj = df_svy23_neet,
  nest = TRUE
)

lr_proj$projection
#>    PROV REGENCY       ypr     var_ypr  rse_ypr
#> 1    40       1 1.1183886 0.006444221 7.177819
#> 2    40       2 1.2557005 0.006105655 6.222717
#> 3    40       3 1.1274297 0.006065054 6.907611
#> 4    40       4 1.1113317 0.005828028 6.869373
#> 5    40       5 1.0800544 0.006010325 7.177998
#> 6    40       6 1.1517670 0.006139404 6.802969
#> 7    40       7 1.0794256 0.007007080 7.754894
#> 8    40       8 1.2443838 0.005809545 6.125150
#> 9    40       9 1.2047362 0.004965885 5.849333
#> 10   40      10 0.9640855 0.006984372 8.668583
#> 11   40      11 1.0933346 0.005826673 6.981636
#> 12   40      12 1.1801704 0.005316427 6.178246
#> 13   40      13 1.1846172 0.007200787 7.163280
#> 14   40      14 1.0845802 0.006903031 7.660520
#> 15   40      15 1.1505433 0.006664377 7.095399
#> 16   40      16 1.1486635 0.005427596 6.413734
#> 17   40      17 1.2093561 0.005661401 6.221681
#> 18   40      18 1.1028660 0.006612135 7.373066
#> 19   40      19 1.2294521 0.004804973 5.638114
#> 20   40      20 1.1578711 0.006367011 6.891402
#> 21   40      21 1.2762862 0.005070637 5.579345
#> 22   40      22 1.2102351 0.005540878 6.150629
#> 23   40      23 1.2059524 0.005400307 6.093672
#> 24   40      24 1.1783683 0.005738102 6.428402
#> 25   40      25 1.1080694 0.005155669 6.480008
#> 26   40      26 1.2509160 0.005157540 5.741074
#> 27   40      27 1.2471413 0.005605561 6.003353
#> 28   40      28 1.1753644 0.004616937 5.781020
#> 29   40      29 1.0877470 0.005710199 6.947007
#> 30   40      30 1.0443887 0.006598061 7.777607
#> 31   40      31 1.2117605 0.006043338 6.415369
#> 32   40      32 1.1688809 0.005417636 6.297014
#> 33   40      33 1.1884573 0.006069021 6.555046
#> 34   40      34 1.1248072 0.006022908 6.899618
#> 35   40      35 1.1672254 0.006522315 6.919045
#> 36   40      36 1.2252635 0.005834061 6.233845
#> 37   40      37 1.0434104 0.006060357 7.460947
#> 38   40      38 1.2600382 0.004835964 5.518967
#> 39   40      39 1.1189457 0.006639541 7.282157
#> 40   40      40 1.0950714 0.005302188 6.649441
#> 41   40      41 1.1898514 0.005923444 6.468363
#> 42   40      42 1.2487870 0.007639726 6.999234
#> 43   40      43 1.0982993 0.006348754 7.254768
#> 44   40      44 1.1818942 0.005524671 6.288899
#> 45   40      45 1.2766450 0.006564731 6.346559
#> 46   40      46 1.1639300 0.005717720 6.496577
#> 47   40      47 1.2213804 0.006462049 6.581641
#> 48   40      48 1.1945465 0.005625430 6.278773
#> 49   40      49 1.0425183 0.005626184 7.194876
#> 50   40      50 0.9518069 0.007261626 8.952989
#> 51   40      51 1.3295578 0.005528923 5.592591
#> 52   40      52 1.0509247 0.005438234 7.017095
#> 53   40      53 1.1885369 0.005763973 6.387756
#> 54   40      54 1.1534033 0.005731217 6.563602
#> 55   40      55 1.2341504 0.005928711 6.238958
#> 56   40      56 1.1968135 0.006673016 6.825502
#> 57   40      57 1.1810190 0.005667091 6.374164
#> 58   40      58 1.1206938 0.005539195 6.641044
#> 59   40      59 1.0108983 0.005954844 7.633571
#> 60   40      60 1.1346988 0.006363699 7.030306
#> 61   40      61 1.2145020 0.005925792 6.338332
#> 62   40      62 1.1014442 0.005650843 6.824866
#> 63   40      63 1.2185587 0.005020100 5.814464
#> 64   40      64 1.1159900 0.005480422 6.633561
#> 65   40      65 1.1039627 0.005699648 6.838638
#> 66   40      66 1.1283841 0.006518595 7.155172
#> 67   40      67 1.0386828 0.005776468 7.317256
#> 68   40      68 0.9854114 0.005870607 7.775421
#> 69   40      69 1.1610866 0.005587405 6.437844
#> 70   40      70 1.1255552 0.006023267 6.895238
#> 71   40      71 1.2091412 0.005741644 6.266732
#> 72   40      72 1.2197376 0.005334661 5.988072
#> 73   40      73 1.1532298 0.005458433 6.406460
#> 74   40      74 1.1886667 0.006554084 6.810765
#> 75   40      75 1.2285712 0.005805988 6.202085
#> 76   40      76 1.1290612 0.006445259 7.110542
#> 77   40      77 1.1334380 0.005447368 6.511718
#> 78   40      78 1.0864061 0.006680768 7.523519
#> 79   40      79 1.1952840 0.004899836 5.856251
```

### LightGBM with Hyperparameter Tunning

``` r
library(bonsai)
show_engines('boost_tree')
#> # A tibble: 7 × 2
#>   engine   mode          
#>   <chr>    <chr>         
#> 1 xgboost  classification
#> 2 xgboost  regression    
#> 3 C5.0     classification
#> 4 spark    classification
#> 5 spark    regression    
#> 6 lightgbm regression    
#> 7 lightgbm classification
lgbm_model <- boost_tree(
  mtry = tune(), trees = tune(), min_n = tune(), tree_depth = tune(), learn_rate = tune(),
  engine = 'lightgbm'
)
```

``` r
lgbm_proj <- projection(
  formula = neet ~ sex + edu + disability,
  id = 'PSU',
  weight = 'WEIGHT',
  strata = 'STRATA',
  domain = c('PROV', 'REGENCY'),
  model = lgbm_model,
  data_model = df_svy22_neet,
  data_proj = df_svy23_neet,
  kfold = 3, 
  grid = 20
)

lgbm_proj$projection
```

### Projection Estimator with Random Forest Algorithm

``` r
data(df_svy_A)
data(df_svy_B)

x_predictors <- names(df_svy_A)[7:32]
result <- projection_randomforest(
    data_model = df_svy_A,
    target_column = "Y",
    predictor_cols = x_predictors,
    data_proj = df_svy_B,
    domain1 = "province",
    domain2 = "regency",
    psu = "psu",
    ssu = "ssu",
    strata = "strata",
    weights = "weight",
    bias_correction = FALSE
)
#> [1] "Bias correction is disabled. Calculating indirect estimation without bias correction."
#> [1] "Starting preprocessing..."
#> [1] "Preprocessing completed. Starting data split..."
#> [1] "Data split completed. Starting RFE..."
#> Loading required package: lattice
#> 
#> Attaching package: 'caret'
#> The following objects are masked from 'package:yardstick':
#> 
#>     precision, recall, sensitivity, specificity
#> The following object is masked from 'package:purrr':
#> 
#>     lift
#> +(rfe) fit Fold1 size: 26 
#> -(rfe) fit Fold1 size: 26 
#> +(rfe) imp Fold1 
#> -(rfe) imp Fold1 
#> +(rfe) fit Fold1 size: 20 
#> -(rfe) fit Fold1 size: 20 
#> +(rfe) fit Fold1 size: 15 
#> -(rfe) fit Fold1 size: 15 
#> +(rfe) fit Fold1 size: 10 
#> -(rfe) fit Fold1 size: 10 
#> +(rfe) fit Fold1 size:  9 
#> -(rfe) fit Fold1 size:  9 
#> +(rfe) fit Fold1 size:  8 
#> -(rfe) fit Fold1 size:  8 
#> +(rfe) fit Fold1 size:  7 
#> -(rfe) fit Fold1 size:  7 
#> +(rfe) fit Fold1 size:  6 
#> -(rfe) fit Fold1 size:  6 
#> +(rfe) fit Fold1 size:  5 
#> -(rfe) fit Fold1 size:  5 
#> +(rfe) fit Fold1 size:  4 
#> -(rfe) fit Fold1 size:  4 
#> +(rfe) fit Fold1 size:  3 
#> -(rfe) fit Fold1 size:  3 
#> +(rfe) fit Fold1 size:  2 
#> -(rfe) fit Fold1 size:  2 
#> +(rfe) fit Fold1 size:  1 
#> -(rfe) fit Fold1 size:  1 
#> +(rfe) fit Fold2 size: 26 
#> -(rfe) fit Fold2 size: 26 
#> +(rfe) imp Fold2 
#> -(rfe) imp Fold2 
#> +(rfe) fit Fold2 size: 20 
#> -(rfe) fit Fold2 size: 20 
#> +(rfe) fit Fold2 size: 15 
#> -(rfe) fit Fold2 size: 15 
#> +(rfe) fit Fold2 size: 10 
#> -(rfe) fit Fold2 size: 10 
#> +(rfe) fit Fold2 size:  9 
#> -(rfe) fit Fold2 size:  9 
#> +(rfe) fit Fold2 size:  8 
#> -(rfe) fit Fold2 size:  8 
#> +(rfe) fit Fold2 size:  7 
#> -(rfe) fit Fold2 size:  7 
#> +(rfe) fit Fold2 size:  6 
#> -(rfe) fit Fold2 size:  6 
#> +(rfe) fit Fold2 size:  5 
#> -(rfe) fit Fold2 size:  5 
#> +(rfe) fit Fold2 size:  4 
#> -(rfe) fit Fold2 size:  4 
#> +(rfe) fit Fold2 size:  3 
#> -(rfe) fit Fold2 size:  3 
#> +(rfe) fit Fold2 size:  2 
#> -(rfe) fit Fold2 size:  2 
#> +(rfe) fit Fold2 size:  1 
#> -(rfe) fit Fold2 size:  1 
#> +(rfe) fit Fold3 size: 26 
#> -(rfe) fit Fold3 size: 26 
#> +(rfe) imp Fold3 
#> -(rfe) imp Fold3 
#> +(rfe) fit Fold3 size: 20 
#> -(rfe) fit Fold3 size: 20 
#> +(rfe) fit Fold3 size: 15 
#> -(rfe) fit Fold3 size: 15 
#> +(rfe) fit Fold3 size: 10 
#> -(rfe) fit Fold3 size: 10 
#> +(rfe) fit Fold3 size:  9 
#> -(rfe) fit Fold3 size:  9 
#> +(rfe) fit Fold3 size:  8 
#> -(rfe) fit Fold3 size:  8 
#> +(rfe) fit Fold3 size:  7 
#> -(rfe) fit Fold3 size:  7 
#> +(rfe) fit Fold3 size:  6 
#> -(rfe) fit Fold3 size:  6 
#> +(rfe) fit Fold3 size:  5 
#> -(rfe) fit Fold3 size:  5 
#> +(rfe) fit Fold3 size:  4 
#> -(rfe) fit Fold3 size:  4 
#> +(rfe) fit Fold3 size:  3 
#> -(rfe) fit Fold3 size:  3 
#> +(rfe) fit Fold3 size:  2 
#> -(rfe) fit Fold3 size:  2 
#> +(rfe) fit Fold3 size:  1 
#> -(rfe) fit Fold3 size:  1 
#> +(rfe) fit Fold4 size: 26 
#> -(rfe) fit Fold4 size: 26 
#> +(rfe) imp Fold4 
#> -(rfe) imp Fold4 
#> +(rfe) fit Fold4 size: 20 
#> -(rfe) fit Fold4 size: 20 
#> +(rfe) fit Fold4 size: 15 
#> -(rfe) fit Fold4 size: 15 
#> +(rfe) fit Fold4 size: 10 
#> -(rfe) fit Fold4 size: 10 
#> +(rfe) fit Fold4 size:  9 
#> -(rfe) fit Fold4 size:  9 
#> +(rfe) fit Fold4 size:  8 
#> -(rfe) fit Fold4 size:  8 
#> +(rfe) fit Fold4 size:  7 
#> -(rfe) fit Fold4 size:  7 
#> +(rfe) fit Fold4 size:  6 
#> -(rfe) fit Fold4 size:  6 
#> +(rfe) fit Fold4 size:  5 
#> -(rfe) fit Fold4 size:  5 
#> +(rfe) fit Fold4 size:  4 
#> -(rfe) fit Fold4 size:  4 
#> +(rfe) fit Fold4 size:  3 
#> -(rfe) fit Fold4 size:  3 
#> +(rfe) fit Fold4 size:  2 
#> -(rfe) fit Fold4 size:  2 
#> +(rfe) fit Fold4 size:  1 
#> -(rfe) fit Fold4 size:  1 
#> +(rfe) fit Fold5 size: 26 
#> -(rfe) fit Fold5 size: 26 
#> +(rfe) imp Fold5 
#> -(rfe) imp Fold5 
#> +(rfe) fit Fold5 size: 20 
#> -(rfe) fit Fold5 size: 20 
#> +(rfe) fit Fold5 size: 15 
#> -(rfe) fit Fold5 size: 15 
#> +(rfe) fit Fold5 size: 10 
#> -(rfe) fit Fold5 size: 10 
#> +(rfe) fit Fold5 size:  9 
#> -(rfe) fit Fold5 size:  9 
#> +(rfe) fit Fold5 size:  8 
#> -(rfe) fit Fold5 size:  8 
#> +(rfe) fit Fold5 size:  7 
#> -(rfe) fit Fold5 size:  7 
#> +(rfe) fit Fold5 size:  6 
#> -(rfe) fit Fold5 size:  6 
#> +(rfe) fit Fold5 size:  5 
#> -(rfe) fit Fold5 size:  5 
#> +(rfe) fit Fold5 size:  4 
#> -(rfe) fit Fold5 size:  4 
#> +(rfe) fit Fold5 size:  3 
#> -(rfe) fit Fold5 size:  3 
#> +(rfe) fit Fold5 size:  2 
#> -(rfe) fit Fold5 size:  2 
#> +(rfe) fit Fold5 size:  1 
#> -(rfe) fit Fold5 size:  1 
#> [1] "RFE completed. Selecting features..."
#> [1] "Features selected. Starting hyperparameter tuning..."
#> [1] "Hyperparameter tuning completed. Training model..."
#> note: only 3 unique complexity parameters in default grid. Truncating the grid to 3 .
#> 
#> [1] "Model trained. Starting model evaluation..."
#> [1] "Evaluation completed. Starting predictions on new data..."
#> [1] "Predictions completed. Starting indirect estimation on domain..."
#> [1] "Estimation completed. Returning results..."
```
