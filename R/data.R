#' @title df_svy22
#' @description A dataset from a survey conducted at the province level in Indonesia in 2022.
#' @format A data frame with 74.070 rows and 11 variables.
#'
#' \describe{
#'   \item{PSU}{Primary Sampling Unit}
#'   \item{WEIGHT}{Weight from survey}
#'   \item{PROV}{province code}
#'   \item{REGENCY}{regency/municipality code}
#'   \item{STRATA}{Strata}
#'   \item{income}{Income}
#'   \item{neet}{Not in education employment or training status}
#'   \item{sex}{sex (1: male, 2: female)}
#'   \item{age}{age}
#'   \item{disability}{disability status (0: False, 1: True)}
#'   \item{edu}{last completed education}
#' }
#' @source \url{https://www.bps.go.id}
"df_svy22"

#' @title df_svy23
#' @description A dataset from a survey conducted at the province level in Indonesia in 2023.
#' @format A data frame with 66.245 rows and 11 variables.
#'
#' \describe{
#'   \item{PSU}{Primary Sampling Unit}
#'   \item{WEIGHT}{Weight from survey}
#'   \item{PROV}{province code}
#'   \item{REGENCY}{regency/municipality code}
#'   \item{STRATA}{Strata}
#'   \item{income}{Income}
#'   \item{neet}{Not in education employment or training status}
#'   \item{sex}{sex (1: male, 2: female)}
#'   \item{age}{age}
#'   \item{disability}{disability status (0: False, 1: True)}
#'   \item{edu}{last completed education}
#' }
#' @source \url{https://www.bps.go.id}
"df_svy23"

#' @title df_svy_A
#' @description A dataset from a survey that was conducted in a certain province within Indonesia, presented only at provincial level (Domain 1).
#' @format A data frame with 3655 rows and 33 variables with 6 domains.
#'
#' \describe{
#'   \item{ID}{Unique identifier for each respondent}
#'   \item{no_sample}{Sample number}
#'   \item{no_household}{Household number}
#'   \item{no_member}{Household member number}
#'   \item{weight}{Weight from survey}
#'   \item{province}{Province code}
#'   \item{X1}{Predictor variables X1}
#'   \item{X2}{Predictor variables X2}
#'   \item{X3}{Predictor variables X3}
#'   \item{X4}{Predictor variables X4}
#'   \item{X5}{Predictor variables X5}
#'   \item{X6}{Predictor variables X6}
#'   \item{X7}{Predictor variables X7}
#'   \item{X8}{Predictor variables X8}
#'   \item{X9}{Predictor variables X9}
#'   \item{X10}{Predictor variables X10}
#'   \item{X11}{Predictor variables X11}
#'   \item{X12}{Predictor variables X12}
#'   \item{X13}{Predictor variables X13}
#'   \item{X14}{Predictor variables X14}
#'   \item{X15}{Predictor variables X15}
#'   \item{X16}{Predictor variables X16}
#'   \item{X17}{Predictor variables X17}
#'   \item{X18}{Predictor variables X18}
#'   \item{X19}{Predictor variables X19}
#'   \item{X20}{Predictor variables X20}
#'   \item{X21}{Predictor variables X21}
#'   \item{X22}{Predictor variables X22}
#'   \item{X23}{Predictor variables X23}
#'   \item{X24}{Predictor variables X24}
#'   \item{X25}{Predictor variables X25}
#'   \item{X26}{Predictor variables X26}
#'   \item{Y}{Target variable (1: Yes, 0: No)}
#' }
"df_svy_A"

#' @title df_svy_B
#' @description A dataset from a survey that was conducted in a certain province within Indonesia, presented at the regency level (Domain 2).
#' @format A data frame with 18842 rows and 37 variables with 6 domains.
#'
#' \describe{
#'   \item{year}{Year the survey was conducted}
#'   \item{psu}{Primary Sampling Unit (PSU)}
#'   \item{ssu}{Secondary Sampling Unit (SSU)}
#'   \item{strata}{Strata used for sampling}
#'   \item{ID}{Unique identifier for each respondent}
#'   \item{no_sample}{Sample number}
#'   \item{no_household}{Household number}
#'   \item{no_member}{Household member number}
#'   \item{weight}{Weight from survey}
#'   \item{province}{Province code}
#'   \item{regency}{Regency or municipality code}
#'   \item{X1}{Predictor variables X1}
#'   \item{X2}{Predictor variables X2}
#'   \item{X3}{Predictor variables X3}
#'   \item{X4}{Predictor variables X4}
#'   \item{X5}{Predictor variables X5}
#'   \item{X6}{Predictor variables X6}
#'   \item{X7}{Predictor variables X7}
#'   \item{X8}{Predictor variables X8}
#'   \item{X9}{Predictor variables X9}
#'   \item{X10}{Predictor variables X10}
#'   \item{X11}{Predictor variables X11}
#'   \item{X12}{Predictor variables X12}
#'   \item{X13}{Predictor variables X13}
#'   \item{X14}{Predictor variables X14}
#'   \item{X15}{Predictor variables X15}
#'   \item{X16}{Predictor variables X16}
#'   \item{X17}{Predictor variables X17}
#'   \item{X18}{Predictor variables X18}
#'   \item{X19}{Predictor variables X19}
#'   \item{X20}{Predictor variables X20}
#'   \item{X21}{Predictor variables X21}
#'   \item{X22}{Predictor variables X22}
#'   \item{X23}{Predictor variables X23}
#'   \item{X24}{Predictor variables X24}
#'   \item{X25}{Predictor variables X25}
#'   \item{X26}{Predictor variables X26}
#' }
"df_svy_B"

#' @title df_survey_A
#' @description The dataset comes from a large-scale survey conducted at a broader geographic level, corresponding to Domain 1.
#' @format A data frame with 5313 rows and 29 variables.
#'
#' \describe{
#'   \item{weight}{Survey weight}
#'   \item{psu}{Primary Sampling Unit}
#'   \item{ssu}{Secondary Sampling Unit}
#'   \item{strata}{Strata used for sampling}
#'   \item{ID}{ID number}
#'   \item{no_sample}{Sample number}
#'   \item{no_household}{Household number}
#'   \item{province}{Province code}
#'   \item{X1}{Predictor variables X1}
#'   \item{X2}{Predictor variables X2}
#'   \item{X3}{Predictor variables X3}
#'   \item{X4}{Predictor variables X4}
#'   \item{X5}{Predictor variables X5}
#'   \item{X6}{Predictor variables X6}
#'   \item{X7}{Predictor variables X7}
#'   \item{X8}{Predictor variables X8}
#'   \item{X9}{Predictor variables X9}
#'   \item{X10}{Predictor variables X10}
#'   \item{X11}{Predictor variables X11}
#'   \item{X12}{Predictor variables X12}
#'   \item{X13}{Predictor variables X13}
#'   \item{X14}{Predictor variables X14}
#'   \item{X15}{Predictor variables X15}
#'   \item{X16}{Predictor variables X16}
#'   \item{X17}{Predictor variables X17}
#'   \item{X18}{Predictor variables X18}
#'   \item{X19}{Predictor variables X19}
#'   \item{X20}{Predictor variables X20}
#'   \item{Y}{Target variable(1: Yes, 0: No)}
#' }
"df_survey_A"

#' @title df_survey_B
#' @description The dataset comes from a survey conducted at a more specific geographic level, corresponding to Domain 2.
#' @format A data frame with 105242 rows and 29 variables.
#'
#' \describe{
#'   \item{weight}{Survey weight}
#'   \item{psu}{Primary Sampling Unit}
#'   \item{ssu}{Secondary Sampling Unit}
#'   \item{strata}{Strata used for sampling}
#'   \item{ID}{ID number}
#'   \item{no_sample}{Sample number}
#'   \item{no_household}{Household number}
#'   \item{province}{Province code}
#'   \item{regency}{Regency or municipality code}
#'   \item{X1}{Predictor variables X1}
#'   \item{X2}{Predictor variables X2}
#'   \item{X3}{Predictor variables X3}
#'   \item{X4}{Predictor variables X4}
#'   \item{X5}{Predictor variables X5}
#'   \item{X6}{Predictor variables X6}
#'   \item{X7}{Predictor variables X7}
#'   \item{X8}{Predictor variables X8}
#'   \item{X9}{Predictor variables X9}
#'   \item{X10}{Predictor variables X10}
#'   \item{X11}{Predictor variables X11}
#'   \item{X12}{Predictor variables X12}
#'   \item{X13}{Predictor variables X13}
#'   \item{X14}{Predictor variables X14}
#'   \item{X15}{Predictor variables X15}
#'   \item{X16}{Predictor variables X16}
#'   \item{X17}{Predictor variables X17}
#'   \item{X18}{Predictor variables X18}
#'   \item{X19}{Predictor variables X19}
#'   \item{X20}{Predictor variables X20}
#' }
"df_survey_B"



