#' Projection Estimator
#'
#' @description The function addresses the problem of combining information from two or more independent surveys, a common challenge in survey sampling. It focuses on cases where: \cr
#' \itemize{
#'    \item Survey 1: A large sample collects only auxiliary information.
#'    \item Survey 2: A much smaller sample collects data on both the variables of interest and the auxiliary variables.
#' }
#' The function implements a model-assisted projection estimation method based on a working model, with the reference distribution derived from a design-based approach.
#'
#' @references
#' \enumerate{
#'  \item Kim, J. K., & Rao, J. N. (2012). Combining data from two independent surveys: a model-assisted approach. Biometrika, 99(1), 85-100.
#'}
#'
#' @param formula  An object of class formula that contains a description of the model to be fitted. The variables included in the formula must be contained in the \code{data_model} dan \code{data_proj}.
#' @param domain Column names in data_model and data_proj representing specific domains for which disaggregated data needs to be produced.
#' @param weight Column name in data_proj representing the survey weight.
#' @param data_model A data frame or a data frame extension (e.g., a tibble) representing the second survey, characterized by a much smaller sample, provides information on both the variable of interest and the auxiliary variables.
#' @param data_proj A data frame or a data frame extension (e.g., a tibble) representing the first survey, characterized by a large sample that collects only auxiliary information or general-purpose variables.
#' @param model The working model to be used in the projection estimator. The available working models are "linear" for linear regression and "logistic" for logistic regression.
#' @param cutoff Cutoff for binary classification (model='logistic')
#' @param ... Further argument to the \code{\link[stats]{lm}} and \code{\link[stats]{glm}} model.
#'
#' @return The function returns a list with the following objects (\code{model}, \code{prediction} and \code{df_result}):
#' \code{model} model yang digunakan dalam projection
#' \code{prediction} A vector containing the prediction results from the model.
#' \code{df_result} A data frame with the following columns:
#'    * \code{domain} The name of the domain.
#'    * \code{yhat} The estimated result for each domain.
#'    * \code{var_yhat} The sample variance of the projection estimator for each domain.
#'    * \code{rse} The Relative Standard Error (RSE) in percentage (%).
#'
#' @export
#' @importFrom dplyr %>%
#' @importFrom rlang .data
#'
#' @examples
#' \dontrun{
#' library(sae.projection)
#' library(dplyr)
#'
#' df_res <- projection_estimator(
#'    formula = unemployment ~ edu + sex + age + head,
#'    domain = c('prov', 'regency'),
#'    weight = 'weight',
#'    data_model = df_svy2,
#'    data_proj = df_svy1,
#'    model = 'logistic'
#' )
#'
#' df_svy2_new <- df_svy2 %>% dplyr::filter(!is.na(income))
#'
#' m2 <- projection_estimator(
#'   formula = income ~ edu + sex + age + head,
#'   domain = c('prov', 'regency'),
#'   weight = 'weight',
#'   data_model = df_svy2_new,
#'   data_proj = df_svy1,
#'   model = 'linear'
#' )
#'}
#' @md
projection_estimator <- function(formula, domain, weight, data_model, data_proj, model, cutoff = "optimum", ...) {
  domain <- .check_variable(domain, data_model, data_proj)
  weight <- .check_variable(weight, data_model, data_proj)
  y_name <- as.character(rlang::f_lhs(formula))
  model <- match.arg(tolower(model), choices = c('linear', 'logistic'))
  projection_result <- list(
    model = NA,
    prediction = NA,
    df_result = NA
  )

  if (model == 'logistic') {
    # model from small data
    working_model <- stats::glm(formula = formula, data = data_model, family = stats::binomial, ...)

    # prediction
    if (cutoff == "optimum") {
      cli::cli_alert_info('Getting the optimum cut-off...')
      cutoff <- MKclass::optCutoff(
        pred = working_model$fitted.values,
        truth = data_model[[y_name]],
        namePos = 1,
        perfMeasure = "F1S",
        parallel = TRUE
      )
      cli::cli_alert_success('The optimum cut-off is {cutoff}')
    }
    predict_result <- stats::predict(working_model, newdata = data_proj, type = "response")
    predict_result <- ifelse(predict_result > cutoff, 1, 0)

    # est
    data_proj$y <- predict_result

    weight <- as.symbol(weight)
    df_result <- data_proj %>%
      dplyr::group_by_at(domain) %>%
      dplyr::summarise(
        n = dplyr::n(),
        yhat = sum(.data$y * !!weight) / sum(!!weight),
        var_yhat = ((sum(!!weight) - .data$n) / sum(!!weight)) * mean(.data$y) * (1 - mean(.data$y)) / (.data$n - 1),
        rse = sqrt(.data$var_yhat) / .data$yhat * 100,
        .groups = "drop"
      )
    projection_result$cutoff <- cutoff

  }else if (model == 'linear') {
    working_model <- stats::lm(formula = formula, data = data_model,...)
    predict_result <- stats::predict(working_model, newdata = data_proj)

    data_proj$y <- predict_result
    # s2.Y <- var(Data2$Y.hat)

    weight <- as.symbol(weight)
    df_result <- data_proj %>%
      dplyr::group_by_at(domain) %>%
      dplyr::summarise(
        n = dplyr::n(),
        yhat = sum(.data$y * !!weight) / sum(!!weight),
        var_yhat = ((sum(!!weight) - .data$n) / sum(!!weight)) * stats::var(.data$y) / .data$n,
        rse = sqrt(.data$var_yhat) / .data$yhat * 100,
        .groups = "drop"
      )
  }

  projection_result$model <- working_model
  projection_result$prediction <- predict_result
  projection_result$df_result <- df_result

  print(df_result)
  return(invisible(projection_result))
}

.check_variable <- function(variable, data_model, data_proj) {
  if (methods::is(variable, "character")) {
    if (length(variable) > 1) {
      if (mean(variable %in% colnames(data_model)) == 1) {
        if (mean(variable %in% colnames(data_proj)) == 1) {
          return(variable)
        } else {
          cli::cli_abort('variable "{setdiff(variable, colnames(data_proj))}" is not found in the data_proj')
        }
      } else {
        cli::cli_abort('variable "{setdiff(variable, colnames(data_model))}" is not found in the data_model')
      }
    } else if (variable %in% colnames(data_model)) {
      if (variable %in% colnames(data_proj)) {
        return(variable)
      } else {
        cli::cli_abort('variable "{variable}" is not found in the data_proj')
      }
    } else {
      cli::cli_abort('variable "{variable}" is not found in the data_model')
    }
  } else {
    cli::cli_abort('variable "{variable}" must be character')
  }
}
