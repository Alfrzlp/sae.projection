feature_selection <- function(y, x, seed = 1, method) {

  if (method == "lasso") {
    idx <- !is.na(y)

    y <- y[idx]
    x <- x[idx, ]

    # find optimal lambda value that minimizes test MSE
    set.seed(seed)
    cv_model <- glmnet::cv.glmnet(as.matrix(x), y, alpha = 1)
    best_lambda <- cv_model$lambda.min

    # produce plot of test MSE by lambda value
    cli::cli_alert_info("Best Lambda : {best_lambda}")
    graphics::plot(cv_model)

    best_model <- glmnet::glmnet(as.matrix(x), y, alpha = alpha, lambda = best_lambda)

    all_var <- as.matrix(stats::coef(best_model))
    all_var <- names(all_var[all_var[, 1] != 0, ][-1])
  } else if (method == "boruta") {

  }

  return(all_var)
}
