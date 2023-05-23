mf <- function(Y, X, K_max = 1L, stop_tol = 0.005, iter_max = 5000, tol = 1e-4, verbose = TRUE) {
  set.seed(123)
  N <- dim(Y)[1]
  M <- dim(Y)[2]

  mf_results <- list()
  Y_hat <- matrix(0, nrow = N, ncol = M)
  Y_k <- Y
  for (k in 1:K_max) {
    mf_result_k <- mf_single_factor(Y_k, X,
      iter_max = iter_max,
      tol = tol,
      verbose = verbose
    )
    Y_hat_k <- as.matrix(mf_result_k$mu) %*% t(mf_result_k$nu)

    if ((var(as.vector(Y_hat_k)) * mf_result_k$tau) > stop_tol) {
      message("Factor ", k, " retained!")
    } else {
      message("Factor ", k, " zeroed out!")
      break
    }

    mf_results <- c(mf_results, list(mf_result_k))
    Y_k <- Y_k - Y_hat_k
  }

  return(mf_results)
}

mf_single_factor <- function(Y, X, iter_max = 5000, tol = 1e-5, verbose = TRUE) {
  N <- dim(Y)[1]
  M <- dim(Y)[2]

  # initialize
  init <- initial(Y, N, M)
  mu <- init$mu
  a2 <- init$a2
  nu <- init$nu
  b2 <- init$b2
  tau <- init$tau
  beta <- init$beta
  FX <- init$FX
  rm(init)

  ProY <- Y
  ProY[is.na(Y)] <- 0

  n_count <- sum(!is.na(Y))

  tau_ma <- matrix(tau, N, M)
  tau_ma[is.na(Y)] <- 0

  proj <- Y
  proj[is.na(Y)] <- 0
  proj[!is.na(Y)] <- 1

  ELBO_old <- 0
  ELBOs <- NULL

  lm_data <- data.frame(mu = mu, X)
  for (iter in 1:iter_max) {
    # E-step
    Ez2 <- mu^2 + a2
    b2 <- 1 / (1 + as.vector(t(Ez2) %*% tau_ma))
    nu <- b2 * as.vector(tau * (t(ProY) %*% as.matrix(mu)))


    Ew2 <- nu^2 + b2
    a2 <- 1 / (beta + as.vector(tau_ma %*% as.matrix(Ew2)))
    mu <- a2 * (FX * beta + as.vector(tau * (ProY %*% as.matrix(nu))))

    # M-step
    mu2 <- mu^2
    nu2 <- nu^2

    tau <- n_count / sum(((Y - as.matrix(mu) %*% t(nu))^2 + as.matrix(mu2 + a2) %*% t(nu2 + b2) - as.matrix(mu2) %*% t(nu2)), na.rm = T)
    tau_ma <- matrix(tau, N, M)
    tau_ma[is.na(Y)] <- 0

    beta <- N / (sum(mu^2) + sum(a2) - 2 * sum(mu * FX) + sum(FX^2))

    # lm fit
    lm_data$mu <- mu
    lm_fit <- lm(mu ~ ., data = lm_data)
    FX <- predict(lm_fit, lm_data)


    ELBO_current <- getELBO(Y, mu, a2, nu, b2, tau, tau_ma, beta, FX)

    gap <- abs(ELBO_current - ELBO_old) / abs(ELBO_old)
    if (gap < tol) break

    if (iter > 1 & ELBO_current < ELBO_old) message("ELBO decreasing")

    ELBO_old <- ELBO_current

    ELBOs <- c(ELBOs, ELBO_current)

    if (verbose) {
      cat(
        "Iteration: ", iter, " ELBO: ", ELBO_current, " tau: ", tau,
        " beta: ", beta, "diff", gap, "\n"
      )
    }
  }
  cat("After ", iter, " iteration ends!\n")

  return(list(mu = mu, a2 = a2, nu = nu, b2 = b2, tau = tau, beta = beta, FX = FX, lm_fit = lm_fit, ELBOs = ELBOs))
}

initial <- function(Y, N, M) {
  mu <- matrix(rnorm(N), N, 1)
  # a2 <- abs(rnorm(1)) + 1
  a2 <- rep(1, N)
  nu <- matrix(0, M, 1)
  # b2 <- 1
  b2 <- rep(1, M)

  tau <- 2 / var(as.vector(Y), na.rm = TRUE)
  beta <- 2 / var(as.vector(mu))

  FX <- rep(0.0, N)
  return(list(mu = mu, a2 = a2, nu = nu, b2 = b2, tau = tau, beta = beta, FX = FX))
}

getELBO <- function(Y, mu, a2, nu, b2, tau, tau_ma, beta, FX) {
  n <- dim(Y)[1]
  m <- dim(Y)[2]

  n_count <- sum(!is.na(Y))

  mu2 <- mu^2
  nu2 <- nu^2

  elbo1 <- -n_count * log(2 * pi / tau) / 2
  elbo2 <- -tau * sum((Y - as.matrix(mu) %*% t(nu))^2 + as.matrix(mu2 + a2) %*% t(nu2 + b2) - as.matrix(mu2) %*% t(nu2), na.rm = T) / 2
  elbo3 <- -n * log(2 * pi / beta) / 2 - beta * (sum(mu2) + sum(a2) - 2 * sum(mu * FX) + sum(FX^2)) / 2
  elbo4 <- -m * log(2 * pi) / 2 - (sum(nu2) + sum(b2)) / 2
  elbo5 <- sum(log(2 * pi * a2)) / 2 + n / 2
  elbo6 <- sum(log(2 * pi * b2)) / 2 + m / 2

  elbo <- c(elbo1 + elbo2 + elbo3 + elbo4 + elbo5 + elbo6)
  return(elbo)
}
