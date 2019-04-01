\name{diabetes}
\alias{diabetes}
\title{Blood and other measurements in diabetics}
\description{
 The \code{diabetes} data frame has 442 rows and 3 columns.
These are the data used in the Efron et al "Least Angle Regression" paper.
}
\format{
  This data frame contains the following columns:
  \describe{
    \item{x}{a matrix with 10 columns}
    \item{y}{a numeric vector}
    \item{x2}{a matrix with 64 columns}
  }
}
\details{
The x matrix has been standardized to have unit L2 norm in each column
and zero mean. The matrix x2 consists of x plus certain interactions.
}
\source{
  \url{http://www-stat.stanford.edu/~hastie/Papers/LARS/LeastAngle_2002.ps}
  }
\references{
Efron, Hastie, Johnstone and Tibshirani (2003) "Least Angle Regression"
(with discussion) \emph{Annals of Statistics}
}
\keyword{datasets}
