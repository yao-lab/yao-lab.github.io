\name{dream}
\alias{dream}
\title{Dream of the Red Chamber, one of the Four Great Classical Novels of Chinese Literature by CAO, Xueqin in the middle of 18th century.}
\description{
 The \code{dream} data frame has 475 rows and 375 columns. It records the appearance of 374 characters in 475 scenes, which are divided into two parts according to the authorship. Column 1 is the indicator of early 80 chapters believed to be composed by CAO, Xueqin, while later 40 chapters attributed to GAO, E. The remaining columns [2:375] list the 374 characters who appeared ("1") or not ("0") in each of the scenes. The dataset was collected via crowdsourcing in the classes of Mathematical Introduction to Data Analysis and Statistical Learning, taught by Prof. Yuan YAO at Peking University. 
}
\format{
  This file contains the following data frame:
  \describe{
    \item{dream}{a data frame of 475 rows and 375 columns}
  }
}
\details{
The first column is 1 if the scene happened in the first 80 chapters by CAO, Xueqin, and 0 if in the remaining 40 chapters by Gao, E. All the other columns are of binary values [0:1]. The character names are in Chinese.
}
\source{
  \url{http://math.stanford.edu/~yuany/course/2014.fall/}
  }
\references{
Yuan YAO, A Mathematical Introduction to Data Analysis, Lecture Notes in \emph{Peking University}
}
\keyword{datasets}
