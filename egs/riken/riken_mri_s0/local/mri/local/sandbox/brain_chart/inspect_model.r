# inspect_model.r
# Quick script to inspect the model structure

library(gamlss)

setwd("/work02/home/bin-wu/workspace/projects/tests/test_dev/Lifespan")

source("100.common-variables.r")
source("101.common-functions.r")
source("300.variables.r")
source("301.functions.r")

# Load one model
FIT <- readRDS("Share/RefittedModels/FIT_GMV.rds")

cat("=== FIT Object Structure ===\n")
cat("Class:", class(FIT), "\n")
cat("Names:", paste(names(FIT), collapse=", "), "\n\n")

if ("param" %in% names(FIT)) {
  cat("=== FIT$param Structure ===\n")
  cat("Class:", class(FIT$param), "\n")
  cat("Names:", paste(names(FIT$param), collapse=", "), "\n\n")
}

if ("fit" %in% names(FIT)) {
  cat("=== FIT$fit Structure ===\n")
  cat("Class:", class(FIT$fit), "\n")
  print(summary(FIT$fit))
}

# Try the example from the documentation
cat("\n=== Testing example code ===\n")
POP.CURVE.LIST <- list(
  AgeTransformed = seq(log(90), log(365*95), length.out = 16),
  sex = c("Female", "Male")
)
POP.CURVE.RAW <- do.call(what = expand.grid, args = POP.CURVE.LIST)

cat("POP.CURVE.RAW structure:\n")
str(POP.CURVE.RAW)

cat("\nAttempting Apply.Param...\n")
tryCatch({
  CURVE <- Apply.Param(NEWData = POP.CURVE.RAW, FITParam = FIT$param)
  cat("Success! CURVE structure:\n")
  str(CURVE)
  cat("\nColumn names:", paste(names(CURVE), collapse=", "), "\n")
}, error = function(e) {
  cat("Error:", conditionMessage(e), "\n")
})