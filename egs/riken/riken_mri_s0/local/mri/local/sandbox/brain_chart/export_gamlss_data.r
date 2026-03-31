# export_gamlss_data.r
# This script exports all centile predictions from GAMLSS models to CSV files

library(gamlss)

setwd("/work02/home/bin-wu/workspace/projects/tests/test_dev/Lifespan")

# Source the required files from the repository
source("100.common-variables.r")
source("101.common-functions.r")
source("300.variables.r")
source("301.functions.r")

# Function to extract centiles and export to CSV
extract_and_export_centiles <- function(fit_path, output_prefix) {
  
  cat(paste0("Processing: ", fit_path, "\n"))
  
  # Load fitted model
  FIT <- readRDS(fit_path)
  
  # Create age sequence (transformed scale)
  # From 90 days (~3 months) to 95 years
  age_transformed <- seq(log(90), log(365*95), length.out = 200)
  
  # Create grid for females and males
  POP.CURVE.LIST <- list(
    AgeTransformed = age_transformed,
    sex = c("Female", "Male")
  )
  POP.CURVE.RAW <- do.call(what = expand.grid, args = POP.CURVE.LIST)
  
  # Apply parameters to get predictions
  CURVE <- Apply.Param(NEWData = POP.CURVE.RAW, FITParam = FIT$param)
  
  # Transform age back to years for easier interpretation
  CURVE$age_days <- exp(CURVE$AgeTransformed)
  CURVE$age_years <- CURVE$age_days / 365.25
  
  # Separate female and male data
  female_data <- CURVE[CURVE$sex == "Female", ]
  male_data <- CURVE[CURVE$sex == "Male", ]
  
  # Create export dataframes
  female_export <- data.frame(
    age_transformed = female_data$AgeTransformed,
    age_days = female_data$age_days,
    age_years = female_data$age_years,
    median = female_data$PRED.m500.pop,
    centile_2.5 = female_data$PRED.l025.pop,
    centile_97.5 = female_data$PRED.u975.pop,
    mean = female_data$PRED.mean.pop,
    variance = female_data$PRED.variance.pop,
    sex = "Female"
  )
  
  male_export <- data.frame(
    age_transformed = male_data$AgeTransformed,
    age_days = male_data$age_days,
    age_years = male_data$age_years,
    median = male_data$PRED.m500.pop,
    centile_2.5 = male_data$PRED.l025.pop,
    centile_97.5 = male_data$PRED.u975.pop,
    mean = male_data$PRED.mean.pop,
    variance = male_data$PRED.variance.pop,
    sex = "Male"
  )
  
  # Combine
  all_data <- rbind(female_export, male_export)
  
  # Save to CSV
  output_file <- paste0(output_prefix, "_centiles.csv")
  write.csv(all_data, output_file, row.names = FALSE)
  
  cat(paste0("  ✓ Saved: ", output_file, "\n"))
  cat(paste0("    Age range: ", round(min(all_data$age_years), 2), 
             " to ", round(max(all_data$age_years), 2), " years\n"))
  cat(paste0("    Rows: ", nrow(all_data), " (", nrow(female_export), 
             " per sex)\n"))
  cat(paste0("    Female median range: ", round(min(female_export$median), 3),
             " to ", round(max(female_export$median), 3), "\n"))
  cat(paste0("    Male median range: ", round(min(male_export$median), 3),
             " to ", round(max(male_export$median), 3), "\n\n"))
  
  return(all_data)
}

# Create output directory
output_dir <- "/work02/home/bin-wu/workspace/projects/tests/test_dev/csv_exports"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Export data for all four tissues
tissues <- list(
  GMV = "Share/RefittedModels/FIT_GMV.rds",
  WMV = "Share/RefittedModels/FIT_WMV.rds",
  sGMV = "Share/RefittedModels/FIT_sGMV.rds",
  Ventricles = "Share/RefittedModels/FIT_Ventricles.rds"
)

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Exporting GAMLSS Model Predictions to CSV\n")
cat(paste(rep("=", 60), collapse = ""), "\n\n")

exported_data <- list()
for (tissue_name in names(tissues)) {
  fit_path <- tissues[[tissue_name]]
  output_prefix <- file.path(output_dir, tissue_name)
  
  tryCatch({
    exported_data[[tissue_name]] <- extract_and_export_centiles(fit_path, output_prefix)
  }, error = function(e) {
    cat("  ✗ Error processing", tissue_name, ":", conditionMessage(e), "\n\n")
  })
}

cat(paste(rep("=", 60), collapse = ""), "\n")
cat("Export process completed!\n")
cat("Successfully exported:", length(exported_data), "out of", length(tissues), "tissues\n")
cat("Output directory:", output_dir, "\n")
cat(paste(rep("=", 60), collapse = ""), "\n")

# Create a summary file if we have any data
if (length(exported_data) > 0) {
  summary_df <- data.frame(
    Tissue = names(exported_data),
    MinAge_years = sapply(exported_data, function(x) round(min(x$age_years), 2)),
    MaxAge_years = sapply(exported_data, function(x) round(max(x$age_years), 2)),
    NumRows = sapply(exported_data, nrow),
    Female_Median_Min = sapply(exported_data, function(x) {
      round(min(x$median[x$sex == "Female"]), 3)
    }),
    Female_Median_Max = sapply(exported_data, function(x) {
      round(max(x$median[x$sex == "Female"]), 3)
    }),
    Male_Median_Min = sapply(exported_data, function(x) {
      round(min(x$median[x$sex == "Male"]), 3)
    }),
    Male_Median_Max = sapply(exported_data, function(x) {
      round(max(x$median[x$sex == "Male"]), 3)
    })
  )
  
  summary_file <- file.path(output_dir, "export_summary.csv")
  write.csv(summary_df, summary_file, row.names = FALSE)
  
  cat("\nSummary:\n")
  print(summary_df)
  cat("\n✓ Summary saved to:", summary_file, "\n")
}

cat("\nDone!\n")
