# export_monthly_fitted_values.r
# This script exports fitted values at monthly intervals from GAMLSS models

library(gamlss)

setwd("/work02/home/bin-wu/workspace/projects/tests/test_dev/Lifespan")

# Source the required files from the repository
source("100.common-variables.r")
source("101.common-functions.r")
source("300.variables.r")
source("301.functions.r")

# Function to extract fitted values at specified intervals
extract_monthly_values <- function(fit_path, output_prefix, interval_type = "monthly") {
  
  cat(paste0("Processing: ", fit_path, "\n"))
  
  # Load fitted model
  FIT <- readRDS(fit_path)
  
  # Create age sequence based on interval type
  if (interval_type == "monthly") {
    # Monthly from 3 months to 95 years
    # 3 months, 4 months, 5 months, ... up to 95 years
    ages_in_months <- seq(3, 95*12, by = 1)  # Every month
    age_days <- ages_in_months * 30.44  # Average days per month
    cat(paste0("  Creating ", length(ages_in_months), " monthly age points\n"))
    
  } else if (interval_type == "weekly_first_year") {
    # Weekly for first year, then monthly
    weeks_first_year <- seq(12, 52, by = 1)  # weeks 12-52 (3-12 months)
    months_after <- seq(13, 95*12, by = 1)   # months 13 onwards
    
    age_days <- c(weeks_first_year * 7, months_after * 30.44)
    ages_in_months <- age_days / 30.44
    cat(paste0("  Creating ", length(age_days), " age points (weekly first year, then monthly)\n"))
    
  } else if (interval_type == "yearly") {
    # Yearly from 0.25 to 95 years
    ages_in_years <- seq(0.25, 95, by = 1)
    age_days <- ages_in_years * 365.25
    ages_in_months <- ages_in_years * 12
    cat(paste0("  Creating ", length(ages_in_years), " yearly age points\n"))
  }
  
  # Transform to log scale (as used in the model)
  age_transformed <- log(age_days)
  
  # Create grid for females and males
  POP.CURVE.LIST <- list(
    AgeTransformed = age_transformed,
    sex = c("Female", "Male")
  )
  POP.CURVE.RAW <- do.call(what = expand.grid, args = POP.CURVE.LIST)
  
  # Apply parameters to get predictions
  CURVE <- Apply.Param(NEWData = POP.CURVE.RAW, FITParam = FIT$param)
  
  # Add readable age columns
  CURVE$age_days <- exp(CURVE$AgeTransformed)
  CURVE$age_months <- CURVE$age_days / 30.44
  CURVE$age_years <- CURVE$age_days / 365.25
  
  # Separate female and male data
  female_data <- CURVE[CURVE$sex == "Female", ]
  male_data <- CURVE[CURVE$sex == "Male", ]
  
  # Create export dataframes with all relevant columns
  female_export <- data.frame(
    age_months = female_data$age_months,
    age_years = female_data$age_years,
    age_days = female_data$age_days,
    age_transformed = female_data$AgeTransformed,
    median = female_data$PRED.m500.pop,
    mean = female_data$PRED.mean.pop,
    centile_2.5 = female_data$PRED.l025.pop,
    centile_25 = female_data$PRED.l250.pop,
    centile_75 = female_data$PRED.u750.pop,
    centile_97.5 = female_data$PRED.u975.pop,
    variance = female_data$PRED.variance.pop,
    mu = female_data$mu.pop,
    sigma = female_data$sigma.pop,
    nu = female_data$nu.pop,
    sex = "Female"
  )
  
  male_export <- data.frame(
    age_months = male_data$age_months,
    age_years = male_data$age_years,
    age_days = male_data$age_days,
    age_transformed = male_data$AgeTransformed,
    median = male_data$PRED.m500.pop,
    mean = male_data$PRED.mean.pop,
    centile_2.5 = male_data$PRED.l025.pop,
    centile_25 = male_data$PRED.l250.pop,
    centile_75 = male_data$PRED.u750.pop,
    centile_97.5 = male_data$PRED.u975.pop,
    variance = male_data$PRED.variance.pop,
    mu = male_data$mu.pop,
    sigma = male_data$sigma.pop,
    nu = male_data$nu.pop,
    sex = "Male"
  )
  
  # Combine
  all_data <- rbind(female_export, male_export)
  
  # Save to CSV
  output_file <- paste0(output_prefix, "_", interval_type, ".csv")
  write.csv(all_data, output_file, row.names = FALSE)
  
  cat(paste0("  ✓ Saved: ", output_file, "\n"))
  cat(paste0("    Total rows: ", nrow(all_data), " (", nrow(female_export), " per sex)\n"))
  cat(paste0("    Age range: ", round(min(all_data$age_years), 2), 
             " to ", round(max(all_data$age_years), 2), " years\n"))
  cat(paste0("    Female median range: ", round(min(female_export$median), 3),
             " to ", round(max(female_export$median), 3), "\n"))
  cat(paste0("    Male median range: ", round(min(male_export$median), 3),
             " to ", round(max(male_export$median), 3), "\n\n"))
  
  return(all_data)
}

# Create output directory
output_dir <- "/work02/home/bin-wu/workspace/projects/tests/test_dev/csv_exports_monthly"
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Export data for all four tissues
tissues <- list(
  GMV = "Share/RefittedModels/FIT_GMV.rds",
  WMV = "Share/RefittedModels/FIT_WMV.rds",
  sGMV = "Share/RefittedModels/FIT_sGMV.rds",
  Ventricles = "Share/RefittedModels/FIT_Ventricles.rds"
)

cat(paste(rep("=", 70), collapse = ""), "\n")
cat("Exporting Monthly GAMLSS Model Predictions to CSV\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# Export at different intervals
interval_types <- c("monthly", "yearly")

for (interval_type in interval_types) {
  cat("\n", paste(rep("-", 70), collapse = ""), "\n")
  cat("Interval type:", interval_type, "\n")
  cat(paste(rep("-", 70), collapse = ""), "\n\n")
  
  exported_data <- list()
  for (tissue_name in names(tissues)) {
    fit_path <- tissues[[tissue_name]]
    output_prefix <- file.path(output_dir, tissue_name)
    
    tryCatch({
      exported_data[[tissue_name]] <- extract_monthly_values(
        fit_path, output_prefix, interval_type
      )
    }, error = function(e) {
      cat("  ✗ Error processing", tissue_name, ":", conditionMessage(e), "\n\n")
    })
  }
  
  # Create a summary file for this interval type
  if (length(exported_data) > 0) {
    summary_df <- data.frame(
      Tissue = names(exported_data),
      Interval = interval_type,
      NumRows_Total = sapply(exported_data, nrow),
      NumRows_PerSex = sapply(exported_data, function(x) nrow(x[x$sex == "Female", ])),
      MinAge_years = sapply(exported_data, function(x) round(min(x$age_years), 2)),
      MaxAge_years = sapply(exported_data, function(x) round(max(x$age_years), 2)),
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
    
    summary_file <- file.path(output_dir, paste0("summary_", interval_type, ".csv"))
    write.csv(summary_df, summary_file, row.names = FALSE)
    
    cat("\nSummary for", interval_type, ":\n")
    print(summary_df)
  }
}

cat("\n", paste(rep("=", 70), collapse = ""), "\n")
cat("Export process completed!\n")
cat("Output directory:", output_dir, "\n")
cat(paste(rep("=", 70), collapse = ""), "\n\n")

# List all created files
cat("Created files:\n")
files <- list.files(output_dir, pattern = "\\.csv$", full.names = FALSE)
for (f in files) {
  cat("  -", f, "\n")
}

cat("\nDone!\n")
