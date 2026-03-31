#!/usr/bin/env Rscript
# LMM analysis - 6 individual system models WITHOUT mask_vol

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
})

# ===========================
# Configuration
# ===========================
INPUT_DIR <- "exp/mri/r_tables"
OUTPUT_DIR <- "exp/mri/r_tables"

cat(strrep("=", 80), "\n")
cat("LINEAR MIXED MODEL ANALYSIS - 6 SYSTEMS (WITHOUT MASK_VOL)\n")
cat(strrep("=", 80), "\n\n")

# ===========================
# Load Data
# ===========================
cat("Loading data...\n")

volume_6way <- read_csv(
  file.path(INPUT_DIR, "volume_by_system_6way.csv"),
  show_col_types = FALSE
) %>%
  mutate(
    Sex = factor(gender),
    Dataset = factor(dataset),
    ID = factor(ID_base),
    System = factor(system_6way)
  )

cat(sprintf("Loaded: %d rows, %d unique subjects\n",
            nrow(volume_6way), n_distinct(volume_6way$ID)))
cat(sprintf("Systems: %d\n\n", n_distinct(volume_6way$System)))

# ===========================
# Fit Individual System Models
# ===========================
cat(strrep("=", 80), "\n")
cat("FITTING INDIVIDUAL MODELS FOR EACH SYSTEM\n")
cat(strrep("=", 80), "\n")
cat("\nFormula: volume ~ log_age + Sex + Dataset + (1 + log_age | ID)\n")
cat("Fallback: volume ~ log_age + Sex + Dataset + (1 | ID)\n\n")

systems <- sort(unique(volume_6way$System))
all_system_results <- list()
model_summaries <- list()

for (sys in systems) {
  cat(strrep("-", 80), "\n")
  cat(sprintf("SYSTEM: %s\n", sys))
  cat(strrep("-", 80), "\n")

  sys_data <- volume_6way %>% filter(System == sys)

  cat(sprintf("  Observations: %d\n", nrow(sys_data)))
  cat(sprintf("  Subjects: %d\n", n_distinct(sys_data$ID)))
  cat(sprintf("  Datasets: Calgary=%d, New England=%d\n\n",
              sum(sys_data$Dataset == "calgary"),
              sum(sys_data$Dataset == "new_england")))

  # Fit model with random slopes (WITHOUT mask_vol)
  cat("  Fitting model with random slopes...\n")
  m_sys <- tryCatch({
    lmer(
      volume ~ log_age + Sex + Dataset + (1 + log_age | ID),
      data = sys_data,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )
  }, error = function(e) {
    cat("  Error:", e$message, "\n")
    return(NULL)
  })

  # Fallback to random intercepts only
  if (is.null(m_sys) || isSingular(m_sys)) {
    if (!is.null(m_sys)) {
      cat("  Singular fit detected.\n")
    }
    cat("  -> Using random intercepts only\n\n")
    m_sys <- lmer(
      volume ~ log_age + Sex + Dataset + (1 | ID),
      data = sys_data
    )
  } else {
    cat("  Converged with random slopes\n\n")
  }

  # Store model summary
  model_summaries[[as.character(sys)]] <- m_sys

  # Extract coefficients
  coef_sys <- as.data.frame(summary(m_sys)$coefficients)
  coef_sys$parameter <- rownames(coef_sys)
  coef_sys$model <- as.character(sys)

  all_system_results[[as.character(sys)]] <- coef_sys

  # Print key results
  cat("  Key Results:\n")
  age_row <- coef_sys[coef_sys$parameter == "log_age", ]
  cat(sprintf("    Age effect:     beta=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
              age_row$Estimate, age_row$`Std. Error`,
              age_row$`t value`, age_row$`Pr(>|t|)`))

  if ("SexM" %in% coef_sys$parameter) {
    sex_row <- coef_sys[coef_sys$parameter == "SexM", ]
    cat(sprintf("    Sex (M vs F):   beta=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
                sex_row$Estimate, sex_row$`Std. Error`,
                sex_row$`t value`, sex_row$`Pr(>|t|)`))
  }

  if ("Datasetnew_england" %in% coef_sys$parameter) {
    dataset_row <- coef_sys[coef_sys$parameter == "Datasetnew_england", ]
    cat(sprintf("    Dataset (NE):   beta=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
                dataset_row$Estimate, dataset_row$`Std. Error`,
                dataset_row$`t value`, dataset_row$`Pr(>|t|)`))
  }

  cat("\n")
}

# ===========================
# Combine and Save Results
# ===========================
cat(strrep("=", 80), "\n")
cat("SAVING RESULTS\n")
cat(strrep("=", 80), "\n\n")

# Combine all coefficients
all_system_coefs <- bind_rows(all_system_results)
output_file <- file.path(OUTPUT_DIR, "lmm_6systems_no_maskvol_results.csv")
write_csv(all_system_coefs, output_file)
cat(sprintf("All coefficients saved: %s\n", output_file))

# ===========================
# Summary Table: Age Effects
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("SUMMARY: AGE EFFECTS ACROSS ALL 6 SYSTEMS\n")
cat(strrep("=", 80), "\n\n")

age_effects <- all_system_coefs %>%
  filter(parameter == "log_age") %>%
  select(model, Estimate, `Std. Error`, `t value`, `Pr(>|t|)`) %>%
  arrange(`Pr(>|t|)`) %>%
  mutate(
    Significance = case_when(
      `Pr(>|t|)` < 0.001 ~ "***",
      `Pr(>|t|)` < 0.01 ~ "**",
      `Pr(>|t|)` < 0.05 ~ "*",
      TRUE ~ ""
    )
  )

# Convert to data frame to avoid print issues
age_effects_df <- as.data.frame(age_effects)
print(age_effects_df)

output_file2 <- file.path(OUTPUT_DIR, "summary_age_effects_6systems_no_maskvol.csv")
write_csv(age_effects, output_file2)
cat(sprintf("\nAge effects summary saved: %s\n", output_file2))

# ===========================
# Summary Table: All Fixed Effects
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("SUMMARY: ALL FIXED EFFECTS\n")
cat(strrep("=", 80), "\n\n")

fixed_effects_summary <- all_system_coefs %>%
  select(model, parameter, Estimate, `Std. Error`, `t value`, `Pr(>|t|)`) %>%
  mutate(
    Significance = case_when(
      `Pr(>|t|)` < 0.001 ~ "***",
      `Pr(>|t|)` < 0.01 ~ "**",
      `Pr(>|t|)` < 0.05 ~ "*",
      TRUE ~ ""
    )
  ) %>%
  arrange(model, parameter)

output_file3 <- file.path(OUTPUT_DIR, "summary_all_effects_6systems_no_maskvol.csv")
write_csv(fixed_effects_summary, output_file3)
cat(sprintf("All effects summary saved: %s\n", output_file3))

# ===========================
# Print Summary Statistics
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("MODEL STATISTICS\n")
cat(strrep("=", 80), "\n\n")

for (sys in names(model_summaries)) {
  m <- model_summaries[[sys]]
  cat(sprintf("%-40s\n", sys))
  cat(sprintf("  AIC: %.1f, BIC: %.1f\n", AIC(m), BIC(m)))

  # Random effects variance
  vc <- as.data.frame(VarCorr(m))
  cat("  Random effects:\n")
  for (i in 1:nrow(vc)) {
    if (vc$grp[i] == "ID") {
      var_name <- ifelse(is.na(vc$var1[i]), "Intercept", as.character(vc$var1[i]))
      cat(sprintf("    %s variance: %.2f (SD: %.2f)\n",
                  var_name, vc$vcov[i], vc$sdcor[i]))
    } else if (vc$grp[i] == "Residual") {
      cat(sprintf("    Residual variance: %.2f (SD: %.2f)\n",
                  vc$vcov[i], vc$sdcor[i]))
    }
  }
  cat("\n")
}

# ===========================
# Interpretation Guide
# ===========================
cat(strrep("=", 80), "\n")
cat("INTERPRETATION GUIDE\n")
cat(strrep("=", 80), "\n\n")

cat("Age Effect Interpretation:\n")
cat("  The coefficient represents change in volume (mm3) per unit increase in log(age)\n")
cat("  Positive beta: Volume increases with age (growth)\n")
cat("  Negative beta: Volume decreases with age (pruning/maturation)\n\n")

cat("Significant age effects (p < 0.05):\n")
sig_age <- age_effects_df[age_effects_df$`Pr(>|t|)` < 0.05, ]
if (nrow(sig_age) > 0) {
  for (i in 1:nrow(sig_age)) {
    direction <- ifelse(sig_age$Estimate[i] > 0, "INCREASES", "DECREASES")
    cat(sprintf("  - %s: %s with age (beta=%.1f, p=%.4f)%s\n",
                sig_age$model[i], direction, sig_age$Estimate[i],
                sig_age$`Pr(>|t|)`[i], sig_age$Significance[i]))
  }
} else {
  cat("  None\n")
}

cat("\nNon-significant age effects (p >= 0.05):\n")
nonsig_age <- age_effects_df[age_effects_df$`Pr(>|t|)` >= 0.05, ]
if (nrow(nonsig_age) > 0) {
  for (i in 1:nrow(nonsig_age)) {
    cat(sprintf("  - %s: No significant change (beta=%.1f, p=%.4f)\n",
                nonsig_age$model[i], nonsig_age$Estimate[i],
                nonsig_age$`Pr(>|t|)`[i]))
  }
} else {
  cat("  None\n")
}

# ===========================
# Compare with mask_vol model
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("COMPARISON WITH MASK_VOL MODEL\n")
cat(strrep("=", 80), "\n\n")

# Try to load previous results with mask_vol
maskvol_file <- file.path(OUTPUT_DIR, "summary_age_effects_6systems.csv")
if (file.exists(maskvol_file)) {
  age_effects_maskvol <- read_csv(maskvol_file, show_col_types = FALSE)

  cat("Age effect estimates comparison:\n")
  cat(sprintf("%-40s %12s %12s %10s\n", "System", "With MaskVol", "No MaskVol", "Difference"))
  cat(strrep("-", 80), "\n")

  for (sys in systems) {
    est_maskvol <- age_effects_maskvol %>% filter(model == sys) %>% pull(Estimate)
    est_no_maskvol <- age_effects_df %>% filter(model == sys) %>% pull(Estimate)

    if (length(est_maskvol) > 0 && length(est_no_maskvol) > 0) {
      diff <- est_no_maskvol - est_maskvol
      cat(sprintf("%-40s %12.2f %12.2f %10.2f\n",
                  sys, est_maskvol, est_no_maskvol, diff))
    }
  }

  cat("\nNote: Differences show how much the age effect changes when NOT controlling for ICV\n")
} else {
  cat("Previous results with mask_vol not found. Skipping comparison.\n")
}

# ===========================
# Final Summary
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 80), "\n\n")

cat("6 Systems analyzed:\n")
for (i in seq_along(sort(unique(volume_6way$System)))) {
  sys <- sort(unique(volume_6way$System))[i]
  cat(sprintf("  %d. %s\n", i, sys))
}

cat("\nOutput files created:\n")
cat("  1. lmm_6systems_no_maskvol_results.csv          - All coefficients\n")
cat("  2. summary_age_effects_6systems_no_maskvol.csv  - Age effects only\n")
cat("  3. summary_all_effects_6systems_no_maskvol.csv  - All fixed effects\n\n")

cat("Model specification:\n")
cat("  DV: Volume (mm3)\n")
cat("  Fixed effects: log(Age) + Sex + Dataset\n")
cat("  Random effects: (1 | ID) [random intercepts only]\n")
cat("  Optimizer: bobyqa\n")
cat("  Method: REML=TRUE\n\n")

cat("Key findings:\n")
cat(sprintf("  - %d/%d systems show significant age effects (p < 0.05)\n",
            nrow(sig_age), length(systems)))
cat(sprintf("  - %d systems show volume increases with age\n",
            sum(sig_age$Estimate > 0)))
cat(sprintf("  - %d systems show volume decreases with age\n",
            sum(sig_age$Estimate < 0)))

cat("\n")
cat(strrep("=", 80), "\n")

# ===========================
# Export Random Effects for Visualization
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("EXPORTING RANDOM EFFECTS FOR INDIVIDUAL TRAJECTORIES\n")
cat(strrep("=", 80), "\n\n")

# Extract random effects for each system
all_random_effects <- list()

for (sys in names(model_summaries)) {
  m <- model_summaries[[sys]]

  # Extract random effects
  ranef_id <- ranef(m)$ID

  # Convert to data frame
  ranef_df <- as.data.frame(ranef_id)
  ranef_df$ID_base <- rownames(ranef_df)
  rownames(ranef_df) <- NULL

  # Rename columns based on what's available
  if (ncol(ranef_df) == 2) {
    # Only random intercept
    colnames(ranef_df)[1] <- "random_intercept"
  } else if (ncol(ranef_df) == 3) {
    # Random intercept and slope
    colnames(ranef_df)[1] <- "random_intercept"
    colnames(ranef_df)[2] <- "random_slope"
  }

  # Add system name
  ranef_df$system <- as.character(sys)

  # Store
  all_random_effects[[as.character(sys)]] <- ranef_df

  cat(sprintf("  Extracted random effects for %s: %d subjects\n",
              sys, nrow(ranef_df)))
}

# Combine all random effects
combined_random_effects <- bind_rows(all_random_effects)

# Save to CSV
output_file_ranef <- file.path(OUTPUT_DIR, "lmm_6systems_no_maskvol_random_effects.csv")
write_csv(combined_random_effects, output_file_ranef)
cat(sprintf("\n✓ Random effects saved: %s\n", output_file_ranef))

cat("\n")
cat(strrep("=", 80), "\n")
cat("✓ ALL OUTPUT FILES CREATED\n")
cat(strrep("=", 80), "\n\n")

cat("Complete list of output files:\n")
cat("  1. lmm_6systems_no_maskvol_results.csv          - All coefficients\n")
cat("  2. summary_age_effects_6systems_no_maskvol.csv  - Age effects only\n")
cat("  3. summary_all_effects_6systems_no_maskvol.csv  - All fixed effects\n")
cat("  4. lmm_6systems_no_maskvol_random_effects.csv   - Random effects (NEW)\n")
cat("  (Subject info already exists: subject_info.csv)\n\n")

cat(strrep("=", 80), "\n")