#!/usr/bin/env Rscript
# Simplified R script - 6 individual system models only

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(dplyr)
  library(readr)
})

# ===========================
# Configuration
# ===========================
INPUT_DIR <- "exp/mri/r_tables_ge_2_scans"
OUTPUT_DIR <- "exp/mri/r_tables_ge_2_scans"

cat(strrep("=", 80), "\n")
cat("LINEAR MIXED MODEL ANALYSIS - 6 INDIVIDUAL SYSTEMS\n")
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

cat(sprintf("✓ Loaded: %d rows, %d unique subjects\n",
            nrow(volume_6way), n_distinct(volume_6way$ID)))
cat(sprintf("✓ Systems: %d\n\n", n_distinct(volume_6way$System)))

# ===========================
# Fit Individual System Models
# ===========================
cat(strrep("=", 80), "\n")
cat("FITTING INDIVIDUAL MODELS FOR EACH SYSTEM\n")
cat(strrep("=", 80), "\n")
cat("\nFormula: volume ~ log_age + Sex + Dataset + mask_vol + (1 + log_age | ID)\n")
cat("Fallback: volume ~ log_age + Sex + Dataset + mask_vol + (1 | ID)\n\n")

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

  # Fit model with random slopes
  cat("  Fitting model with random slopes...\n")
  m_sys <- tryCatch({
    lmer(
      volume ~ log_age + Sex + Dataset + mask_vol + (1 + log_age | ID),
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
      cat("  ⚠ Singular fit detected.\n")
    }
    cat("  → Using random intercepts only\n\n")
    m_sys <- lmer(
      volume ~ log_age + Sex + Dataset + mask_vol + (1 | ID),
      data = sys_data
    )
  } else {
    cat("  ✓ Converged with random slopes\n\n")
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
  cat(sprintf("    Age effect:     β=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
              age_row$Estimate, age_row$`Std. Error`,
              age_row$`t value`, age_row$`Pr(>|t|)`))

  if ("SexM" %in% coef_sys$parameter) {
    sex_row <- coef_sys[coef_sys$parameter == "SexM", ]
    cat(sprintf("    Sex (M vs F):   β=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
                sex_row$Estimate, sex_row$`Std. Error`,
                sex_row$`t value`, sex_row$`Pr(>|t|)`))
  }

  if ("Datasetnew_england" %in% coef_sys$parameter) {
    dataset_row <- coef_sys[coef_sys$parameter == "Datasetnew_england", ]
    cat(sprintf("    Dataset (NE):   β=%.2f, SE=%.2f, t=%.2f, p=%.4f\n",
                dataset_row$Estimate, dataset_row$`Std. Error`,
                dataset_row$`t value`, dataset_row$`Pr(>|t|)`))
  }

  mask_row <- coef_sys[coef_sys$parameter == "mask_vol", ]
  cat(sprintf("    MaskVol (ICV):  β=%.2e, SE=%.2e, t=%.2f, p=%.4f\n\n",
              mask_row$Estimate, mask_row$`Std. Error`,
              mask_row$`t value`, mask_row$`Pr(>|t|)`))
}

# ===========================
# Combine and Save Results
# ===========================
cat(strrep("=", 80), "\n")
cat("SAVING RESULTS\n")
cat(strrep("=", 80), "\n\n")

# Combine all coefficients
all_system_coefs <- bind_rows(all_system_results)
output_file <- file.path(OUTPUT_DIR, "lmm_6systems_results.csv")
write_csv(all_system_coefs, output_file)
cat(sprintf("✓ All coefficients saved: %s\n", output_file))

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

output_file2 <- file.path(OUTPUT_DIR, "summary_age_effects_6systems.csv")
write_csv(age_effects, output_file2)
cat(sprintf("\n✓ Age effects summary saved: %s\n", output_file2))

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

output_file3 <- file.path(OUTPUT_DIR, "summary_all_effects_6systems.csv")
write_csv(fixed_effects_summary, output_file3)
cat(sprintf("✓ All effects summary saved: %s\n", output_file3))

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
cat("  The coefficient represents change in volume (mm³) per unit increase in log(age)\n")
cat("  Positive β: Volume increases with age (growth)\n")
cat("  Negative β: Volume decreases with age (pruning/maturation)\n\n")

cat("Significant age effects (p < 0.05):\n")
sig_age <- age_effects_df[age_effects_df$`Pr(>|t|)` < 0.05, ]
if (nrow(sig_age) > 0) {
  for (i in 1:nrow(sig_age)) {
    direction <- ifelse(sig_age$Estimate[i] > 0, "INCREASES", "DECREASES")
    cat(sprintf("  • %s: %s with age (β=%.1f, p=%.4f)%s\n",
                sig_age$model[i], direction, sig_age$Estimate[i],
                sig_age$`Pr(>|t|)`[i], sig_age$Significance[i]))
  }
} else {
  cat("  None\n")
}

cat("\nNon-significant age effects (p ≥ 0.05):\n")
nonsig_age <- age_effects_df[age_effects_df$`Pr(>|t|)` >= 0.05, ]
if (nrow(nonsig_age) > 0) {
  for (i in 1:nrow(nonsig_age)) {
    cat(sprintf("  • %s: No significant change (β=%.1f, p=%.4f)\n",
                nonsig_age$model[i], nonsig_age$Estimate[i],
                nonsig_age$`Pr(>|t|)`[i]))
  }
} else {
  cat("  None\n")
}

# ===========================
# Final Summary
# ===========================
cat("\n")
cat(strrep("=", 80), "\n")
cat("✓ ANALYSIS COMPLETE\n")
cat(strrep("=", 80), "\n\n")

cat("6 Systems analyzed:\n")
for (i in seq_along(sort(unique(volume_6way$System)))) {
  sys <- sort(unique(volume_6way$System))[i]
  cat(sprintf("  %d. %s\n", i, sys))
}

cat("\nOutput files created:\n")
cat("  1. lmm_6systems_results.csv          - All coefficients for all systems\n")
cat("  2. summary_age_effects_6systems.csv  - Age effects only (sorted by p-value)\n")
cat("  3. summary_all_effects_6systems.csv  - All fixed effects with significance\n\n")

cat("Model specification:\n")
cat("  DV: Volume (mm³)\n")
cat("  Fixed effects: log(Age) + Sex + Dataset + MaskVol\n")
cat("  Random effects: (1 | ID) [random intercepts only due to limited data]\n")
cat("  Optimizer: bobyqa\n")
cat("  Method: REML=TRUE\n\n")

cat("Key findings:\n")
cat(sprintf("  • %d/%d systems show significant age effects (p < 0.05)\n",
            nrow(sig_age), length(systems)))
cat(sprintf("  • %d systems show volume increases with age\n",
            sum(sig_age$Estimate > 0)))
cat(sprintf("  • %d systems show volume decreases with age\n",
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
output_file_ranef <- file.path(OUTPUT_DIR, "lmm_6systems_random_effects.csv")
write_csv(combined_random_effects, output_file_ranef)
cat(sprintf("\nRandom effects saved: %s\n", output_file_ranef))

# Also save subject-level information for easier plotting
subject_info <- volume_6way %>%
  group_by(ID_base) %>%
  summarise(
    gender = first(gender),
    dataset = first(dataset),
    mean_mask_vol = mean(mask_vol, na.rm = TRUE),
    min_age = min(age, na.rm = TRUE),
    max_age = max(age, na.rm = TRUE),
    n_observations = n()
  )

output_file_info <- file.path(OUTPUT_DIR, "subject_info.csv")
write_csv(subject_info, output_file_info)
cat(sprintf("Subject info saved: %s\n", output_file_info))

cat("\n")
cat(strrep("=", 80), "\n")

# ===========================
# Final Summary
# ===========================
cat("ANALYSIS COMPLETE\n")
cat(strrep("=", 80), "\n\n")

cat("6 Systems analyzed:\n")
for (i in seq_along(sort(unique(volume_6way$System)))) {
  sys <- sort(unique(volume_6way$System))[i]
  cat(sprintf("  %d. %s\n", i, sys))
}

cat("\nOutput files created:\n")
cat("  1. lmm_6systems_results.csv          - All coefficients for all systems\n")
cat("  2. summary_age_effects_6systems.csv  - Age effects only (sorted by p-value)\n")
cat("  3. summary_all_effects_6systems.csv  - All fixed effects with significance\n")
cat("  4. lmm_6systems_random_effects.csv   - Random effects for individual trajectories\n")
cat("  5. subject_info.csv                  - Subject-level information\n\n")

cat("Model specification:\n")
cat("  DV: Volume (mm3)\n")
cat("  Fixed effects: log(Age) + Sex + Dataset + MaskVol\n")
cat("  Random effects: (1 | ID) [random intercepts only due to limited data]\n")
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