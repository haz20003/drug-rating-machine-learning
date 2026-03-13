library(readr)

df <- read_csv("~/Desktop/intro to busn analytics project/drugs_side_effects_drugs_com.csv")

dim(df)
names(df)
head(df)
library(dplyr)
library(stringr)
library(forcats)
library(randomForest)
install.packages("caret")
library(caret)
install.packages("vip")
install.packages("pdp")
install.packages("patchwork")
install.packages("ggpubr")
library(dplyr); library(ggplot2); library(broom)
library(vip); library(pdp); library(patchwork); library(ggpubr)

# 1) 特征工程 + 严格处理缺失与类型
df_rf <- df %>%
  mutate(
    # 目标变量：把字符型评分转数值（必要时用 parse_number 更稳）
    rating = suppressWarnings(readr::parse_number(rating)),
    
    # 数值特征
    num_side_effects = if_else(!is.na(side_effects) & str_detect(side_effects, "[A-Za-z0-9]"),
                               str_count(side_effects, ",") + 1L, 0L),
    activity_num = suppressWarnings(as.numeric(str_remove(activity, "%"))),
    
    # 类别特征（2–3 个 factor）
    rx_otc = fct_explicit_na(as.factor(rx_otc), "Unknown"),
    pregnancy_category = fct_explicit_na(as.factor(pregnancy_category), "Unknown"),
    
    # drug_classes 类别太多：保留前 30 个高频，其余归为 Other，避免过多水平
    drug_classes = fct_lump_n(as.factor(drug_classes), n = 30, other_level = "Other")
  ) %>%
  # 只保留建模用到的列，并且**去掉任何 NA**
  select(rating, num_side_effects, activity_num, rx_otc, pregnancy_category, drug_classes) %>%
  tidyr::drop_na()

# 2) 线性回归（作为基线，可解释）
fit_lm <- lm(rating ~ num_side_effects + activity_num + rx_otc +
               pregnancy_category + drug_classes, data = df_rf)
summary(fit_lm)
tidy_lm <- broom::tidy(fit_lm) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    var = case_when(
      grepl("^rx_otc", term) ~ "rx_otc",
      grepl("^pregnancy_category", term) ~ "pregnancy_category",
      grepl("^drug_classes", term) ~ "drug_classes",
      grepl("^num_side_effects", term) ~ "num_side_effects",
      grepl("^activity_num", term) ~ "activity_num",
      TRUE ~ "other"
    ),
    direction = ifelse(estimate >= 0, "Positive","Negative"),
    signif = cut(p.value, c(-Inf,.001,.01,.05,.1,Inf),
                 labels = c("***","**","*",".",""))
  ) %>%
  filter(var != "other")

# 2) 分开取 TopN：drug_classes 取 15，其余每个分组各取 10
top_drug  <- tidy_lm %>%
  filter(var == "drug_classes") %>%
  slice_max(order_by = abs(estimate), n = 15, with_ties = FALSE)

top_other <- tidy_lm %>%
  filter(var != "drug_classes") %>%
  group_by(var) %>%
  slice_max(order_by = abs(estimate), n = 10, with_ties = FALSE) %>%
  ungroup()

tidy_lm_top <- dplyr::bind_rows(top_drug, top_other)

# 3) 画图
# ==== 1) 先选要关注的药物类别（可自行改）====
classes_focus <- c(
  "Antacids",
  "Atypical antipsychotics",
  "Contraceptives",
  "Antidiabetic combinations",
  "Insulin",
  "Miscellaneous analgesics"
)

# ==== 2) 整理线性回归系数（fit_lm 已训练好）====
tidy_lm_all <- tidy(fit_lm) %>%
  filter(term != "(Intercept)") %>%
  mutate(
    var = case_when(
      str_starts(term, "rx_otc")              ~ "rx_otc",
      str_starts(term, "pregnancy_category")  ~ "pregnancy_category",
      str_starts(term, "drug_classes")        ~ "drug_classes",
      str_starts(term, "num_side_effects")    ~ "num_side_effects",
      str_starts(term, "activity_num")        ~ "activity_num",
      TRUE ~ "other"
    ),
    clean_term = term |>                      # 用于更友好显示
      str_remove("^drug_classes") |>
      str_remove("^rx_otc") |>
      str_remove("^pregnancy_category"),
    direction = ifelse(estimate >= 0, "Positive","Negative"),
    sig = cut(p.value, c(-Inf,.001,.01,.05,.1,Inf), labels=c("***","**","*",".",""))
  ) %>% 
  filter(var != "other")

# 只保留我们关注的几类 drug_classes；其他变量全部保留
tidy_lm_focus <- tidy_lm_all %>%
  filter(var != "drug_classes" |
           clean_term %in% classes_focus)

# ==== 3) 系数图（只画关注类别）====
p_coef <- ggplot(tidy_lm_focus,
                 aes(x = reorder(clean_term, estimate),
                     y = estimate,
                     color = direction,
                     size = -log10(p.value))) +
  geom_point() +
  geom_hline(yintercept = 0, linetype = "dashed") +
  coord_flip() +
  facet_wrap(~ var, scales = "free_y", ncol = 1, strip.position = "top") +
  scale_size_continuous(name = "-log10(p.value)", range = c(2,6)) +
  scale_color_manual(values = c("Positive"="#1f77b4","Negative"="#d62728")) +
  labs(title = "Linear Model Coefficients (focused)",
       x = NULL, y = "Effect on rating") +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face="bold", size=12),
    legend.position = "bottom",
    axis.text.y = element_text(size = 10)
  )

# ==== 4) 原始关系的直观小图（同风格，辅助解释）====
p_rx  <- ggplot(df_rf, aes(rx_otc, rating)) +
  geom_boxplot(outlier.alpha = .15) +
  labs(title="Rating by Rx vs OTC", x=NULL, y=NULL) +
  theme_minimal()

p_preg <- ggplot(df_rf, aes(pregnancy_category, rating)) +
  geom_boxplot(outlier.alpha = .15) +
  labs(title="Rating by Pregnancy Category", x=NULL, y=NULL) +
  theme_minimal()

p_act <- ggplot(df_rf, aes(activity_num, rating)) +
  geom_point(alpha=.2) + geom_smooth(method="loess", se=FALSE) +
  labs(title="Activity vs Rating", x=NULL, y=NULL) +
  theme_minimal()

p_se  <- ggplot(df_rf, aes(num_side_effects, rating)) +
  geom_point(alpha=.2) + geom_smooth(method="loess", se=FALSE) +
  labs(title="Side effects vs Rating", x=NULL, y=NULL) +
  theme_minimal()

# ==== 5) 统一拼版：左侧系数图，右侧四宫格小图 ====
layout_right <- (p_rx | p_preg) / (p_act | p_se)
final_plot <- p_coef | layout_right
final_plot

# 3) 随机森林（允许非线性 + 交互），显式设定 na.action=na.omit（以防万一）
set.seed(42)
fit_rf <- randomForest(
  rating ~ num_side_effects + activity_num + rx_otc +
    pregnancy_category + drug_classes,
  data = df_rf,
  ntree = 500,
  importance = TRUE,
  na.action = na.omit
)

# 4) 简单评估对比
pred_lm <- predict(fit_lm, newdata = df_rf)
pred_rf <- predict(fit_rf, newdata = df_rf)

cat("\n== Metrics (in-sample) ==\n")
print(list(
  LM = caret::postResample(pred_lm, df_rf$rating),
  RF = caret::postResample(pred_rf, df_rf$rating)
))

# 5) RF 特征重要性（看哪些因子最关键）
varImpPlot(fit_rf, main = "Random Forest Variable Importance")

# 6) 通过CI去判断药物的成功概率
## ====== 安装/加载依赖 ======
need <- c("dplyr","ggplot2","broom","caret","pROC")
new  <- need[!need %in% installed.packages()[,1]]
if(length(new)) install.packages(new)
library(dplyr); library(ggplot2); library(broom)
library(caret); library(pROC)

set.seed(42)
## ====== 0) 数据准备：定义成功标签，并做基本清洗 ======
# 成功阈值可以改，比如 rating >= 8 视为“成功”
df_entry <- df_rf %>%
  mutate(
    success = if_else(rating >= 8, 1L, 0L),
    success = factor(success, levels = c(0,1))
  ) %>%
  # 只保留建模用到的列，去 NA
  select(success, num_side_effects, activity_num, rx_otc, pregnancy_category, drug_classes) %>%
  tidyr::drop_na()

## ====== 1) 90/10 分层切分（按 success 分层，保持类别比例稳定） ======
idx <- caret::createDataPartition(df_entry$success, p = 0.9, list = FALSE)
train_df <- droplevels(df_entry[idx, ])
test_df  <- droplevels(df_entry[-idx, ])

## ====== 2) 训练 Logistic 回归 ======
fit_logit <- glm(success ~ num_side_effects + activity_num + rx_otc +
                   pregnancy_category + drug_classes,
                 data = train_df, family = binomial)
summary(fit_logit)
e## ====== 3) 在测试集上预测 概率 + 95% 置信区间（基于链接函数的标准误） ======
# predict(..., type="link", se.fit=TRUE) -> 得到线性预测值及SE；再用plogis映射到[0,1]
pred_link <- predict(fit_logit, newdata = test_df, type = "link", se.fit = TRUE)
test_pred <- test_df %>%
  mutate(
    logit = pred_link$fit,
    se    = pred_link$se.fit,
    prob  = plogis(logit),
    lower = plogis(logit - 1.96*se),
    upper = plogis(logit + 1.96*se)
  )

## ====== 4) 指标评估（测试集）=====
# 4.1 AUC
roc_obj <- pROC::roc(response = test_pred$success, predictor = test_pred$prob, levels = c("0","1"))
auc_val <- as.numeric(pROC::auc(roc_obj))

# 4.2 0.5 阈值下的 Accuracy / Recall / Precision（可按需调阈值）
## ====== 4) 统一评估（Logistic vs RF-Classifier）======

# —— 4.1 Logistic：阈值评估 + ROC ----
thr <- 0.5
pred_prob_logit <- test_pred$prob  # 来自你前面 fit_logit 的预测
pred_cls_logit  <- factor(ifelse(pred_prob_logit >= thr, 1, 0), levels = c(0,1))
cm_logit <- caret::confusionMatrix(pred_cls_logit, test_pred$success, positive = "1")

roc_logit <- pROC::roc(response = test_pred$success, predictor = pred_prob_logit, levels = c("0","1"))
auc_logit <- as.numeric(pROC::auc(roc_logit))

cat("\n==== Logistic (Test) ====\n")
cat(sprintf("AUC: %.3f\n", auc_logit))
print(cm_logit)

library(forcats)   # 确保已经加载

## 0) 基于 df_rf 构造用于“成功/失败”分类的数据 df_entry
## 成功阈值：rating >= 8 视为 success = 1
df_entry <- df_rf %>%
  mutate(
    success = factor(if_else(rating >= 8, 1L, 0L), levels = c(0, 1)),
    # 再保险：把类别变量都处理成 factor，缺失并成 "Unknown"
    rx_otc  = fct_explicit_na(as.factor(rx_otc), "Unknown"),
    pregnancy_category = fct_explicit_na(as.factor(pregnancy_category), "Unknown"),
    # 再 lump 一次 drug_classes，防止太多小类；train/test 用同一套水平
    drug_classes = fct_lump_n(as.factor(drug_classes), n = 30, other_level = "Other")
  ) %>%
  select(success, num_side_effects, activity_num,
         rx_otc, pregnancy_category, drug_classes) %>%
  tidyr::drop_na()
## —— 关键：把 test 的因子水平“强制”对齐到 train 的水平集 —— 
align_levels <- function(tr, te) {
  cat_cols <- c("rx_otc","pregnancy_category","drug_classes","success")
  for (v in cat_cols) {
    # 若 test 出现 train 没有的水平，先并入 Other（若有），再强制设定 levels
    if (v == "drug_classes") {
      te[[v]] <- fct_other(te[[v]], keep = levels(tr[[v]]))
      # 确保 train/test 都含有 "Other" 这个水平（即使该折里没出现）
      tr[[v]] <- fct_expand(tr[[v]], "Other")
      te[[v]] <- fct_expand(te[[v]], "Other")
    }
    te[[v]] <- factor(te[[v]], levels = levels(tr[[v]]))
    tr[[v]] <- factor(tr[[v]], levels = levels(tr[[v]]))
  }
  num_cols <- c("num_side_effects","activity_num")
  for (v in num_cols) {
    tr[[v]] <- as.numeric(tr[[v]])
    te[[v]] <- as.numeric(te[[v]])
  }
  list(train = tr, test = te)
}
sets <- align_levels(train_df, test_df)
train_df <- sets$train
test_df  <- sets$test
# 注意：这是“分类RF”，和你之前做的“rating 回归RF”不同，起名 fit_rf_cls 以免冲突
set.seed(42)
fit_rf_cls <- randomForest(
  success ~ num_side_effects + activity_num + rx_otc + pregnancy_category + drug_classes,
  data = train_df,
  ntree = 500,
  importance = TRUE
)

pred_prob_rf <- predict(fit_rf_cls, newdata = test_df, type = "prob")[,2]
pred_cls_rf  <- factor(ifelse(pred_prob_rf >= thr, 1, 0), levels = c(0,1))
cm_rf <- caret::confusionMatrix(pred_cls_rf, test_df$success, positive = "1")

roc_rf <- pROC::roc(response = test_df$success, predictor = pred_prob_rf, levels = c("0","1"))
auc_rf <- as.numeric(pROC::auc(roc_rf))

cat("\n==== Random Forest Classifier (Test) ====\n")
cat(sprintf("AUC: %.3f\n", auc_rf))
print(cm_rf)

# 可选：RF 特征重要性
varImpPlot(fit_rf_cls, main = "Random Forest (Classifier) Variable Importance")

## ====== 5) ROC 曲线对比（含AUC） ======

# 假设前面已经有：
# roc_logit, auc_logit
# roc_rf,    auc_rf

# 1) 把两个 ROC 曲线都整理成一个 data.frame
roc_logit_df <- data.frame(
  fpr   = rev(1 - roc_logit$specificities),
  tpr   = rev(roc_logit$sensitivities),
  model = "Logistic"
)

roc_rf_df <- data.frame(
  fpr   = rev(1 - roc_rf$specificities),
  tpr   = rev(roc_rf$sensitivities),
  model = "Random Forest"
)

roc_both <- dplyr::bind_rows(roc_logit_df, roc_rf_df)

# 2) 画对比图
p_roc_compare <- ggplot(roc_both, aes(x = fpr, y = tpr, color = model)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  scale_color_manual(values = c("Logistic" = "#1f77b4", "Random Forest" = "#d62728")) +
  labs(
    title = sprintf("ROC Curve Comparison\nLogistic AUC = %.3f  |  RF AUC = %.3f",
                    auc_logit, auc_rf),
    x = "1 - Specificity (False Positive Rate)",
    y = "Sensitivity (True Positive Rate)",
    color = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  )

print(p_roc_compare)


## ====== 6) 校准图（Logistic vs RF） ======
calib_logit <- data.frame(prob = pred_prob_logit, success = test_df$success, model = "Logistic")
calib_rf    <- data.frame(prob = pred_prob_rf,    success = test_df$success, model = "Random Forest")
calib_all   <- rbind(calib_logit, calib_rf)

calib_df <- calib_all %>%
  group_by(model) %>%
  mutate(bin = cut(prob, breaks = quantile(prob, probs = seq(0,1,0.1), na.rm = TRUE),
                   include.lowest = TRUE)) %>%
  group_by(model, bin) %>%
  summarise(
    n = n(),
    pred_mean = mean(prob),
    obs_rate  = mean(as.integer(as.character(success))),
    se_obs    = sqrt(obs_rate * (1 - obs_rate) / n),
    .groups = "drop"
  ) %>%
  mutate(
    lower = pmax(obs_rate - 1.96*se_obs, 0),
    upper = pmin(obs_rate + 1.96*se_obs, 1)
  )

p_calib <- ggplot(calib_df, aes(x = pred_mean, y = obs_rate)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.015) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  facet_wrap(~ model) +
  labs(title = "Calibration Plot (Test Set)",
       x = "Mean predicted probability", y = "Observed success rate (95% CI)") +
  theme_minimal()
print(p_calib)

## ====== 7) 各药物类别的“预测成功概率 + 95%CI”条形图（按模型对比） ======
## 先把两种模型的预测概率与 test_df 拼在一起
calib_logit <- test_df %>% 
  mutate(prob = pred_prob_logit, model = "Logistic")

calib_rf <- test_df %>% 
  mutate(prob = pred_prob_rf, model = "Random Forest")

calib_all <- dplyr::bind_rows(calib_logit, calib_rf)

## 按药物类别聚合（过滤 n>=3，取各模型 Top10）
by_class <- calib_all %>%
  group_by(model, drug_classes) %>%
  summarise(
    n        = n(),
    prob_mean = mean(prob, na.rm = TRUE),
    prob_sd   = sd(prob,   na.rm = TRUE),
    .groups = "drop"
  ) %>%
  filter(n >= 3) %>%
  mutate(
    se    = prob_sd / sqrt(n),
    lower = pmax(prob_mean - 1.96 * se, 0),
    upper = pmin(prob_mean + 1.96 * se, 1)
  ) %>%
  group_by(model) %>%
  slice_max(order_by = prob_mean, n = 10, with_ties = FALSE) %>%
  ungroup()

## 画图（和之前一致）
p_bar <- ggplot(by_class,
                aes(x = reorder(drug_classes, prob_mean), y = prob_mean, fill = model)) +
  geom_col(position = position_dodge(width = 0.9)) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
                width = .2, position = position_dodge(width = 0.9)) +
  coord_flip() +
  labs(title = "Predicted Success Probability by Drug Class (Top 10 per model)",
       x = NULL, y = "Mean predicted P(success) with 95% CI", fill = NULL) +
  theme_minimal()

print(p_bar)

## ====== 8) 单条记录的“预测概率 + 95%CI”（仅 Logistic 有解析CI） ======
p_ci_points <- test_pred %>%      # test_pred 含 logit+se+prob+CI（来自 Logistic）
  sample_n(min(300, nrow(.))) %>%
  ggplot(aes(x = prob, y = lower, ymin = lower, ymax = upper, color = success)) +
  geom_errorbar(alpha = .35) +
  geom_point(alpha = .7) +
  scale_color_manual(values = c("0" = "#999999", "1" = "#E64B35")) +
  labs(title = "Per-record Predicted Probability with 95% CI (Logistic, Sampled Test Set)",
       x = "Predicted probability", y = "Lower CI (error bar shows 95% CI)", color = "Success") +
  theme_minimal()
print(p_ci_points)


## ====== 5) 可视化 1：ROC 曲线（含AUC） ======
p_roc <- ggplot() +
  geom_line(data = data.frame(
    tpr = rev(roc_obj$sensitivities),
    fpr = rev(1 - roc_obj$specificities)
  ), aes(x = fpr, y = tpr)) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  labs(title = sprintf("ROC Curve (AUC = %.3f)", auc_val),
       x = "1 - Specificity (FPR)", y = "Sensitivity (TPR)") +
  theme_minimal()

print(p_roc)

## ====== 6) 可视化 2：校准图（分位分箱，预测概率 vs 实际成功率） ======
# 将预测概率分成10个分箱，画均值预测 vs 实际命中率 + 误差条
calib_df <- test_pred %>%
  mutate(bin = cut(prob, breaks = quantile(prob, probs = seq(0,1,0.1)), include.lowest = TRUE)) %>%
  group_by(bin) %>%
  summarise(
    n = n(),
    pred_mean = mean(prob),
    obs_rate  = mean(as.integer(as.character(success))),
    se_obs    = sqrt(obs_rate * (1 - obs_rate) / n),
    .groups = "drop"
  ) %>%
  mutate(
    lower = pmax(obs_rate - 1.96*se_obs, 0),
    upper = pmin(obs_rate + 1.96*se_obs, 1)
  )

p_calib <- ggplot(calib_df, aes(x = pred_mean, y = obs_rate)) +
  geom_point(size = 2) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.015) +
  geom_abline(slope = 1, intercept = 0, linetype = "dotted") +
  labs(title = "Calibration Plot (Test Set)",
       x = "Mean predicted probability",
       y = "Observed success rate (with 95% CI)") +
  theme_minimal()

print(p_calib)

## ====== 7) 可视化 3：各药物类别的“预测成功概率 + 置信区间”条形图 ======
# 说明：这里用“预测概率的均值±1.96*SE(=sd/sqrt(n))”作为类别层面的CI，表达模型对该类的成功率估计及不确定性
top_classes <- test_pred %>%
  group_by(drug_classes) %>%
  summarise(n = n(),
            prob_mean = mean(prob),
            prob_sd   = sd(prob),
            .groups = "drop") %>%
  filter(n >= 10) %>%               # 只展示样本数>=10的类别，避免极端不稳
  mutate(se = prob_sd/sqrt(n),
         lower = pmax(prob_mean - 1.96*se, 0),
         upper = pmin(prob_mean + 1.96*se, 1)) %>%
  arrange(desc(prob_mean)) %>%
  slice_head(n = 12)                # 只取Top 12类，避免太拥挤

p_bar <- ggplot(top_classes,
                aes(x = reorder(drug_classes, prob_mean), y = prob_mean)) +
  geom_col(fill = "#2C7FB8") +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = .2) +
  coord_flip() +
  labs(title = "Predicted Success Probability by Drug Class (Test Set)",
       x = NULL, y = "Mean predicted P(success) with 95% CI") +
  theme_minimal()

print(p_bar)

## ====== 8) 可视化 4：关键自变量的“预测概率置信区间”散点图（单条记录层面） ======
# 展示每条记录的 prob 与 [lower, upper]，并按是否成功着色，便于直观查看模型不确定性
p_ci_points <- test_pred %>%
  sample_n(min(300, nrow(.))) %>% # 抽样一点，避免过密；想看全量就去掉这一行
  ggplot(aes(x = prob, y = lower, ymin = lower, ymax = upper, color = success)) +
  geom_errorbar(alpha = .35) +
  geom_point(alpha = .7) +
  scale_color_manual(values = c("0" = "#999999", "1" = "#E64B35")) +
  labs(title = "Per-record Predicted Probability with 95% CI (Sampled Test Set)",
       x = "Predicted probability", y = "Lower CI (error bar shows 95% CI)") +
  theme_minimal()

print(p_ci_points)

## ====== 9) 可选：输出逻辑回归的OR与95%置信区间（便于写报告） ======
or_table <- broom::tidy(fit_logit, conf.int = TRUE, conf.level = 0.95, exponentiate = TRUE) %>%
  arrange(desc(estimate))
head(or_table, 12)
ggplot(or_table %>% filter(!is.na(conf.low), estimate < 1e5),
       aes(x = reorder(term, estimate), y = estimate)) +
  geom_point() +
  geom_errorbar(aes(ymin = conf.low, ymax = conf.high), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  scale_y_log10() +
  coord_flip() +
  labs(title = "Odds Ratios (95% CI) from Logistic Regression",
       y = "Odds Ratio (log scale)", x = NULL) +
  theme_minimal()

