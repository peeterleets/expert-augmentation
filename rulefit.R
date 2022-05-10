

# Libraries ====
library(rtemis)
library(tidyverse)
library(DT)
library(splitstackshape)
library(caret)
library(rsample)
library(pROC)
library(glmnet)
library(ggplot2)
library(tibble)
library(ggtext)
library(gbm)
library(pre)
library(datasets)
library(forcats)
library(grid)
library(MLmetrics)



set.seed(69)

setwd('C:\\Users\\peeterl\\iCloudDrive\\Desktop\\MAKATÖÖ\\andmed')
load('C:\\Users\\peeterl\\Documents\\MAKA_ANDMED\\leets_tootu_isik_final_uus_horisont.RData')
# '- Check data ====
checkData(tootu_isik_final)


#####   DATA PREPARATION   #####


tootu_isik_final <- tootu_isik_final %>%
  mutate(across(c(VIIM_TEGEVUS_LIIKGR, VIIM_HOIVE_LOPP_POHJUSGR, VIIM_HOIVE_VALDKOND,
                  MAAKOND_NIMI_34, KODAKONDSUS, JUHATUSE_LIIGE, EMAILOLEMAS, TOOVOIME,
                  KAT_B, ARVUTIOSKUS, HOIVESSEREGTOOTUSEALGUSEST180, VIIM_HOIVE_LIIKGR), as.factor)) %>% 
  mutate(VANUS_34 = factor(VANUS_34, ordered = TRUE, levels = c("< 20", "20-30", "31-40", "41-50", "51-60", "> 60")),
         HARIDUS_TASE_GRUPP_NIMI_34 = factor(HARIDUS_TASE_GRUPP_NIMI_34, ordered = TRUE, levels = c("None/Unknown", "Primary level", "Secondary level", "Tertiary level")),
         AEG_HOIVEST = factor(AEG_HOIVEST, ordered = TRUE, levels = c("KUNI_3_KUUD", "3_KUNI_6_KUUD", "6_KUNI_12_KUUD", "1_KUNI_2_AASTAT", "2_KUNI_3_AASTAT", "3_KUNI_5_AASTAT", "YLE_5_AASTA", "TEADMATA_PUUDUB")),
         TOOSOOVE_ISCO3GR = factor(TOOSOOVE_ISCO3GR, ordered = TRUE, levels = c("0", "1", "2", "3", "4", "5", "6_JA_ROHKEM")),
         VIIM_HOIVE_KESTUSGR = factor(VIIM_HOIVE_KESTUSGR, ordered = TRUE, levels = c("KUNI_3_KUUD", "3_KUNI_12_KUUD", "1_KUNI_3_AASTAT", "3_KUNI_10_AASTAT", "YLE_10_AASTA", "TEADMATA_PUUDUB")),
         KEELEOSKUS_TASE_ET_34 = factor(KEELEOSKUS_TASE_ET_34, ordered = TRUE, levels = c("PUUDUB", "TEADMATA", "A1", "A2", "B1", "B2", "C1", "C2")),
         VARASEMADTOOTUSED = factor(VARASEMADTOOTUSED, ordered = TRUE, levels = c("0", "1", "2", "3", "4", "5", "6_JA_ROHKEM")),
         )

### preparing training, test, and future data samples ###

tootu_isik_pre2020_sample <- tootu_isik_final %>% 
  filter(ARVEL_ALGUS_KP < as.Date('2019-09-13')) %>% 
  rename("y" = "HOIVESSEREGTOOTUSEALGUSEST180")

split <- initial_split(tootu_isik_pre2020_sample, prop = 0.2, strata = y)
training.data.testo <- training(split)

test.data.wew <- tootu_isik_pre2020_sample %>% 
  filter(!(SEISUND_ID %in% training.data$SEISUND_ID)) %>% 
  stratified(group = c("EMAILOLEMAS"), size = 0.1)

test.data.new <- tootu_isik_final %>% 
  filter(ARVEL_ALGUS_KP >= as.Date('2020-03-12')) %>%
  filter(ARVEL_ALGUS_KP <= as.Date('2020-07-05')) %>% 
  rename("y" = "HOIVESSEREGTOOTUSEALGUSEST180") 
  #stratified(group = c("y"), size = 0.5)

test.data.new.long <- tootu_isik_final %>% 
  filter(ARVEL_ALGUS_KP >= as.Date('2020-03-12')) %>% 
  rename("y" = "HOIVESSEREGTOOTUSEALGUSEST180")
  #stratified(group = c("y"), size = 0.2) 

checkData(training.data)



#####   TRAINING BASE MODELS   #####

# Random Forest base model:
RF.x <- s.RANGER(
  training.data[,-c(1:4)], 
  test.data[,-c(1:4)],
  y = training.data$y,
  y.test = test.data$y,
  n.trees = 500,
  mtryStart = 10,
  stepFactor = 1,
  autotune = TRUE,
  probability = TRUE
)

# GBM base model:
GBM.x <- gbm::gbm(
  data = training.data[,-c(1:3)] %>% 
    mutate(KEELEOSKUS_TASE_ET_34 = as.factor(KEELEOSKUS_TASE_ET_34),
           AEG_HOIVEST = as.factor(AEG_HOIVEST),
           VIIM_HOIVE_KESTUSGR = as.factor(VIIM_HOIVE_KESTUSGR),
           HARIDUS_TASE_GRUPP_NIMI_34 = as.factor(HARIDUS_TASE_GRUPP_NIMI_34),
           VARASEMADTOOTUSED = as.factor(VARASEMADTOOTUSED),
           TOOSOOVE_ISCO3GR = as.factor(TOOSOOVE_ISCO3GR),
           y = as.character(y)
    ),
  formula = y ~ .,
  #bag.fraction = 0.5,
  shrinkage = 0.01, 
  interaction.depth = 10, 
  n.trees = 500
  
)


# Boosting based RuleFit model:
ruleFeat.x <- s.RULEFEAT(training.data.test[,-c(1:4)], 
                       #test.data[,-c(1:4)],
                       y = training.data.test$y,
                       #y.test = test.data$y,
                       #cases.by.rules = ruleFeat.x$mod$cases.by.rules,
                       n.trees = 500,
                       #meta.lambda = 0.055,
                       gbm.params = list(bag.fraction = 0.5, shrinkage = 0.001, interaction.depth = 10, ipw = T)
)

# Exctracting 142 LASSO-selected rules:

datatable(ruleFeat.x$mod$rules.selected.coef.er) %>%
  formatRound(columns = c("Coefficient", "Empirical_Risk"), digits = 3)


training.data.cxr <- matchCasesByRules(training.data, ruleFeat.x$mod$rules.selected)
test.data.cxr <- matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected)
test.data.new.cxr <- matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected)

# Ridge model (RuleFit linear model with all 142 rules):
ridge.x <- s.GLMNET(training.data.cxr, training.data$y,
                    alpha = 0
)





#####   SURVEY RESPONSES   #####


vastused <- read_delim(file = "C:\\Users\\peeterl\\iCloudDrive\\Desktop\\MAKATÖÖ\\results-survey777742(1).csv",
                     delim = ",") %>% 
  select(R85:R21)

vastused[vastused == "Ei oska öelda"] <- NA
vastused[vastused == "Oluliselt väiksem"] <- "1"
vastused[vastused == "Väiksem"] <- "0.75"
vastused[vastused == "Sama"] <- "0.5"
vastused[vastused == "Suurem"] <- "0.25"
vastused[vastused == "Oluliselt suurem"] <- "0"

vastused <- data.frame(lapply(vastused, function(x) as.numeric(as.character(x))))
sapply(vastused, class)

# calculating the mean and SD for expert assessments:
vastused_mean <- as.data.frame(lapply(vastused, function(x) mean(x, na.rm=T)))
vastused_sd <- as.data.frame(lapply(vastused, function(x) sd(x, na.rm=T)))

# calculating and plotting variable importance for expert assessments vs RF model:
test_varimp <- arvuta_varimp(vastused_mean)
test_varimp <- gather(test_varimp, var, imp, AEG_HOIVEST:OPPEVALDKOND)

test_varimp[test_varimp == "AEG_HOIVEST"] <- "Time since last employ-\nment spell"
test_varimp[test_varimp == "TOOTAMISE_PERIOODE"] <- "N employment spells\n in the last 3 years"
test_varimp[test_varimp == "VANUS_34"] <- "Age group"
test_varimp[test_varimp == "DAYS_APPOINTED"] <- "Length of assigned UIB"
test_varimp[test_varimp == "VALJAMAKSEDKUUD24ANTE"] <- "N months with pay-\nment in the last 2 years"
test_varimp[test_varimp == "TOOVOIME"] <- "Work capacity"
test_varimp[test_varimp == "EMAILOLEMAS"] <- "Has e-mail account"
test_varimp[test_varimp == "KAT_B"] <- "Has driver's license"
test_varimp[test_varimp == "TOOTU_PAEVI_3A"] <- "N unemployment days\n in the past 3 years"
test_varimp[test_varimp == "VIIM_HOIVE_LOPP_POHJUSGR"] <- "Reason of ending last\n employment spell"
test_varimp[test_varimp == "OPPEVALDKOND"] <- "Field of education"
test_varimp[test_varimp == "SUM_TK_YLEVAL_TS_SUHE"] <- "Competition for suitable\n job vacancies"
test_varimp[test_varimp == "VIIM_HOIVE_KESTUSGR"] <- "Duration of last employ-\nment spell"
test_varimp[test_varimp == "PALGATOETUS_3APOS"] <- "Received wage subsidy\n in the past 3 years"

range01 <- function(x){(x-min(x))/(max(x)-min(x))}

# PLOT VARIMP FOR EXPERT ASSESSMENTS:
exp_varimp <- data.frame(test_varimp$var,range01(test_varimp$imp))
ggplot(data = exp_varimp) +
  geom_bar(stat="identity", fill = "#00bfc4", aes(x = range01.test_varimp.imp., y = reorder(test_varimp.var, range01.test_varimp.imp.))) +
  scale_x_continuous(name = "Variable importance (exp. assessment)") +
  scale_y_discrete(name = "") +
  theme_bw() +
  theme(text = element_text(family = 'serif', size = 12)) +
  theme(axis.title.x = element_text(hjust=-0.2))

# PLOT VARIMP FOR EMPIRICAL MODEL:
emp_varimp <- tibble::rownames_to_column(data.frame(range01(RF.x$varimp)), "var")
ggplot(emp_varimp %>% 
         mutate(highlight = ifelse(var %in% c("AEG_HOIVEST",
                                              "TOOTAMISE_PERIOODE",
                                              "VANUS_34",
                                              "DAYS_APPOINTED",
                                              "VALJAMAKSEDKUUD24ANTE",
                                              "TOOVOIME",
                                              "EMAILOLEMAS",
                                              "KAT_B",
                                              "TOOTU_PAEVI_3A",
                                              "VIIM_HOIVE_LOPP_POHJUSGR",
                                              "OPPEVALDKOND",
                                              "SUM_TK_YLEVAL_TS_SUHE",
                                              "VIIM_HOIVE_KESTUSGR",
                                              "PALGATOETUS_3APOS"), 
                                   T, F))) +
  geom_bar(stat="identity", aes(x = range01.RF.x.varimp., y = reorder(var, range01.RF.x.varimp.), fill = highlight)) +
  scale_fill_manual(name = "", labels = c("FALSE", "TRUE"), values = c("grey", "#00bfc4")) +
  scale_x_continuous(name = "Variable importance (emp. model)") +
  scale_y_discrete(name = "",
                   labels = rev(c("Competition for suitable job vacancies",
                              "% exiting unemployment in the last 30 days",
                              "N registering as unemployed at the same time",
                              "N months with payment in the last 2 years",
                              "N unemployment days in the past 3 years",
                              "Field of last employment spell",
                              "N employment spells in the last 3 years",
                              "N of suitable job vacancies",
                              "Time since last employment spell",
                              "ISCO code of last employment spell",
                              "Assigned daily UIB payment",
                              "Field of education",
                              "Age group",
                              "Region",
                              "Language skills (Estonian)",
                              "N unique employers in the last 3 years",
                              "Duration of last employment spell",
                              "Level of education",
                              "Reason of ending last employment spell",
                              "N short job spells in the last 3 years",
                              "Length of assigned UIB",
                              "N unemployment spells in the last 3 years",
                              "Computer skills",
                              "Work capacity",
                              "Length of assigned unemployment allowance",
                              "Citizenship",
                              "Has driver's license",
                              "Last status before unemployment",
                              "Type of last employment",
                              "Has e-mail account",
                              "Received wage subsidy in the past 3 years",
                              "Is board member"
                              ))) +
  theme_bw() +
  guides(fill="none") +
  theme(text = element_text(family = 'serif', size = 12))




# CALCULATING DELTA RANK - DIFFERENCE OF EMPIRICAL AND EXPERT ASSESSMENTS:

.rules_assessment <- data.frame(ruleFeat.x$mod$rules.selected.coef.er) %>% 
  mutate(Rule_ID = paste0("R",Rule_ID))

.vastused_long <- merge(gather(vastused_mean, Rule_ID, expert_assessment, R85:R21),
                            gather(vastused_sd, Rule_ID, ea_sd, R85:R21),
                            by = "Rule_ID")

assessments <- merge(x=.vastused_long, 
                     y=.rules_assessment, 
                     all.x=TRUE) %>% 
  mutate(.se = ea_sd / sqrt(7),
         .ci.lower = expert_assessment - qt(1 - (0.05 / 2), 7 - 1) * .se,
         .ci.upper = expert_assessment + qt(1 - (0.05 / 2), 7 - 1) * .se) %>% 
  #mutate(empirical_rank = dense_rank(desc(Empirical_Risk))) %>%
  #mutate(ea_rank = dense_rank(desc(expert_assessment))) %>% 
  mutate(empirical_risk_quintiles = cut(Empirical_Risk, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))) %>% 
  #mutate(delta_rank = ifelse(!is.na(ea_rank), empirical_rank - ea_rank, 0)) %>% 
  #mutate(delta_rank_abs = abs(delta_rank)) %>% 
  #mutate(delta_rank_final = delta_rank_abs / ea_sd) %>% 
  mutate(vahe = Empirical_Risk - expert_assessment) %>% 
  mutate(abs_vahe = abs(vahe)) %>% 
  mutate(abs_vahe_x_stdv = abs_vahe / ea_sd) %>% 
  mutate(abs_vahe_quintiles = cut(abs_vahe_x_stdv, 5)) %>% 
  mutate(penalty_rank = ifelse(abs_vahe_quintiles == "(1.6,2]", 5,
                                     ifelse(abs_vahe_quintiles == "(1.21,1.6]", 4,
                                            ifelse(abs_vahe_quintiles == "(0.814,1.21]", 3,
                                                   ifelse(abs_vahe_quintiles == "(0.421,0.814]", 2,
                                                          ifelse(abs_vahe_quintiles == "(0.0256,0.421]", 1,
                                                                 NA))))))

# BOXPLOTS: 20 rules vs 142 rules risk score:
rbind(assessments %>% 
        mutate(sample = "20 Expert-\nassessed rules") %>% 
        dplyr::select(Empirical_Risk, sample),
      .rules_assessment %>% 
        mutate(sample = "142 LASSO-\nselected rules") %>% 
        dplyr::select(Empirical_Risk, sample)) %>%
  ggplot() +
  geom_boxplot(aes(x = sample, y = Empirical_Risk, fill = sample), alpha = 1) +
  stat_summary(aes(x = sample, y = Empirical_Risk), fun=mean, geom="point", shape=20, size=5) +
  scale_x_discrete(name = "Set of decision rules") +
  scale_y_continuous(name = "Empirical risk") +
  labs(fill="") +
  guides(fill="none") +
  theme_bw() +
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
  theme(text = element_text(family = 'serif', size = 14))

# BOXPLOTS: difference between empirical risk and expert assessment:
gather(assessments, measure, value, c("Empirical_Risk", "expert_assessment")) %>% 
  ggplot() +
  geom_boxplot(aes(x = measure, y = value, fill = measure)) +
  scale_fill_manual(name = "", labels = c("Empirical risk", "Expert assessed risk"), values = c("#f8766d", "#00bfc4")) +
  scale_x_discrete(name = "") +
  stat_summary(aes(x = measure, y = value), fun=mean, geom="point", shape=20, size=5) +
  scale_y_continuous(name = "Risk of not resuming\n work within 180 days") +
  theme_bw() +
  theme(axis.text.x=element_blank(), axis.ticks.x=element_blank()) +
  theme(text = element_text(family = 'serif', size = 14)) +
  guides(fill="none")



gather(assessments %>% mutate(order = dense_rank(abs_vahe_x_stdv)), measure, value, c("abs_vahe_x_stdv", "abs_vahe")) %>% 
  ggplot(aes(x = fct_reorder(Rule_ID, order))) +
  geom_bar(aes(y = value, fill = measure), position = "identity", stat = "identity", alpha = 1) +
  #geom_bar(aes(y = abs_vahe), stat = "identity", alpha = 0.5) +
  geom_errorbar(aes(ymin=0, ymax=ea_sd), position = "identity", alpha = 0.5, width = 0, size = 1) +
  scale_x_discrete(name = "Rule") +
  scale_y_continuous(name = expression(paste("Penalty"))) +
  scale_fill_manual(name = "", labels = c("Absolute difference between empirical and expert assessed risk (|∆Risk|)", "|∆Risk| divided by the standard deviation of experts' responses (vertical black bars)"), values = c("#f8766d", "#00bfc4")) +
  geom_hline(yintercept = 1.6, alpha = .5) +
  geom_hline(yintercept = 1.21, alpha = .5) +
  geom_hline(yintercept = 0.814, alpha = .5) +
  geom_hline(yintercept = 0.421, alpha = .5) +
  geom_hline(yintercept = 0.0256, alpha = .5) +
  # Create a text
  annotation_custom(grobTree(textGrob("R5", x=0.01,  y=0.97, hjust=0,
                            gp=gpar(fontsize=13, fontface="italic", fontfamily="serif")))) +
  annotation_custom(grobTree(textGrob("R4", x=0.01,  y=0.74, hjust=0,
                                      gp=gpar(fontsize=13, fontface="italic", fontfamily="serif")))) +
  annotation_custom(grobTree(textGrob("R3", x=0.01,  y=0.56, hjust=0,
                                      gp=gpar(fontsize=13, fontface="italic", fontfamily="serif")))) +
  annotation_custom(grobTree(textGrob("R2", x=0.01,  y=0.39, hjust=0,
                                      gp=gpar(fontsize=13, fontface="italic", fontfamily="serif")))) +
  annotation_custom(grobTree(textGrob("R1", x=0.01,  y=0.21, hjust=0,
                                      gp=gpar(fontsize=13, fontface="italic", fontfamily="serif")))) +
  theme_bw() +
  theme(text = element_text(family = 'serif', size = 14)) +
  theme(legend.position="bottom")+
  guides(fill=guide_legend(nrow=2, byrow = TRUE))

# Plotting distribution of delta rank vs n of rules
gather(assessments %>%
         mutate(order = dense_rank(Empirical_Risk)), measure, value, c("Empirical_Risk", "expert_assessment")) %>% 
  ggplot(aes(x = fct_reorder(Rule_ID,order), y = value, color = measure, group = measure)) +
  geom_errorbar(aes(ymin=.ci.lower, ymax=.ci.upper, color = measure), width = 0.3, size = 1) +
  geom_point(size = 3) +
  scale_x_discrete(name = "Rule") +
  scale_y_continuous(name = "Risk of not resuming work within 180 days") +
  geom_hline(yintercept = 1, color = "black", linetype = "dashed") +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed") +
  theme_bw() +
  guides(color="none") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(text = element_text(family = 'serif', size = 14))
  

# Dividing empirical risks of the 20 selected rules into 5 bins and plotting the mean risk for each:
rbind(assessments %>% 
        mutate(sample = "20 Expert-assessed rules") %>% 
        select(Empirical_Risk, sample),
      .rules_assessment %>% 
        mutate(sample = "142 LASSO-selected rules") %>% 
        select(Empirical_Risk, sample)) %>% 
  
  mutate(empirical_risk_quintiles = cut(Empirical_Risk, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))) %>% 
  group_by(sample, empirical_risk_quintiles) %>% 
  dplyr::summarize(.mean = mean(Empirical_Risk),
                   .sd = sd(Empirical_Risk),
                   .n = n()) %>% 
  mutate(.se = .sd / sqrt(.n),
         .ci.lower = .mean - qt(1 - (0.05 / 2), .n - 1) * .se,
         .ci.upper = .mean + qt(1 - (0.05 / 2), .n - 1) * .se) %>% 
  ggplot(aes(x = empirical_risk_quintiles, y = .mean, fill = sample)) + 
  geom_bar(stat="identity", position="dodge", alpha = 1) +
  geom_errorbar(aes(ymin=.ci.lower, ymax=.ci.upper), position = position_dodge(width = 0.8), width = 0.5) +
  scale_x_discrete(name = "Distribution of empirical risk scores by quintiles") +
  scale_y_continuous(name = "Mean empirical risk") +
  labs(fill='') +
  guides(color="none") +
  #guides(fill="none") +
  theme_bw() +
  theme(legend.position="bottom") +
  theme(text = element_text(family = 'serif', size = 14))




## PREDICT FOR RF:
load("RF.x")
load("test.data")
load("test.data.new")
load("test.data.new.long")

# for test data:

test.data <- cbind(test.data ,
                   data.frame(predict(RF.x, newdata = test.data, type="response")) %>% 
                     rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
test.data$RF.prob.cut <- cut(test.data$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
test.data <- test.data %>% 
  mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

pROC::auc(test.data$y, test.data$RF.prob)
#test.data$RF.estimate.cut <- cut(test.data$RF.estimate, breaks=c(0,0.25,0.5,0.75, 1))

RF.groups <- unique(test.data$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$RF.prob)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF.test = gather(data.frame(RF.groups = RF.groups,
                            Predicted = RF.oodatav,
                            Actual = RF.tegelik,
                            sample = "Test data"),
                 var, value, Actual:Predicted)


# for new test data:
test.data.new <- cbind(test.data.new ,
                       data.frame(predict(RF.x, newdata = test.data.new, type="response")) %>% 
                         rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
test.data.new$RF.prob.cut <- cut(test.data.new$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
test.data.new <- test.data.new %>% 
  mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

pROC::auc(test.data.new$y, test.data.new$RF.prob)

RF.groups <- unique(test.data.new$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data.new %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$RF.prob)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF.test.new = gather(data.frame(RF.groups = RF.groups,
                                Predicted = RF.oodatav,
                                Actual = RF.tegelik,
                                sample = "Short-term future test data"),
                     var, value, Actual:Predicted)


# for new long test data:
test.data.new.long <- cbind(test.data.new.long ,
                            data.frame(predict(RF.x, newdata = test.data.new.long, type="response")) %>% 
                              rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
test.data.new.long$RF.prob.cut <- cut(test.data.new.long$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
test.data.new.long <- test.data.new.long %>% 
  mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

pROC::auc(test.data.new.long$y, test.data.new.long$RF.prob)

RF.groups <- unique(test.data.new.long$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data.new.long %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$RF.prob)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF.test.new.long = gather(data.frame(RF.groups = RF.groups,
                                     Predicted = RF.oodatav,
                                     Actual = RF.tegelik,
                                     sample = "Mid-term future test data"),
                          var, value, Actual:Predicted)



ggplot(data = rbind(RF.test, rbind(RF.test.new, RF.test.new.long))) +
  geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = RF.groups, y = value, fill = var)) +
  geom_line(size = 1, stat='identity', aes(x = RF.groups, y = value, color = var, group = var)) +
  facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
  scale_x_discrete(name = "Quintile distribution of risk scores predicted with Random Forest") +
  scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
  theme_bw() +
  labs(fill='') +
  guides(fill="none") +
  guides(color="none") +
  theme(legend.position="bottom") +
  theme(text = element_text(family = 'serif', size = 14)) 





##########################################
############## PREDICTIONS FOR BASE MODELS
##########################################


## PREDICT FOR GBM BASE MODEL:
load("GBM.x")

# for test data:

test.data$GBM.prob = predict(GBM.x, newdata = test.data, type="response")
test.data$GBM.prob.cut <- cut(test.data$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data$y, test.data$GBM.prob)
#test.data$RF.estimate.cut <- cut(test.data$RF.estimate, breaks=c(0,0.25,0.5,0.75, 1))

GBM.groups <- unique(test.data$GBM.prob.cut)
GBM.oodatav <- c(NA,NA,NA,NA,NA)
GBM.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in GBM.groups) {
  tmp <- test.data %>% 
    filter(GBM.prob.cut == group)
  
  GBM.oodatav[i] = mean(tmp$GBM.prob)
  GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
  
  i = i + 1
}

GBM.test = gather(data.frame(GBM.groups = GBM.groups,
                             Predicted = GBM.oodatav,
                             Actual = GBM.tegelik,
                             sample = "Test data"),
                  var, value, Actual:Predicted)


# for new test data:
test.data.new$GBM.prob = predict(GBM.x, newdata = test.data.new, type="response")
test.data.new$GBM.prob.cut <- cut(test.data.new$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, test.data.new$GBM.prob)

GBM.groups <- unique(test.data.new$GBM.prob.cut)
GBM.oodatav <- c(NA,NA,NA,NA,NA)
GBM.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in GBM.groups) {
  tmp <- test.data.new %>% 
    filter(GBM.prob.cut == group)
  
  GBM.oodatav[i] = mean(tmp$GBM.prob)
  GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
  
  i = i + 1
}

GBM.test.new = gather(data.frame(GBM.groups = GBM.groups,
                                 Predicted = GBM.oodatav,
                                 Actual = GBM.tegelik,
                                 sample = "Short-term future test data"),
                      var, value, Actual:Predicted)


# for new long test data:
test.data.new.long$GBM.prob = predict(GBM.x, newdata = test.data.new.long, type="response")
test.data.new.long$GBM.prob.cut <- cut(test.data.new.long$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new.long$y, test.data.new.long$GBM.prob)

GBM.groups <- unique(test.data.new.long$GBM.prob.cut)
GBM.oodatav <- c(NA,NA,NA,NA,NA)
GBM.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in GBM.groups) {
  tmp <- test.data.new.long %>% 
    filter(GBM.prob.cut == group)
  
  GBM.oodatav[i] = mean(tmp$GBM.prob)
  GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
  
  i = i + 1
}

GBM.test.new.long = gather(data.frame(GBM.groups = GBM.groups,
                                      Predicted = GBM.oodatav,
                                      Actual = GBM.tegelik,
                                      sample = "Mid-term future test data"),
                           var, value, Actual:Predicted)



ggplot(data = rbind(GBM.test, rbind(GBM.test.new, GBM.test.new.long))) +
  geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = GBM.groups, y = value, fill = var)) +
  geom_line(size = 1, stat='identity', aes(x = GBM.groups, y = value, color = var, group = var)) +
  facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
  scale_x_discrete(name = "Quintile distribution of risk scores predicted with Gradient Boosting") +
  scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
  theme_bw() +
  labs(fill='') +
  guides(color="none") +
  guides(fill="none") +
  theme(legend.position="bottom") +
  theme(text = element_text(family = 'serif', size = 14)) 






## PREDICT FOR RuleFit, ALL 3199 RULES:

# for test data:
test.data <- cbind(test.data,
                   data.frame(predict(ruleFeat.x, newdata = test.data, type="prob")) %>% 
                     rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
test.data$ruleFeat.prob.cut <- cut_number(test.data$ruleFeat.prob, 5)

pROC::auc(test.data$y, as.numeric(test.data$ruleFeat.prob))


# for new test data:
test.data.new <- cbind(test.data.new,
                       data.frame(predict(ruleFeat.x, newdata = test.data.new, type="prob")) %>% 
                         rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
test.data.new$ruleFeat.prob.cut <- cut_number(test.data.new$ruleFeat.prob, 5)

pROC::auc(test.data.new$y, as.numeric(test.data.new$ruleFeat.prob))


# for new long test data:
test.data.new.long <- cbind(test.data.new.long,
                            data.frame(predict(ruleFeat.x, newdata = test.data.new.long, type="prob")) %>% 
                              rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
test.data.new.long$ruleFeat.prob.cut <- cut_number(test.data.new.long$ruleFeat.prob, 5)

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$ruleFeat.prob))




## PREDICT FOR Ridge, 142 REEGLIT:
load("ridge.x")

# for test data

test.data$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected), type = "response")
test.data$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected), type = "class")
test.data$ridge.prob.cut <- cut(test.data$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))


pROC::auc(test.data$y, as.numeric(test.data$ridge.prob))

ridge.groups <- unique(test.data$ridge.prob.cut)
ridge.oodatav <- c(NA,NA,NA,NA,NA)
ridge.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in ridge.groups) {
  tmp <- test.data %>% 
    filter(ridge.prob.cut == group)
  
  ridge.oodatav[i] = mean(tmp$ridge.prob)
  ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}


ridge.test <- gather(data.frame(ridge.groups = ridge.groups,
                                Predicted = ridge.oodatav,
                                Actual = ridge.tegelik,
                                sample = "Test data"),
                     var, value, Actual:Predicted)


# for new test data:

test.data.new$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected), type = "response")
test.data.new$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected), type = "class")
test.data.new$ridge.prob.cut <- cut(test.data.new$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$ridge.prob))

ridge.groups <- unique(test.data.new$ridge.prob.cut)
ridge.oodatav <- c(NA,NA,NA,NA,NA)
ridge.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in ridge.groups) {
  tmp <- test.data.new %>% 
    filter(ridge.prob.cut == group)
  
  ridge.oodatav[i] = mean(tmp$ridge.prob)
  ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}


ridge.test.new <- gather(data.frame(ridge.groups = ridge.groups,
                                    Predicted = ridge.oodatav,
                                    Actual = ridge.tegelik,
                                    sample = "Short-term future test data"),
                         var, value, Actual:Predicted)


# for new longtest data:

test.data.new.long$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data.new.long, ruleFeat.x$mod$rules.selected), type = "response")
test.data.new.long$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data.new.long, ruleFeat.x$mod$rules.selected), type = "class")
test.data.new.long$ridge.prob.cut <- cut(test.data.new.long$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))


pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$ridge.prob))


ridge.groups <- unique(test.data.new.long$ridge.prob.cut)
ridge.oodatav <- c(NA,NA,NA,NA,NA)
ridge.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in ridge.groups) {
  tmp <- test.data.new.long %>% 
    filter(ridge.prob.cut == group)
  
  ridge.oodatav[i] = mean(tmp$ridge.prob)
  ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}


ridge.test.new.long <- gather(data.frame(ridge.groups = ridge.groups,
                                         Predicted = ridge.oodatav,
                                         Actual = ridge.tegelik,
                                         sample = "Mid-term future test data"),
                              var, value, Actual:Predicted)

ggplot(data = rbind(ridge.test, rbind(ridge.test.new, ridge.test.new.long))) +
  geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = ridge.groups, y = value, fill = var)) +
  geom_line(size = 1, stat='identity', aes(x = ridge.groups, y = value, color = var, group = var)) +
  facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
  scale_x_discrete(name = "Quintile distribution of risk scores predicted with GBM-based RuleFit") +
  scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
  theme_bw() +
  labs(fill='') +
  labs(color='') +
  theme(legend.position="bottom") +
  theme(text = element_text(family = 'serif', size = 14)) 






##########################################
############## EAML MODELS ###############
##########################################

rules_and_assessments <- merge(x = .rules_assessment,
                     y = assessments,
                     all.x = TRUE)

training.data.cxr <- matchCasesByRules(training.data, ruleFeat.x$mod$rules.selected)
test.data.cxr <- matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected)
test.data.new.cxr <- matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected)
test.data.new.long.cxr <- matchCasesByRules(test.data.new.long, ruleFeat.x$mod$rules.selected)

## EAML MODEL P <= 4
eaml_model_R4 <- s.GLMNET(training.data.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))],
                            training.data$y,
                            alpha = 0)
test.data$eaml_R4.prob <- predict(eaml_model_R4, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data$eaml_R4.estimate <- predict(eaml_model_R4, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data$eaml_R4.prob.cut <- cut_number(test.data$eaml_R4.prob, 5)

pROC::auc(test.data$y, as.numeric(test.data$eaml_R4.prob))

test.data.new$eaml_R4.prob <- predict(eaml_model_R4, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new$eaml_R4.estimate <- predict(eaml_model_R4, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new$eaml_R4.prob.cut <- cut(test.data.new$eaml_R4.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$eaml_R4.prob))

test.data.new.long$eaml_R4.prob <- predict(eaml_model_R4, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new.long$eaml_R4.estimate <- predict(eaml_model_R4, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new.long$eaml_R4.prob.cut <- cut(test.data.new.long$eaml_R4.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$eaml_R4.prob))

groups <- unique(test.data.new$eaml_R4.prob.cut)
oodatav <- c(NA,NA,NA,NA,NA)
tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in groups) {
  tmp <- test.data.new %>% 
    filter(eaml_R4.prob.cut == group)
  
  oodatav[i] = mean(tmp$eaml_R4.prob)
  tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

eaml_R4_test_new = gather(data.frame(groups = groups,
                                     Predicted = oodatav,
                                     Actual = tegelik,
                                     model = "R4"),
                          var, value, Actual:Predicted)


## EAML MODEL P <= 3
eaml_model_R3 <- s.GLMNET(training.data.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))],
                          training.data$y,
                          alpha = 0)
test.data$eaml_R3.prob <- predict(eaml_model_R3, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data$eaml_R3.estimate <- predict(eaml_model_R3, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data$eaml_R3.prob.cut <- cut_number(test.data$eaml_R3.prob, 5)

pROC::auc(test.data$y, as.numeric(test.data$eaml_R3.prob))

test.data.new$eaml_R3.prob <- predict(eaml_model_R3, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new$eaml_R3.estimate <- predict(eaml_model_R3, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new$eaml_R3.prob.cut <- cut(test.data.new$eaml_R3.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$eaml_R3.prob))

test.data.new.long$eaml_R3.prob <- predict(eaml_model_R3, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new.long$eaml_R3.estimate <- predict(eaml_model_R3, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new.long$eaml_R3.prob.cut <- cut_number(test.data.new.long$eaml_R3.prob, 5)

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$eaml_R3.prob))

groups <- unique(test.data.new$eaml_R3.prob.cut)
oodatav <- c(NA,NA,NA,NA,NA)
tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in groups) {
  tmp <- test.data.new %>% 
    filter(eaml_R3.prob.cut == group)
  
  oodatav[i] = mean(tmp$eaml_R3.prob)
  tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

eaml_R3_test_new = gather(data.frame(groups = groups,
                                     Predicted = oodatav,
                                     Actual = tegelik,
                                     model = "R3"),
                          var, value, Actual:Predicted)

## EAML MODEL P <= 2
eaml_model_R2 <- s.GLMNET(training.data.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))],
                          training.data$y,
                          alpha = 0)
test.data$eaml_R2.prob <- predict(eaml_model_R2, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data$eaml_R2.estimate <- predict(eaml_model_R2, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data$eaml_R2.prob.cut <- cut_number(test.data$eaml_R2.prob, 5)

pROC::auc(test.data$y, as.numeric(test.data$eaml_R2.prob))

test.data.new$eaml_R2.prob <- predict(eaml_model_R2, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new$eaml_R2.estimate <- predict(eaml_model_R2, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new$eaml_R2.prob.cut <- cut(test.data.new$eaml_R2.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$eaml_R2.prob))

test.data.new.long$eaml_R2.prob <- predict(eaml_model_R2, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new.long$eaml_R2.estimate <- predict(eaml_model_R2, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new.long$eaml_R2.prob.cut <- cut_number(test.data.new.long$eaml_R2.prob, 5)

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$eaml_R2.prob))

groups <- unique(test.data.new$eaml_R2.prob.cut)
oodatav <- c(NA,NA,NA,NA,NA)
tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in groups) {
  tmp <- test.data.new %>% 
    filter(eaml_R2.prob.cut == group)
  
  oodatav[i] = mean(tmp$eaml_R2.prob)
  tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

eaml_R2_test_new = gather(data.frame(groups = groups,
                                     Predicted = oodatav,
                                     Actual = tegelik,
                                     model = "R2"),
                          var, value, Actual:Predicted)

## EAML MODEL P <= 1
eaml_model_R1 <- s.GLMNET(training.data.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))],
                          training.data$y,
                          alpha = 0)
test.data$eaml_R1.prob <- predict(eaml_model_R1, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data$eaml_R1.estimate <- predict(eaml_model_R1, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data$eaml_R1.prob.cut <- cut(test.data$eaml_R1.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data$y, as.numeric(test.data$eaml_R1.prob))

test.data.new$eaml_R1.prob <- predict(eaml_model_R1, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new$eaml_R1.estimate <- predict(eaml_model_R1, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new$eaml_R1.prob.cut <- cut(test.data.new$eaml_R1.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$eaml_R1.prob))

test.data.new.long$eaml_R1.prob <- predict(eaml_model_R1, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "response")
test.data.new.long$eaml_R1.estimate <- predict(eaml_model_R1, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))], type = "class")
test.data.new.long$eaml_R1.prob.cut <- cut_number(test.data.new.long$eaml_R1.prob, 5)

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$eaml_R1.prob))

groups <- unique(test.data.new$eaml_R1.prob.cut)
oodatav <- c(NA,NA,NA,NA,NA)
tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in groups) {
  tmp <- test.data.new %>% 
    filter(eaml_R1.prob.cut == group)
  
  oodatav[i] = mean(tmp$eaml_R1.prob)
  tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

eaml_R1_test_new = gather(data.frame(groups = groups,
                                  Predicted = oodatav,
                                  Actual = tegelik,
                                  model = "R1"),
                          var, value, Actual:Predicted)


## EAML MODEL THAT REMOVES THE HIDDEN CONFOUNDER (50-60 YEAR OLDS THAT HAD NOT WORKED FOR A WHILE) 
eaml_model_confounder <- s.GLMNET(training.data.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))],
                          training.data$y,
                          alpha = 0)
test.data$eaml_conf.prob <- predict(eaml_model_confounder, newdata = test.data.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response")
test.data$eaml_conf.estimate <- predict(eaml_model_confounder, newdata = test.data.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "class")
test.data$eaml_conf.prob.cut <- cut(test.data$eaml_conf.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data$y, as.numeric(test.data$eaml_conf.prob))

test.data.new$eaml_conf.prob <- predict(eaml_model_confounder, newdata = test.data.new.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response")
test.data.new$eaml_conf.estimate <- predict(eaml_model_confounder, newdata = test.data.new.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "class")
test.data.new$eaml_conf.prob.cut <- cut(test.data.new$eaml_conf.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new$y, as.numeric(test.data.new$eaml_conf.prob))

test.data.new.long$eaml_conf.prob <- predict(eaml_model_confounder, newdata = test.data.new.long.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response")
test.data.new.long$eaml_conf.estimate <- predict(eaml_model_confounder, newdata = test.data.new.long.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "class")
test.data.new.long$eaml_conf.prob.cut <- cut(test.data.new.long$eaml_conf.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$eaml_conf.prob))



### pred for random forest base model
RF.groups <- unique(test.data.new$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data.new %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$pred.0)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF_test_new = gather(data.frame(groups = RF.groups,
                                Predicted = RF.oodatav,
                                Actual = RF.tegelik,
                                model = "RF"),
                     var, value, Actual:Predicted)

### pred for ridge model 
ridge.groups <- unique(test.data.new$ridge.prob.cut)
ridge.oodatav <- c(NA,NA,NA,NA,NA)
ridge.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in ridge.groups) {
  tmp <- test.data.new %>% 
    filter(ridge.prob.cut == group)
  
  ridge.oodatav[i] = mean(tmp$ridge.prob)
  ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}


ridge_test_new = gather(data.frame(groups = ridge.groups,
                                  Predicted = ridge.oodatav,
                                  Actual = ridge.tegelik,
                                  model = "RuleFit base"),
                        var, value, Actual:Predicted)


ggplot(data = rbind(ridge_test_new, rbind(RF_test_new, rbind(eaml_R1_test_new, eaml_R3_test_new)))) +
  geom_bar(stat="identity", position = "dodge", alpha = 1, aes(x = groups, y = value, fill = var)) +
  facet_wrap(~fct_relevel(model, c("R1", "R3"))) +
  scale_x_discrete(name = "Distribution of predicted risk scores by quintiles") +
  scale_y_continuous(name = "Risk of not resuming work in 180 days") +
  theme_bw() +
  labs(fill='') +
  theme(legend.position="bottom") +
  theme(text = element_text(family = 'serif', size = 10)) 





#####   PERFORMANCE EVALUATION ON DIFFERENT SIZE TRAINING SETS   #####

# Testing the accuracy for the three test sets for: 1000, 2000, 4000, 8000, 16000 training observations
#n_obs <- c(500, 1000, 2000, 4000, 8000, 16000)


n_obs <- c(250, 500, 1000, 2000, 4000, 8000, 16000)

mod <- c()
training_obs <- c()
auc_test <- c()
auc_test_new <- c()
auc_test_new_long <- c()
sample <- c()


for (i in n_obs) {
  for (s in 1:5) {
    sample_train <- stratified(training.data, group = c("y"), size = i/2)
    train.cxr <- matchCasesByRules(sample_train, ruleFeat.x$mod$rules.selected)
  
    mod_RF <-  s.RANGER(
      sample_train[,-c(1:4)], 
      y = sample_train$y,
      n.trees = 500,
      mtryStart = 10,
      stepFactor = 1,
      autotune = TRUE,
      probability = TRUE
    )
  
    mod_RuleFit <- s.GLMNET(train.cxr,
                     sample_train$y,
                     alpha = 0)
    
    mod_confounder <- s.GLMNET(train.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))],
                               sample_train$y,
                               alpha = 0) 
  
    mod_R1 <- s.GLMNET(train.cxr[, which(rules_and_assessments$penalty_rank <= 1 | is.na(rules_and_assessments$penalty_rank))],
                       sample_train$y,
                       alpha = 0)
    mod_R2 <- s.GLMNET(train.cxr[, which(rules_and_assessments$penalty_rank <= 2 | is.na(rules_and_assessments$penalty_rank))],
                       sample_train$y,
                       alpha = 0)
    mod_R3 <- s.GLMNET(train.cxr[, which(rules_and_assessments$penalty_rank <= 3 | is.na(rules_and_assessments$penalty_rank))],
                       sample_train$y,
                       alpha = 0)
    mod_R4 <- s.GLMNET(train.cxr[, which(rules_and_assessments$penalty_rank <= 4 | is.na(rules_and_assessments$penalty_rank))],
                       sample_train$y,
                       alpha = 0)
    for (j in 1:7) {
    
      if (j == 5) {
        mod <- c(mod, "RuleFit")
      
        tmp_auc_test <- pROC::auc(test.data$y, as.numeric(predict(mod_RuleFit, newdata = test.data.cxr, type = "response")))
        tmp_auc_test_new <- pROC::auc(test.data.new$y, as.numeric(predict(mod_RuleFit, newdata = test.data.new.cxr, type = "response")))
        tmp_auc_test_new_long <- pROC::auc(test.data.new.long$y, as.numeric(predict(mod_RuleFit, newdata = test.data.new.long.cxr, type = "response"))) 
      
      } else if (j == 6) { 
        mod <- c(mod, "Random Forest")
      
        tmp_auc_test <- pROC::auc(test.data$y, (data.frame(predict(mod_RF, newdata = test.data, type="response")) %>% 
                                                       mutate(RF.estimate = ifelse(X1 > X0, 1, 0)))$X0)
        tmp_auc_test_new <- pROC::auc(test.data.new$y, (data.frame(predict(mod_RF, newdata = test.data.new, type="response")) %>% 
                                                       mutate(RF.estimate = ifelse(X1 > X0, 1, 0)))$X0)
        tmp_auc_test_new_long <- pROC::auc(test.data.new.long$y, (data.frame(predict(mod_RF, newdata = test.data.new.long, type="response")) %>% 
                                                       mutate(RF.estimate = ifelse(X1 > X0, 1, 0)))$X0)
      
      } else if (j == 7) {
        mod <- c(mod, "Discarded R116, R15 & R25")
        
        tmp_auc_test <- pROC::auc(test.data$y, as.numeric(predict(mod_confounder, newdata = test.data.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response")))
        tmp_auc_test_new <- pROC::auc(test.data.new$y, as.numeric(predict(mod_confounder, newdata = test.data.new.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response")))
        tmp_auc_test_new_long <- pROC::auc(test.data.new.long$y, as.numeric(predict(mod_confounder, newdata = test.data.new.long.cxr[, which(!(rules_and_assessments$Rule_ID %in% c("R116", "R15", "R25")))], type = "response"))) 
      } else {
        tmp_mod <- eval(parse(text = paste0("mod_R", toString(j))))
        mod <- c(mod, paste0("EAML", toString(j)))
      
        tmp_auc_test <- pROC::auc(test.data$y, as.numeric(predict(tmp_mod, newdata = test.data.cxr[, which(rules_and_assessments$penalty_rank <= j | is.na(rules_and_assessments$penalty_rank))], type = "response")))
        tmp_auc_test_new <- pROC::auc(test.data.new$y, as.numeric(predict(tmp_mod, newdata = test.data.new.cxr[, which(rules_and_assessments$penalty_rank <= j | is.na(rules_and_assessments$penalty_rank))], type = "response")))
        tmp_auc_test_new_long <- pROC::auc(test.data.new.long$y, as.numeric(predict(tmp_mod, newdata = test.data.new.long.cxr[, which(rules_and_assessments$penalty_rank <= j | is.na(rules_and_assessments$penalty_rank))], type = "response"))) 
      }
      
      training_obs <- c(training_obs, i)
      auc_test <- c(auc_test, tmp_auc_test)
      auc_test_new <- c(auc_test_new, tmp_auc_test_new)
      auc_test_new_long <- c(auc_test_new_long, tmp_auc_test_new_long)
      sample <- c(sample, s)
    }
    print(paste0("Processed sample number ",  toString(s), " for training size ", toString(i)))
  }
}

auc_train_size <- data.frame(model = mod,
                             n_train = as.factor(training_obs),
                             auc_test = auc_test,
                             auc_test_new = auc_test_new,
                             auc_test_new_long = auc_test_new_long,
                             sample = sample) %>% 
  group_by(model, n_train) %>% 
  dplyr::mutate(test1.mean_auc = mean(auc_test),
                   .sd = sd(auc_test),
                   .n = n()) %>% 
  mutate(.se = .sd / sqrt(.n),
         test1.ci_lower = test1.mean_auc - qt(1 - (0.05 / 2), n() - 1) * .se,
         test1.ci_upper = test1.mean_auc + qt(1 - (0.05 / 2), n() - 1) * .se) %>% 
  dplyr::mutate(test2.mean_auc = mean(auc_test_new),
                   .sd = sd(auc_test_new),
                   .n = n()) %>%
  mutate(.se = .sd / sqrt(.n),
         test2.ci_lower = test2.mean_auc - qt(1 - (0.05 / 2), n() - 1) * .se,
         test2.ci_upper = test2.mean_auc + qt(1 - (0.05 / 2), n() - 1) * .se) %>% 
  dplyr::mutate(test3.mean_auc = mean(auc_test_new_long),
                   .sd = sd(auc_test_new_long),
                   .n = n()) %>%
  mutate(.se = .sd / sqrt(.n),
         test3.ci_lower = test3.mean_auc - qt(1 - (0.05 / 2), n() - 1) * .se,
         test3.ci_upper = test3.mean_auc + qt(1 - (0.05 / 2), n() - 1) * .se) %>% 
  select(-c(auc_test, auc_test_new, auc_test_new_long, sample, .sd, .se, .n, )) %>% 
  group_by_all() %>% 
  summarize()


ggplot(data = reshape(auc_train_size, direction='long', 
                      varying=c('test1.mean_auc', 'test1.ci_lower','test1.ci_upper',
                                'test2.mean_auc', 'test2.ci_lower', 'test2.ci_upper',
                                'test3.mean_auc', 'test3.ci_lower', 'test3.ci_upper'), 
                      timevar='test_set',
                      times=c('Test data', 'Short-term future', 'Mid-term future'),
                      v.names=c('measure1','measure2','measure3'),
                      idvar=c('model', 'n_train')) %>% 
         rename("mean_auc" = "measure1", "ci_lower" = "measure2", "ci_upper" = "measure3") %>% 
         filter(model != "Random Forest") %>% 
         filter(test_set == "Test data") %>% 
         filter(!(n_train %in% c(250))),
       aes(x = n_train, y = mean_auc, color = model, group = model)) +
  geom_line(stat="identity", position = "dodge", size = 1) +
  scale_x_discrete(name = "N training observations", labels = c("500", "1000", "2000", "4000", "8000", "16000")) + # labels = c("500", "1K", "2K", "4K", "8K")
  scale_y_continuous(name = "Mean AUC") +
  scale_color_manual(labels = c("Discarded R116, R15 & R25", "R ≤ 1", "R ≤ 2", "R ≤ 3", "R ≤ 4", "Base RuleFit"),
                     values = c("#F8766D", "#BB9D00", "#00b81f", "#00c0b8", "#e76bf4", "#000000")) +
  geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), position = position_dodge(width = 0), width = 1) +
  facet_wrap(~fct_relevel(test_set, c('Test data', 'Short-term future', 'Mid-term future'))) +
  theme_bw() +
  #theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
  theme(text = element_text(family = 'serif', size = 14)) +
  labs(color='Model') +
  guides(color="none") +
  guides(colour = guide_legend(nrow = 1)) +
  theme(legend.position="bottom")

training_obs <- c(training_obs, i)
auc_test <- c(auc_test, mean(samples_auc_test))
auc_test_new <- c(auc_test_new, tmp_auc_test_new)
auc_test_new_long <- c(auc_test_new_long, tmp_auc_test_new_long)
ci.lower_test <- c(ci.lower_test, mean(samples_auc_test) - qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test) / sqrt(3))
ci.lower_test_new <- c(ci.lower_test_new, mean(samples_auc_test_new) - qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test_new) / sqrt(3))
ci.lower_test_new_long <- c(ci.lower_test_new_long, mean(samples_auc_test_new_long) - qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test_new_long) / sqrt(3))
ci.upper_test <- c(ci.upper_test, mean(samples_auc_test) + qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test) / sqrt(3))
ci.upper_test_new <- c(ci.upper_test_new, mean(samples_auc_test_new) + qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test_new) / sqrt(3))
ci.upper_test_new_long <- c(ci.upper_test_new_long, mean(samples_auc_test_new_long) + qt(1 - (0.05 / 2), 7 - 1) * sd(samples_auc_test_new_long) / sqrt(3))





## PREDICT FOR RF:
load("RF.x")
load("test.data")
load("test.data.new")
load("test.data.new.long")

  # for test data:

test.data <- cbind(test.data ,
                    data.frame(predict(RF.x, newdata = test.data, type="response")) %>% 
                    rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
test.data$RF.prob.cut <- cut(test.data$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
test.data <- test.data %>% 
  mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

pROC::auc(test.data$y, test.data$RF.prob)
  #test.data$RF.estimate.cut <- cut(test.data$RF.estimate, breaks=c(0,0.25,0.5,0.75, 1))

RF.groups <- unique(test.data$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$RF.prob)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF.test = gather(data.frame(RF.groups = RF.groups,
                                Predicted = RF.oodatav,
                                Actual = RF.tegelik,
                                sample = "Test data"),
                     var, value, Actual:Predicted)


# for new test data:
test.data.new <- cbind(test.data.new ,
                            data.frame(predict(RF.x, newdata = test.data.new, type="response")) %>% 
                         rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
test.data.new$RF.prob.cut <- cut(test.data.new$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
test.data.new <- test.data.new %>% 
  mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

pROC::auc(test.data.new$y, test.data.new$RF.prob)

RF.groups <- unique(test.data.new$RF.prob.cut)
RF.oodatav <- c(NA,NA,NA,NA,NA)
RF.tegelik <- c(NA,NA,NA,NA,NA)

i = 1
for (group in RF.groups) {
  tmp <- test.data.new %>% 
    filter(RF.prob.cut == group)
  
  RF.oodatav[i] = mean(tmp$RF.prob)
  RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
  i = i + 1
}

RF.test.new = gather(data.frame(RF.groups = RF.groups,
                            Predicted = RF.oodatav,
                            Actual = RF.tegelik,
                            sample = "Short-term future test data"),
                 var, value, Actual:Predicted)


# for new long test data:
  test.data.new.long <- cbind(test.data.new.long ,
                        data.frame(predict(RF.x, newdata = test.data.new.long, type="response")) %>% 
                          rename("RF.prob.rev" = "X1", "RF.prob" = "X0"))
  test.data.new.long$RF.prob.cut <- cut(test.data.new.long$RF.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
  test.data.new.long <- test.data.new.long %>% 
    mutate(RF.estimate = ifelse(RF.prob.rev > RF.prob, 1, 0))

  pROC::auc(test.data.new.long$y, test.data.new.long$RF.prob)
  
  RF.groups <- unique(test.data.new.long$RF.prob.cut)
  RF.oodatav <- c(NA,NA,NA,NA,NA)
  RF.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in RF.groups) {
    tmp <- test.data.new.long %>% 
      filter(RF.prob.cut == group)
    
    RF.oodatav[i] = mean(tmp$RF.prob)
    RF.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
    
    i = i + 1
  }
  
  RF.test.new.long = gather(data.frame(RF.groups = RF.groups,
                                  Predicted = RF.oodatav,
                                  Actual = RF.tegelik,
                                  sample = "Mid-term future test data"),
                       var, value, Actual:Predicted)



  ggplot(data = rbind(RF.test, rbind(RF.test.new, RF.test.new.long))) +
    geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = RF.groups, y = value, fill = var)) +
    geom_line(size = 1, stat='identity', aes(x = RF.groups, y = value, color = var, group = var)) +
    facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
    scale_x_discrete(name = "Quintile distribution of risk scores predicted with Random Forest") +
    scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
    theme_bw() +
    labs(fill='') +
    guides(fill="none") +
    guides(color="none") +
    theme(legend.position="bottom") +
    theme(text = element_text(family = 'serif', size = 14)) 
    


  
  
  
  
  
  
  ## PREDICT FOR GBM BASE MODEL:
  load("GBM.x")
  
  # for test data:
  
  test.data$GBM.prob = predict(GBM.x, newdata = test.data, type="response")
  test.data$GBM.prob.cut <- cut(test.data$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
  
  pROC::auc(test.data$y, test.data$GBM.prob)
  #test.data$RF.estimate.cut <- cut(test.data$RF.estimate, breaks=c(0,0.25,0.5,0.75, 1))
  
  GBM.groups <- unique(test.data$GBM.prob.cut)
  GBM.oodatav <- c(NA,NA,NA,NA,NA)
  GBM.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in GBM.groups) {
    tmp <- test.data %>% 
      filter(GBM.prob.cut == group)
    
    GBM.oodatav[i] = mean(tmp$GBM.prob)
    GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
    
    i = i + 1
  }
  
  GBM.test = gather(data.frame(GBM.groups = GBM.groups,
                              Predicted = GBM.oodatav,
                              Actual = GBM.tegelik,
                              sample = "Test data"),
                   var, value, Actual:Predicted)
  
  
  # for new test data:
  test.data.new$GBM.prob = predict(GBM.x, newdata = test.data.new, type="response")
  test.data.new$GBM.prob.cut <- cut(test.data.new$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
  
  pROC::auc(test.data.new$y, test.data.new$GBM.prob)
  
  GBM.groups <- unique(test.data.new$GBM.prob.cut)
  GBM.oodatav <- c(NA,NA,NA,NA,NA)
  GBM.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in GBM.groups) {
    tmp <- test.data.new %>% 
      filter(GBM.prob.cut == group)
    
    GBM.oodatav[i] = mean(tmp$GBM.prob)
    GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
    
    i = i + 1
  }
  
  GBM.test.new = gather(data.frame(GBM.groups = GBM.groups,
                                  Predicted = GBM.oodatav,
                                  Actual = GBM.tegelik,
                                  sample = "Short-term future test data"),
                       var, value, Actual:Predicted)
  
  
  # for new long test data:
  test.data.new.long$GBM.prob = predict(GBM.x, newdata = test.data.new.long, type="response")
  test.data.new.long$GBM.prob.cut <- cut(test.data.new.long$GBM.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))
  
  pROC::auc(test.data.new.long$y, test.data.new.long$GBM.prob)
  
  GBM.groups <- unique(test.data.new.long$GBM.prob.cut)
  GBM.oodatav <- c(NA,NA,NA,NA,NA)
  GBM.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in GBM.groups) {
    tmp <- test.data.new.long %>% 
      filter(GBM.prob.cut == group)
    
    GBM.oodatav[i] = mean(tmp$GBM.prob)
    GBM.tegelik[i] = nrow(tmp %>% filter(y == 1)) / nrow(tmp)
    
    i = i + 1
  }
  
  GBM.test.new.long = gather(data.frame(GBM.groups = GBM.groups,
                                       Predicted = GBM.oodatav,
                                       Actual = GBM.tegelik,
                                       sample = "Mid-term future test data"),
                            var, value, Actual:Predicted)
  
  
  
  ggplot(data = rbind(GBM.test, rbind(GBM.test.new, GBM.test.new.long))) +
    geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = GBM.groups, y = value, fill = var)) +
    geom_line(size = 1, stat='identity', aes(x = GBM.groups, y = value, color = var, group = var)) +
    facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
    scale_x_discrete(name = "Quintile distribution of risk scores predicted with Gradient Boosting") +
    scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
    theme_bw() +
    labs(fill='') +
    guides(color="none") +
    guides(fill="none") +
    theme(legend.position="bottom") +
    theme(text = element_text(family = 'serif', size = 14)) 
  
  
  
  
  
  
## PREDICT FOR RuleFit, KÕIK 3199 REEGLIT:

# for test data:
  test.data <- cbind(test.data,
                         data.frame(predict(ruleFeat.x, newdata = test.data, type="prob")) %>% 
                           rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
  test.data$ruleFeat.prob.cut <- cut_number(test.data$ruleFeat.prob, 5)
  
  pROC::auc(test.data$y, as.numeric(test.data$ruleFeat.prob))


# for new test data:
test.data.new <- cbind(test.data.new,
                       data.frame(predict(ruleFeat.x, newdata = test.data.new, type="prob")) %>% 
                         rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
test.data.new$ruleFeat.prob.cut <- cut_number(test.data.new$ruleFeat.prob, 5)

pROC::auc(test.data.new$y, as.numeric(test.data.new$ruleFeat.prob))


# for new long test data:
test.data.new.long <- cbind(test.data.new.long,
                       data.frame(predict(ruleFeat.x, newdata = test.data.new.long, type="prob")) %>% 
                         rename("ruleFeat.prob" = "prob", "ruleFeat.estimate" = "estimate"))
test.data.new.long$ruleFeat.prob.cut <- cut_number(test.data.new.long$ruleFeat.prob, 5)

pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$ruleFeat.prob))




## PREDICT FOR Ridge, 142 REEGLIT:
load("ridge.x")

  # for test data

  test.data$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected), type = "response")
  test.data$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data, ruleFeat.x$mod$rules.selected), type = "class")
  test.data$ridge.prob.cut <- cut(test.data$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))


  pROC::auc(test.data$y, as.numeric(test.data$ridge.prob))
  
  ridge.groups <- unique(test.data$ridge.prob.cut)
  ridge.oodatav <- c(NA,NA,NA,NA,NA)
  ridge.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in ridge.groups) {
    tmp <- test.data %>% 
      filter(ridge.prob.cut == group)
    
    ridge.oodatav[i] = mean(tmp$ridge.prob)
    ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
    
    i = i + 1
  }
  
  
  ridge.test <- gather(data.frame(ridge.groups = ridge.groups,
                                           Predicted = ridge.oodatav,
                                           Actual = ridge.tegelik,
                                           sample = "Test data"),
                                var, value, Actual:Predicted)
  

  # for new test data:

  test.data.new$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected), type = "response")
  test.data.new$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data.new, ruleFeat.x$mod$rules.selected), type = "class")
  test.data.new$ridge.prob.cut <- cut(test.data.new$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

  pROC::auc(test.data.new$y, as.numeric(test.data.new$ridge.prob))
  
  ridge.groups <- unique(test.data.new$ridge.prob.cut)
  ridge.oodatav <- c(NA,NA,NA,NA,NA)
  ridge.tegelik <- c(NA,NA,NA,NA,NA)
  
  i = 1
  for (group in ridge.groups) {
    tmp <- test.data.new %>% 
      filter(ridge.prob.cut == group)
    
    ridge.oodatav[i] = mean(tmp$ridge.prob)
    ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
    
    i = i + 1
  }
  
  
  ridge.test.new <- gather(data.frame(ridge.groups = ridge.groups,
                                           Predicted = ridge.oodatav,
                                           Actual = ridge.tegelik,
                                           sample = "Short-term future test data"),
                                var, value, Actual:Predicted)
  
  
  # for new longtest data:
  
  test.data.new.long$ridge.prob <- predict(ridge.x, newdata = matchCasesByRules(test.data.new.long, ruleFeat.x$mod$rules.selected), type = "response")
  test.data.new.long$ridge.estimate <- predict(ridge.x, newdata = matchCasesByRules(test.data.new.long, ruleFeat.x$mod$rules.selected), type = "class")
  test.data.new.long$ridge.prob.cut <- cut(test.data.new.long$ridge.prob, breaks = c(0,0.2,0.4,0.6,0.8,1), labels = c("20%", "40%", "60%", "80%", "100%"))

  
  pROC::auc(test.data.new.long$y, as.numeric(test.data.new.long$ridge.prob))
  

  ridge.groups <- unique(test.data.new.long$ridge.prob.cut)
  ridge.oodatav <- c(NA,NA,NA,NA,NA)
  ridge.tegelik <- c(NA,NA,NA,NA,NA)

  i = 1
  for (group in ridge.groups) {
    tmp <- test.data.new.long %>% 
      filter(ridge.prob.cut == group)
  
    ridge.oodatav[i] = mean(tmp$ridge.prob)
    ridge.tegelik[i] = nrow(tmp %>% filter(y == 0)) / nrow(tmp)
  
    i = i + 1
  }


  ridge.test.new.long <- gather(data.frame(ridge.groups = ridge.groups,
                                           Predicted = ridge.oodatav,
                                           Actual = ridge.tegelik,
                                           sample = "Mid-term future test data"),
                                var, value, Actual:Predicted)

  ggplot(data = rbind(ridge.test, rbind(ridge.test.new, ridge.test.new.long))) +
    geom_bar(stat="identity", position = "dodge", alpha = 0.5, aes(x = ridge.groups, y = value, fill = var)) +
    geom_line(size = 1, stat='identity', aes(x = ridge.groups, y = value, color = var, group = var)) +
    facet_wrap(~fct_relevel(sample, c("Test data", "Short-term future test data", "Mid-term future test data"))) +
    scale_x_discrete(name = "Quintile distribution of risk scores predicted with GBM-based RuleFit") +
    scale_y_continuous(name = "Risk of not resuming\n work in 180 days") +
    theme_bw() +
    labs(fill='') +
    labs(color='') +
    theme(legend.position="bottom") +
    theme(text = element_text(family = 'serif', size = 14)) 

  


#####   COMPARING PREDICTIONS ACROSS DIFFERENT SUBPOPULATIONS (RULES)   #####
  
  .n_test <- c()
  .n_test_new <- c()
  .n_test_new_long <- c()
  
  .rule_probs_test_actual <- c()
  .rule_probs_test_conf <- c()
  .rule_probs_test_ridge <- c()
  .rule_probs_test_rf <- c()
  
  .rule_probs_test_new_actual <- c()
  .rule_probs_test_new_conf <- c()
  .rule_probs_test_new_ridge <- c()
  .rule_probs_test_new_rf <- c()
  
  .rule_probs_test_new_long_actual <- c()
  .rule_probs_test_new_long_conf <- c()
  .rule_probs_test_new_long_ridge <- c()
  .rule_probs_test_new_long_rf <- c()
  
  .rule_test_mse_conf <- c()
  .rule_test_new_mse_conf <- c()
  .rule_test_new_long_mse_conf <- c()
  
  .rule_test_mse_ridge <- c()
  .rule_test_new_mse_ridge <- c()
  .rule_test_new_long_mse_ridge <- c()
  
  .rule_test_mse_rf <- c()
  .rule_test_new_mse_rf <- c()
  .rule_test_new_long_mse_rf <- c()
  
    
  for (rule_id in 1:length(ruleFeat.x[["mod"]][["rules.selected"]])) {
    #print(rule_id)
    
    .rule_sub_test <- test.data %>% 
      filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][rule_id])))
    .rule_sub_test_new <- test.data.new %>% 
      filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][rule_id])))
    .rule_sub_test_new_long <- test.data.new.long %>% 
      filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][rule_id])))
    
    .n_test <- c(.n_test, nrow(.rule_sub_test))
    .rule_probs_test_actual <- c(.rule_probs_test_actual, prop.table(table(.rule_sub_test$y))[[1]])
    .rule_probs_test_conf <- c(.rule_probs_test_conf, mean(.rule_sub_test$eaml_conf.prob))
    .rule_probs_test_ridge <- c(.rule_probs_test_ridge, mean(.rule_sub_test$ridge.prob))
    .rule_probs_test_rf <- c(.rule_probs_test_rf, mean(.rule_sub_test$RF.prob))
    
    .n_test_new <- c(.n_test_new, nrow(.rule_sub_test_new))
    .rule_probs_test_new_actual <- c(.rule_probs_test_new_actual, prop.table(table(.rule_sub_test_new$y))[[1]])
    .rule_probs_test_new_conf <- c(.rule_probs_test_new_conf, mean(.rule_sub_test_new$eaml_conf.prob))
    .rule_probs_test_new_ridge <- c(.rule_probs_test_new_ridge, mean(.rule_sub_test_new$ridge.prob))
    .rule_probs_test_new_rf <- c(.rule_probs_test_new_rf, mean(.rule_sub_test_new$RF.prob))
    
    .n_test_new_long <- c(.n_test_new_long, nrow(.rule_sub_test_new_long))
    .rule_probs_test_new_long_actual <- c(.rule_probs_test_new_long_actual, prop.table(table(.rule_sub_test_new_long$y))[[1]])
    .rule_probs_test_new_long_conf <- c(.rule_probs_test_new_long_conf, mean(.rule_sub_test_new_long$eaml_conf.prob))
    .rule_probs_test_new_long_ridge <- c(.rule_probs_test_new_long_ridge, mean(.rule_sub_test_new_long$ridge.prob))
    .rule_probs_test_new_long_rf <- c(.rule_probs_test_new_long_rf, mean(.rule_sub_test_new_long$RF.prob))
    
    # subpopulation MSE

    .rule_test_mse_conf <- c(.rule_test_mse_conf, pROC::auc(.rule_sub_test$y, as.numeric(.rule_sub_test$eaml_conf.prob)))
    .rule_test_new_mse_conf <- c(.rule_test_new_mse_conf, pROC::auc(.rule_sub_test_new$y, as.numeric(.rule_sub_test_new$eaml_conf.prob)))
    .rule_test_new_long_mse_conf <- c(.rule_test_new_long_mse_conf, pROC::auc(.rule_sub_test_new_long$y, as.numeric(.rule_sub_test_new_long$eaml_conf.prob)))
    
    .rule_test_mse_ridge <- c(.rule_test_mse_ridge, pROC::auc(.rule_sub_test$y, as.numeric(.rule_sub_test$ridge.prob)))
    .rule_test_new_mse_ridge <- c(.rule_test_new_mse_ridge,pROC::auc(.rule_sub_test_new$y, as.numeric(.rule_sub_test_new$ridge.prob)))
    .rule_test_new_long_mse_ridge <- c(.rule_test_new_long_mse_ridge, pROC::auc(.rule_sub_test_new_long$y, as.numeric(.rule_sub_test_new_long$ridge.prob)))
    
    .rule_test_mse_rf <- c(.rule_test_mse_rf, pROC::auc(.rule_sub_test$y, as.numeric(.rule_sub_test$RF.prob)))
    .rule_test_new_mse_rf <- c(.rule_test_new_mse_rf,pROC::auc(.rule_sub_test_new$y, as.numeric(.rule_sub_test_new$RF.prob)))
    .rule_test_new_long_mse_rf <- c(.rule_test_new_long_mse_rf, pROC::auc(.rule_sub_test_new_long$y, as.numeric(.rule_sub_test_new_long$RF.prob)))
  }
  
  rule_probs <- data.frame(
    rule = .rule <- ruleFeat.x[["mod"]][["rules.selected"]],
    rule_id = sprintf("R%s",seq(1:142)),
    
    training_empirical_risk = ruleFeat.x$mod$rules.selected.coef.er$Empirical_Risk,
    
    test_data = .rule_probs_test_actual,
    test_data_n = .n_test,
    test_data_pred_conf = .rule_probs_test_conf,
    test_data_pred_ridge = .rule_probs_test_ridge,
    test_data_pred_rf = .rule_probs_test_rf,
    
    test_data_new = .rule_probs_test_new_actual,
    test_data_new_n = .n_test_new,
    test_data_new_pred_conf = .rule_probs_test_new_conf,
    test_data_new_pred_ridge = .rule_probs_test_new_ridge,
    test_data_new_pred_rf = .rule_probs_test_new_rf,
    
    test_data_new_long = .rule_probs_test_new_long_actual,
    test_data_new_long_n = .n_test_new_long,
    test_data_new_long_pred_conf = .rule_probs_test_new_long_conf,
    test_data_new_long_pred_ridge = .rule_probs_test_new_long_ridge,
    test_data_new_long_pred_rf = .rule_probs_test_new_long_rf,
    
    test_data_mse_conf = .rule_test_mse_conf,
    test_data_new_mse_conf = .rule_test_new_mse_conf,
    test_data_new_long_mse_conf = .rule_test_new_long_mse_conf,
    
    test_data_mse_ridge = .rule_test_mse_ridge,
    test_data_new_mse_ridge = .rule_test_new_mse_ridge,
    test_data_new_long_mse_ridge = .rule_test_new_long_mse_ridge,
    
    test_data_mse_rf = .rule_test_mse_rf,
    test_data_new_mse_rf = .rule_test_new_mse_rf,
    test_data_new_long_mse_rf = .rule_test_new_long_mse_rf
  ) %>% 
    mutate(diff_conf_ridge = test_data_new_pred_conf - test_data_new_pred_ridge,
           diff_conf_ridge_abs = abs(diff_conf_ridge),
           test_did = (test_data-test_data_pred_ridge) - (test_data - test_data_pred_conf),
           test_new_did = (test_data_new - test_data_new_pred_ridge) - (test_data_new - test_data_new_pred_conf),
           exp_aug_improvement = test_did - test_new_did,
           abs_exp_aug_improvement = abs(exp_aug_improvement),
           test_data_prop_obs = test_data_n*100/nrow(test.data),
           test_data_new_prop_obs = test_data_new_n*100/nrow(test.data.new),
           test_data_new_long_prop_obs = test_data_new_long_n*100/nrow(test.data.new.long)
           )
  
  
  #### LOOKING AT THE 5 RULES OUTSIDE THE 95% CI ####

  # RULE 116
  .rule_116_training <- training.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][116])))
  .rule_116_test <- test.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][116])))
  .rule_116_test.new <- test.data.new %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][116])))
  .rule_116_test.new.long <- test.data.new.long %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][116])))
  
  # RULE 15
  .rule_15_training <- training.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][15])))
  .rule_15_test <- test.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][15])))
  .rule_15_test.new <- test.data.new %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][15])))
  .rule_15_test.new.long <- test.data.new.long %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][15])))
  
  # RULE 25
  .rule_25_training <- training.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][25])))
  .rule_25_test <- test.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][25])))
  .rule_25_test.new <- test.data.new %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][25])))
  .rule_25_test.new.long <- test.data.new.long %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][25])))
  
  # RULE 61
  .rule_61_training <- training.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][61])))
  .rule_61_test <- test.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][61])))
  .rule_61_test.new <- test.data.new %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][61])))
  .rule_61_test.new.long <- test.data.new.long %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][61])))
  
  # RULE 69
  .rule_69_training <- training.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][69])))
  .rule_69_test <- test.data %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][69])))
  .rule_69_test.new <- test.data.new %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][69])))
  .rule_69_test.new.long <- test.data.new.long %>% 
    filter(eval(parse(text = ruleFeat.x[["mod"]][["rules.selected"]][69])))
  
  rule_probs <- data.frame(
    rule = c("R116", "R15", "R25", "R61", "R69", "All\njob-seekers"),
    #training_data = c(prop.table(table(rule_116_training$y))[[1]],
    #                  prop.table(table(rule_15_training$y))[[1]],
    #                  prop.table(table(rule_25_training$y))[[1]],
    #                  prop.table(table(rule_61_training$y))[[1]],
    #                  prop.table(table(rule_69_training$y))[[1]],
    #                  prop.table(table(training.data$y))[[1]]),
    test_data = c(prop.table(table(.rule_116_test$y))[[1]],
                  prop.table(table(.rule_15_test$y))[[1]],
                  prop.table(table(.rule_25_test$y))[[1]],
                  prop.table(table(.rule_61_test$y))[[1]],
                  prop.table(table(.rule_69_test$y))[[1]],
                  prop.table(table(test.data$y))[[1]]),
    test_data_pred_conf = c(mean(.rule_116_test$eaml_conf.prob),
                        mean(.rule_15_test$eaml_conf.prob),
                        mean(.rule_25_test$eaml_conf.prob),
                        mean(.rule_61_test$eaml_conf.prob),
                        mean(.rule_69_test$eaml_conf.prob),
                        mean(test.data$eaml_conf.prob)),
    test_data_pred_ridge = c(mean(.rule_116_test$ridge.prob),
                          mean(.rule_15_test$ridge.prob),
                          mean(.rule_25_test$ridge.prob),
                          mean(.rule_61_test$ridge.prob),
                          mean(.rule_69_test$ridge.prob),
                          mean(test.data$ridge.prob)),
    test_data_new = c(prop.table(table(.rule_116_test.new$y))[[1]],
                      prop.table(table(.rule_15_test.new$y))[[1]],
                      prop.table(table(.rule_25_test.new$y))[[1]],
                      prop.table(table(.rule_61_test.new$y))[[1]],
                      prop.table(table(.rule_69_test.new$y))[[1]],
                      prop.table(table(test.data.new$y))[[1]]),
    test_data_new_pred_conf = c(mean(.rule_116_test.new$eaml_conf.prob),
                           mean(.rule_15_test.new$eaml_conf.prob),
                           mean(.rule_25_test.new$eaml_conf.prob),
                           mean(.rule_61_test.new$eaml_conf.prob),
                           mean(.rule_69_test.new$eaml_conf.prob),
                           mean(test.data.new$eaml_conf.prob)),
    test_data_new_pred_ridge = c(mean(.rule_116_test.new$ridge.prob),
                             mean(.rule_15_test.new$ridge.prob),
                             mean(.rule_25_test.new$ridge.prob),
                             mean(.rule_61_test.new$ridge.prob),
                             mean(.rule_69_test.new$ridge.prob),
                             mean(test.data.new$ridge.prob)),
    test_data_new_long = c(prop.table(table(.rule_116_test.new.long$y))[[1]],
                           prop.table(table(.rule_15_test.new.long$y))[[1]],
                           prop.table(table(.rule_25_test.new.long$y))[[1]],
                           prop.table(table(.rule_61_test.new.long$y))[[1]],
                           prop.table(table(.rule_69_test.new.long$y))[[1]],
                           prop.table(table(test.data.new.long$y))[[1]]),
    test_data_new_long_pred_conf = c(mean(.rule_116_test.new.long$eaml_conf.prob),
                                mean(.rule_15_test.new.long$eaml_conf.prob),
                                mean(.rule_25_test.new.long$eaml_conf.prob),
                                mean(.rule_61_test.new.long$eaml_conf.prob),
                                mean(.rule_69_test.new.long$eaml_conf.prob),
                                mean(test.data.new.long$eaml_conf.prob)),
    test_data_new_long_pred_ridge = c(mean(.rule_116_test.new.long$ridge.prob),
                                 mean(.rule_15_test.new.long$ridge.prob),
                                 mean(.rule_25_test.new.long$ridge.prob),
                                 mean(.rule_61_test.new.long$ridge.prob),
                                 mean(.rule_69_test.new.long$ridge.prob),
                                 mean(test.data.new.long$ridge.prob))
  )
  
  
  ggplot(data = gather(reshape(rule_probs, direction='long', 
                               varying=c('test_data', 'test_data_pred_conf', 'test_data_pred_ridge',
                                        'test_data_new', 'test_data_new_pred_conf', 'test_data_new_pred_ridge',
                                        'test_data_new_long', 'test_data_new_long_pred_conf', 'test_data_new_long_pred_ridge'), 
                               timevar='test_set',
                               v.names=c('col1','col2', 'col3'),
                               idvar=c('rule'))
                       , risk_score, measure, col1:col3) %>% 
           mutate(test_set_verb = ifelse(test_set == "1", "Test data", 
                                         ifelse(test_set == "2", "Short-term future", 
                                                ifelse(test_set == "3", "Mid-term future", "error"))))) +
    geom_point(stat="identity", position = "dodge", size = 5, stroke = 1, aes(x = rule,  color = rule, y = measure, shape = risk_score)) +
    #labs(fill="Data set") +
    facet_wrap(~fct_relevel(test_set_verb, c("Test data", "Short-term future", "Mid-term future"))) +
    scale_shape_manual(labels = c("Actual", "Predicted: best expert-\naugmented model", "Predicted: base RuleFit"), values=c(19, 4, 1)) +
    scale_x_discrete(name = "Decision rule defining a group of job-seekers") +
    scale_y_continuous(name = "Mean risk of not resuming work in 180 days") +
    theme_bw() +
    labs(fill='') +
    theme(legend.title = element_blank()) +
    theme(text = element_text(family = 'serif', size = 14))
    

  
  # emp + predicted riskid top 6 reeglile, mille puhul eaml pred erines ridge predist:

  ggplot(data = gather(reshape(rule_probs %>% 
                                 filter(rule_id %in% c("R50", "R61", "R30", "R53", "R121", "R18", "R119", "R134", "R6", "R20")), direction='long', 
                               varying=c('test_data', 'test_data_pred_conf', 'test_data_pred_ridge',
                                         'test_data_new', 'test_data_new_pred_conf', 'test_data_new_pred_ridge',
                                         'test_data_new_long', 'test_data_new_long_pred_conf', 'test_data_new_long_pred_ridge'
                                         ), 
                               timevar='test_set',
                               v.names=c('col1','col2', 'col3'),
                               idvar=c('rule_id'))
                       , risk_score, measure, col1:col3) %>% 
           mutate(test_set_verb = ifelse(test_set == "1", "Test data", 
                                         ifelse(test_set == "2", "Short-term future", 
                                                ifelse(test_set == "3", "Mid-term future", "error")))) %>% 
           filter(test_set_verb != "Mid-term future")) +
      geom_point(stat="identity", position = "dodge", size = 5, stroke = 1, aes(x = fct_relevel(rule_id, c("R50", "R61", "R30", "R53", "R121", "R18", "R119", "R134", "R6", "R20")),  color = fct_relevel(rule_id, c("R50", "R61", "R30", "R53", "R121", "R18", "R119", "R134", "R6", "R20")), y = measure, shape = risk_score)) +
    #labs(fill="Data set") +
    facet_wrap(~fct_relevel(test_set_verb, c("Test data", "Short-term future"))) +
    scale_shape_manual(labels = c("Empirical risk", "Predicted: best expert-\naugmented model", "Predicted: base RuleFit"), values=c(19, 4, 1)) +
    scale_x_discrete(name = "Decision rule defining a group of job-seekers") +
    scale_y_continuous(name = "Mean risk of not resuming work in 180 days") +
    scale_color_manual(values = c("#f8766d", #R101
                                  "#26545a", #R50
                                  "#2d656c", #R85
                                  "#35757e", #R93
                                  "#3c8690", #R127
                                  "#4497a2", #R96
                                  "#4ba8b4", #R61
                                  "#5db0bb", #R18
                                  "#6fb9c3", #R119
                                  "#81c2ca") #R39
                       ) +
    theme_bw() +
    labs(color='') +
    theme(legend.title = element_blank()) +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
    theme(text = element_text(family = 'serif', size = 14))
  
  
  
  ### MSE FOR EACH RULE

  
  ggplot(data = reshape(rule_probs, direction='long', 
                        varying=c('test_data', 'test_data_prop_obs', 'test_data_pred_conf', 'test_data_pred_ridge', 'test_data_pred_rf', 'test_data_mse_conf', 'test_data_mse_ridge', 'test_data_mse_rf',
                                  'test_data_new', 'test_data_new_prop_obs','test_data_new_pred_conf', 'test_data_new_pred_ridge', 'test_data_new_pred_rf','test_data_new_mse_conf', 'test_data_new_mse_ridge', 'test_data_new_mse_rf',
                                  'test_data_new_long', 'test_data_new_long_prop_obs','test_data_new_long_pred_conf', 'test_data_new_long_pred_ridge', 'test_data_new_long_pred_rf', 'test_data_new_long_mse_conf', 'test_data_new_long_mse_ridge', 'test_data_new_long_mse_rf'
                        ), 
                        timevar='test_set',
                        times=c('t1', 't2', 't3'),
                        v.names=c('measurement_1', 'measurement_2', 'measurement_3', 'measurement_4', 'measurement_5', 'measurement_6', 'measurement_7', 'measurement_8'),
                        idvar=c('rule_id')) %>% 
           rename("empirical_risk" = 'measurement_1', "prop_obs" = 'measurement_2', "pred_eaml1" = 'measurement_3', "pred_ridge" = 'measurement_4', "pred_rf" = 'measurement_5', "mse_eaml1" = 'measurement_6', "mse_ridge" = 'measurement_7', "mse_rf" = 'measurement_8') %>% 
           mutate(test_set_verb = ifelse(test_set == "t1", "Test data", 
                                         ifelse(test_set == "t2", "Short-term future", 
                                                ifelse(test_set == "t3", "Mid-term future", "error"))))) +
    #geom_smooth(span = 0.3, show.legend = FALSE, aes(y = mse_eaml1, x = training_empirical_risk, fill = "Expert-augmented model (R ≤ 1)")) +
    #geom_smooth(span = 0.3, show.legend = FALSE, aes(y = mse_ridge, x = training_empirical_risk, fill = "Base RuleFit model")) +
    #geom_smooth(span = 0.3, show.legend = FALSE, aes(y = mse_rf, x = training_empirical_risk, fill = "Random Forest")) +
    geom_smooth(span = 0.3, size = 1.1, se=FALSE, aes(y = mse_eaml1, x = training_empirical_risk, color = "Best expert-augmented model")) +
    geom_smooth(span = 0.3, size = 1.1, se=FALSE, aes(y = mse_ridge, x = training_empirical_risk, color = "Base RuleFit model")) +
    geom_smooth(span = 0.3, size = 1.1, se=FALSE, aes(y = mse_rf, x = training_empirical_risk, color = "Random Forest")) +
    geom_point(data = data.frame(
      test_set_verb = "Short-term future", "Test data", "Mid-term future"
    ), mapping = aes(x = 0.7, y = 0.645), size=25, shape=1, color="black") +
    scale_x_continuous(name = "Mean empirical risk of not resuming work in 180 days") +
    scale_y_continuous(name = "Area Under the Curve (AUC)") +
    facet_wrap(~fct_relevel(test_set_verb, c("Test data", "Short-term future", "Mid-term future"))) +
    theme_bw() +
    labs(color='') +
    theme(legend.title = element_blank()) +
    theme(legend.position="bottom") +
    theme(text = element_text(family = 'serif', size = 14))
  
  ### PROPORTION OF CASES FOR EACH RISK:
  
  ggplot(aes(x = prob, fill = model),
    data = gather(rbind(test.data.new.long %>% 
                        mutate(test_set = "Mid-term test data"), 
                      rbind(test.data %>% 
                        mutate(test_set = "Test data"),
                      test.data.new %>% 
                        mutate(test_set = "Short-term test data"))) %>% 
           #mutate(loige = cut(ridge.prob, breaks = c(0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1), labels = c("10%","20%", "30%","40%", "50%", "60%", "70%", "80%", "90%","100%"))),
         model, prob, c(ridge.prob, eaml_R1.prob))) +
    stat_bin(aes(y=..density..)) +
    geom_line(stat="density", size = 1) +
    scale_x_continuous(name = "Predicted risk of not resuming work in 180 days (mean)") +
    scale_y_continuous(labels = percent, name = "percent") +
    facet_wrap(~test_set) +
    theme_bw() +
    labs(color='') +
    theme(legend.title = element_blank()) +
    theme(legend.position="bottom") +
    theme(text = element_text(family = 'serif', size = 10))


  
  
  
  
  
  
#####   FUNCTIONS   #####
  
  # Help function to calculate variable importance for survey responses
  
  arvuta_varimp <- function(vastused) {
    
    # defining vars -----------------------------------------------------------
    
    AEG_HOIVEST = c()
    TOOVOIME = c()
    VANUS_34 = c()
    TOOTAMISE_PERIOODE = c()
    DAYS_APPOINTED = c()
    TOOTU_PAEVI_3A = c()
    SUM_TK_YLEVAL_TS_SUHE = c()
    VALJAMAKSEDKUUD24ANTE = c()
    EMAILOLEMAS = c()
    KAT_B = c()
    VIIM_HOIVE_LOPP_POHJUSGR = c()
    VIIM_HOIVE_KESTUSGR = c()
    PALGATOETUS_3APOS = c()
    OPPEVALDKOND = c()
    
    R85 = c("TOOTAMISE_PERIOODE", "VIIM_HOIVE_LOPP_POHJUSGR", "VALJAMAKSEDKUUD24ANTE", "AEG_HOIVEST", "DAYS_APPOINTED")
    R54 = c("AEG_HOIVEST", "VANUS_34", "KAT_B")
    R60 = c("VIIM_HOIVE_LOPP_POHJUSGR", "VALJAMAKSEDKUUD24ANTE", "AEG_HOIVEST", "DAYS_APPOINTED")
    R36 = c("TOOTAMISE_PERIOODE", "AEG_HOIVEST")
    R69 = c("AEG_HOIVEST", "VIIM_HOIVE_KESTUSGR", "DAYS_APPOINTED", "PALGATOETUS_3APOS")
    R59 = c("TOOTAMISE_PERIOODE", "VALJAMAKSEDKUUD24ANTE", "DAYS_APPOINTED")
    R61 = c("TOOTAMISE_PERIOODE", "VALJAMAKSEDKUUD24ANTE", "AEG_HOIVEST", "TOOTU_PAEVI_3A", "DAYS_APPOINTED")
    R109 = c("TOOTAMISE_PERIOODE", "AEG_HOIVEST", "TOOVOIME", "EMAILOLEMAS")
    R41 = c("TOOTAMISE_PERIOODE", "VALJAMAKSEDKUUD24ANTE", "AEG_HOIVEST", "VANUS_34", "DAYS_APPOINTED")
    R126 = c("AEG_HOIVEST", "VANUS_34", "OPPEVALDKOND")
    R137 = c("AEG_HOIVEST", "TOOVOIME")
    R25 = c("AEG_HOIVEST", "VANUS_34")
    R42 = c("TOOTAMISE_PERIOODE", "AEG_HOIVEST", "TOOVOIME", "DAYS_APPOINTED")
    R97 = c("TOOTAMISE_PERIOODE", "TOOVOIME", "TOOTU_PAEVI_3A")
    R15 = c("TOOTAMISE_PERIOODE", "AEG_HOIVEST", "VANUS_34")
    R116 = c("AEG_HOIVEST", "SUM_TK_YLEVAL_TS_SUHE", "VANUS_34")
    R105 = c("TOOTAMISE_PERIOODE", "VALJAMAKSEDKUUD24ANTE", "EMAILOLEMAS")
    R57 = c("TOOTAMISE_PERIOODE", "AEG_HOIVEST", "TOOVOIME")
    R13 = c("AEG_HOIVEST", "EMAILOLEMAS")
    R21 = c("AEG_HOIVEST", "VANUS_34", "KAT_B")
    
    reeglid = c("R85", "R54", "R60", "R36", "R69", "R59", "R61", "R109", "R41", "R126", "R137", "R25", "R42", "R97", "R15", "R116", "R105", "R57", "R13", "R21")
    
    # end of defining vars ----------------------------------------------------
    
    for (reegel in reeglid) {
      reegli_tunnused = eval(parse(text = reegel))
      print(reegli_tunnused)
      reegli_vastused = vastused %>% 
        select(reegel)
      reegli_vastused = as.numeric(as.vector(reegli_vastused[1,]))
      
      for (tunnus in reegli_tunnused) {
        varimp_uhele_tunnusele_uhes_reeglis = c()
        for (vastus in reegli_vastused) {
          varimp_uhele_tunnusele_uhes_reeglis = c(varimp_uhele_tunnusele_uhes_reeglis, abs(0.5-vastus)/length(reegli_tunnused))
          
          if (tunnus == "AEG_HOIVEST") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            AEG_HOIVEST = c(AEG_HOIVEST, varimp_skaleeritud)
          } else if (tunnus == "TOOVOIME") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            TOOVOIME = c(TOOVOIME, varimp_skaleeritud)
          } else if (tunnus == "VANUS_34") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            VANUS_34 = c(VANUS_34, varimp_skaleeritud)
          } else if (tunnus == "TOOTAMISE_PERIOODE") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            TOOTAMISE_PERIOODE = c(TOOTAMISE_PERIOODE, varimp_skaleeritud)
          } else if (tunnus == "DAYS_APPOINTED") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            DAYS_APPOINTED = c(DAYS_APPOINTED, varimp_skaleeritud)
          } else if (tunnus == "TOOTU_PAEVI_3A") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            TOOTU_PAEVI_3A = c(TOOTU_PAEVI_3A, varimp_skaleeritud)
          } else if (tunnus == "SUM_TK_YLEVAL_TS_SUHE") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            SUM_TK_YLEVAL_TS_SUHE = c(SUM_TK_YLEVAL_TS_SUHE, varimp_skaleeritud)
          } else if (tunnus == "VALJAMAKSEDKUUD24ANTE") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            VALJAMAKSEDKUUD24ANTE = c(VALJAMAKSEDKUUD24ANTE, varimp_skaleeritud)
          } else if (tunnus == "EMAILOLEMAS") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            EMAILOLEMAS = c(EMAILOLEMAS, varimp_skaleeritud)
          } else if (tunnus == "KAT_B") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            KAT_B = c(KAT_B, varimp_skaleeritud)
          } else if (tunnus == "VIIM_HOIVE_LOPP_POHJUSGR") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            VIIM_HOIVE_LOPP_POHJUSGR = c(VIIM_HOIVE_LOPP_POHJUSGR, varimp_skaleeritud)
          } else if (tunnus == "VIIM_HOIVE_KESTUSGR") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            VIIM_HOIVE_KESTUSGR = c(VIIM_HOIVE_KESTUSGR, varimp_skaleeritud)
          } else if (tunnus == "PALGATOETUS_3APOS") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            PALGATOETUS_3APOS = c(PALGATOETUS_3APOS, varimp_skaleeritud)
          } else if (tunnus == "OPPEVALDKOND") {
            varimp_skaleeritud = varimp_uhele_tunnusele_uhes_reeglis
            OPPEVALDKOND = c(OPPEVALDKOND, varimp_skaleeritud)
          }
          
        }
      }
    }
    
    result <- data.frame (AEG_HOIVEST = sum(AEG_HOIVEST)/length(AEG_HOIVEST),
                          TOOVOIME = sum(TOOVOIME)/length(TOOVOIME),
                          VANUS_34 = sum(VANUS_34)/length(VANUS_34),
                          TOOTAMISE_PERIOODE = sum(TOOTAMISE_PERIOODE)/length(TOOTAMISE_PERIOODE),
                          DAYS_APPOINTED = sum(DAYS_APPOINTED)/length(DAYS_APPOINTED),
                          TOOTU_PAEVI_3A = sum(TOOTU_PAEVI_3A)/length(TOOTU_PAEVI_3A),
                          SUM_TK_YLEVAL_TS_SUHE = sum(SUM_TK_YLEVAL_TS_SUHE)/length(SUM_TK_YLEVAL_TS_SUHE),
                          VALJAMAKSEDKUUD24ANTE = sum(VALJAMAKSEDKUUD24ANTE)/length(VALJAMAKSEDKUUD24ANTE),
                          EMAILOLEMAS = sum(EMAILOLEMAS)/length(EMAILOLEMAS),
                          KAT_B = sum(KAT_B)/length(KAT_B),
                          VIIM_HOIVE_LOPP_POHJUSGR = sum(VIIM_HOIVE_LOPP_POHJUSGR)/length(VIIM_HOIVE_LOPP_POHJUSGR),
                          VIIM_HOIVE_KESTUSGR = sum(VIIM_HOIVE_KESTUSGR)/length(VIIM_HOIVE_KESTUSGR),
                          PALGATOETUS_3APOS = sum(PALGATOETUS_3APOS)/length(PALGATOETUS_3APOS),
                          OPPEVALDKOND = sum(OPPEVALDKOND)/length(OPPEVALDKOND))
    
    result
  } 
  
  # Help function for displaying means, medians and modes for rule-defined subpopulations:
  
  konverteeri_reegel <- function(tunnused, reegli_id, mudel) {
    subpopulation_rule <- mudel$mod$rules.selected[reegli_id]
    subpopulation <- data.frame(training.data %>% 
                                  filter(eval(parse(text=subpopulation_rule))))
    
    for (tunnus in tunnused) {
      
      col_id = which(colnames(subpopulation) == tunnus)
      vartype = sapply(subpopulation, class)[col_id]
      
      print(paste0("TUNNUS: ", tunnus, " (", vartype, ")"))
      
      if (vartype %in% c("factor", "c(\"ordered\", \"factor\")")) {
        mood = tail(names(sort(table(subpopulation[,col_id]))), 1)
        print(paste0("Tunnuse ", tunnus, " mood: ", mood))
        print(paste0("Tunnuse ", tunnus, " kõik tasemed: ", toString(unique(subpopulation[,col_id]))))
      } else if (vartype == "numeric") {
        keskmine = mean(subpopulation[,col_id])
        print(paste0("Tunnuse ", tunnus, " keskmine: ", keskmine))
        print(paste0("Tunnuse ", tunnus, " range: ", min(subpopulation[,col_id]), "-", max(subpopulation[,col_id])))
      }
      
      print("-------------------------")
    }
  }
  
  
  konverteeri_reegel(c("AEG_HOIVEST", "TOOVOIME"), 56, ruleFeat)
  
  
  
  
  
  
  


