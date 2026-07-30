# OULAD Student Dropout Risk Prediction

## Issue Resolution Log

### [Issue #6 - Loss Function Selection](https://github.com/adityamd/iiith-capstone-project/issues/6)

**Decision**

Use **binary cross-entropy (log loss) with balanced sample weights** for the preferred HistGradientBoosting model. The target is binary (`Withdrawn = 1`; all other outcomes = `0`), and the dataset is moderately imbalanced (`31.16%` withdrawn versus `68.84%` not withdrawn). Weighting gives the minority withdrawal class appropriate importance without changing the held-out test distribution.

This choice also reflects the operational cost of a false negative: failing to identify a student who later withdraws means losing an opportunity for early support. Accuracy alone is therefore insufficient. Model evaluation should prioritize ROC-AUC and PR-AUC and report withdrawal precision, recall, F1, and the confusion matrix. The intervention threshold should be selected from validation data using the recall/precision trade-off and available advisor capacity rather than permanently defaulting to `0.5`.

Focal loss is not recommended at this stage because withdrawal is moderately, rather than extremely, rare, and weighted log loss has already produced strong baseline results. Class weighting and the Random Oversampling, SMOTE, and ADASYN experiments from [Issue #4](https://github.com/adityamd/iiith-capstone-project/issues/4) should be compared as alternative imbalance treatments; they should not be combined automatically.

Evidence: [Checkpoint 1 EDA and model comparison notebook](https://github.com/adityamd/iiith-capstone-project/blob/master/dev_playground_shujath/checkpoint1_EDA_v1.ipynb)

---

### [Issue #5 - Model Selection](https://github.com/adityamd/iiith-capstone-project/issues/5)

**Decision**

Carry forward **weighted HistGradientBoosting as the preferred current candidate**, while retaining balanced Logistic Regression as the interpretable benchmark. This is a provisional selection until the augmentation experiments from [Issue #4](https://github.com/adityamd/iiith-capstone-project/issues/4), hyperparameter tuning, intervention-threshold selection, and subgroup evaluation are complete.

| Model | ROC-AUC | PR-AUC | Withdrawal precision | Withdrawal recall | Withdrawal F1 |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting (weighted) | **0.881** | **0.820** | 0.721 | **0.753** | **0.737** |
| Random Forest (balanced) | 0.878 | 0.807 | **0.804** | 0.642 | 0.714 |
| Logistic Regression (balanced) | 0.871 | 0.786 | 0.708 | 0.737 | 0.722 |

These results were produced using a stratified 75%/25% train-test split, balanced class/sample weighting, and a threshold of `0.5` for precision, recall, and F1. HistGradientBoosting provided the strongest overall ranking performance and identified more withdrawn students than Random Forest, while achieving the highest F1 score.

**HistGradientBoosting primer**

HistGradientBoosting is a tree-based boosting algorithm. It groups continuous values into histogram bins to make split searches efficient, then builds small decision trees sequentially. Each new tree focuses on correcting errors made by the existing ensemble. This allows the model to learn nonlinear effects and interactions, such as low engagement combined with high workload or limited prior education.

- **Logistic Regression** learns one linear probability relationship. It is fast and interpretable but requires nonlinearities and interactions to be specified explicitly.
- **Random Forest** trains many independent trees on different samples and features, then averages their predictions. It is robust and captured nonlinear effects, but in this experiment it had the lowest withdrawal recall.
- **HistGradientBoosting** trains trees sequentially so that later trees correct earlier errors. It produced the best ROC-AUC, PR-AUC, withdrawal recall, and F1 in the completed baseline comparison.

The subgroup disparities identified in [Issue #2](https://github.com/adityamd/iiith-capstone-project/issues/2) mean that final confirmation must include performance checks across sufficiently sized education, workload, deprivation, disability, gender, and other relevant groups.

Evidence: [Checkpoint 1 EDA and model comparison notebook](https://github.com/adityamd/iiith-capstone-project/blob/master/dev_playground_shujath/checkpoint1_EDA_v1.ipynb)

---

### [Issue #2 - Correlation data for outcomes across multiple variables](https://github.com/adityamd/iiith-capstone-project/issues/2)

**Evidence**

Add the result here. Evidence can be a short explanation, metric, output, code or notebook link, table, or plot.

The following analyses examine student withdrawal rates across different combinations of demographic, socioeconomic, academic, and behavioural variables. The objective is to identify high-risk student groups and understand how interactions between multiple factors influence the likelihood of student withdrawal.

Notebook:
https://github.com/adityamd/iiith-capstone-project/blob/master/Kiran/Student_Dropout_EDA_v1.ipynb
- Result:

Gender × Region — Withdrawal Analysis
Plot:
<img width="1061" height="490" alt="image" src="https://github.com/user-attachments/assets/3bcfbcf6-89f9-4e20-9f6e-b99e1a6a7f52" />
Summary:
Withdrawal rates are relatively similar for male and female students across most regions, suggesting that gender alone does not significantly influence student withdrawal. However, regional differences are evident, with the North Western Region and West Midlands Region showing the highest withdrawal rates for both genders, while Ireland, Wales, and Scotland consistently exhibit lower withdrawal rates. Overall, these findings indicate that region has a greater influence on student withdrawal than gender, although the differences across regions are moderate rather than substantial.

Gender × Age Band — Withdrawal Analysis 
Plot:
<img width="1069" height="490" alt="image" src="https://github.com/user-attachments/assets/2f956256-1883-4f8c-8dc4-7c023168df1e" />
Summary:
Withdrawal rates are relatively similar for male and female students in the 0–35 and 35–55 age groups, indicating that gender has only a limited influence on withdrawal within these age categories. However, a notable difference is observed in the 55+ age group, where male students have a considerably lower withdrawal rate (24.1%) than female students (31.0%). Overall, the analysis suggests that age has a greater influence on withdrawal than gender, with older students, particularly males, demonstrating better retention compared to younger and middle-aged students.

Gender × Highest Education — Withdrawal Analysis 
Plot:
<img width="1060" height="490" alt="image" src="https://github.com/user-attachments/assets/15bb2eb6-d3b6-42b4-a3cb-9b7451185b37" />
Summary:
Withdrawal rates vary considerably across different levels of highest education, indicating that educational background has a much stronger influence on student withdrawal than gender. Students with No Formal Qualifications exhibit the highest withdrawal rates for both males (42.9%) and females (43.0%), whereas students with Post Graduate Qualifications consistently record the lowest withdrawal rates (25.0% for males and 20.2% for females). Overall, the findings suggest that lower educational attainment is strongly associated with a higher likelihood of withdrawal, while gender differences within each education level remain relatively small.
Gender × IMD Band — Withdrawal Analysis 
Plot:
<img width="1069" height="490" alt="image" src="https://github.com/user-attachments/assets/9170d7d6-3484-4de9-86c5-fa748fa8fad1" />
Summary:
Withdrawal rates show a clear relationship with IMD (Index of Multiple Deprivation) band, indicating that socioeconomic status has a greater influence on student withdrawal than gender. Both male and female students from the most deprived areas (0–30% IMD bands) consistently exhibit the highest withdrawal rates, while students from the least deprived areas (70–100% IMD bands) generally record the lowest withdrawal rates. Overall, the findings suggest that withdrawal rates decrease as socioeconomic status improves, with only minor differences observed between male and female students within each deprivation band.

Gender × Disability — Withdrawal Analysis 
Plot:
<img width="1069" height="490" alt="image" src="https://github.com/user-attachments/assets/a65b4bd2-578b-43a5-a144-c39c2b7137fb" />
Summary:
Withdrawal rates are consistently higher for students reporting a disability compared to those without a disability, regardless of gender. Male students with a disability exhibit the highest withdrawal rate (41.7%), followed by female students with a disability (37.1%), whereas students without a disability have considerably lower withdrawal rates (30.8% for males and 29.7% for females). Overall, the findings suggest that disability status has a stronger influence on student withdrawal than gender, indicating that students with disabilities may require additional academic and support interventions to improve retention.

Region × Age Band — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/3bac6547-cada-4b18-b1ae-67b8b3367767" />
Summary:
Withdrawal rates vary considerably across different region and age group combinations, indicating a clear interaction between these two variables. The highest withdrawal rates are observed among students aged 55+ in the North Western Region, North Region, and Wales, whereas students aged 55+ in the East Anglian Region, South East Region, and Yorkshire Region exhibit the lowest withdrawal rates. Overall, the findings suggest that the effect of age on student withdrawal differs across regions, demonstrating that regional factors influence withdrawal differently for each age group rather than following a single consistent trend.

Region × Highest Education — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/2b5d9cad-15f9-47de-95c0-d0a7d90d684c" />
Summary:
Withdrawal rates vary substantially across different combinations of region and highest education level, indicating that these two factors jointly influence student withdrawal. Students with No Formal Qualifications consistently exhibit the highest withdrawal rates across almost all regions, whereas those with Post Graduate Qualifications generally record the lowest withdrawal rates regardless of region. Overall, the findings suggest that highest education level has a stronger influence on withdrawal than region, although regional differences still contribute to variations in withdrawal risk within each education level.

Region × IMD Band — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/f0ad0bfa-37db-46e6-a44b-5c11e4643fdd" />
Summary:
Withdrawal rates vary across different combinations of region and IMD (Index of Multiple Deprivation) band, indicating that both geographical location and socioeconomic status jointly influence student withdrawal. Students from the most deprived IMD bands (0–20%) consistently exhibit the highest withdrawal rates across several regions, while students from the least deprived IMD bands (80–100%) generally record the lowest withdrawal rates regardless of region. Overall, the findings suggest that socioeconomic deprivation has a stronger influence on withdrawal than geographical region, although regional differences further contribute to variations in withdrawal risk within each IMD band.

Previous Attempts × Gender — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/2b57c8a5-b861-4176-8bd2-2847a6c978e7" />
Summary:
Withdrawal rates generally increase as the number of previous course attempts increases, indicating that students with a history of repeated enrolments are more likely to withdraw from subsequent courses. Female students with three or more previous attempts exhibit the highest withdrawal rates, reaching 50.0% for six previous attempts, while students with no previous attempts have the lowest withdrawal rates for both genders. Overall, the findings suggest that the number of previous attempts has a stronger influence on student withdrawal than gender, although the results for students with four or more attempts should be interpreted cautiously due to the very small number of observations in these groups.

Previous Attempts × Age Band — Withdrawal Analysis 
Plot:
<img width="1589" height="790" alt="image" src="https://github.com/user-attachments/assets/83dee218-33f4-4109-84ff-2b19533d1913" />
Summary:
Withdrawal rates generally increase as the number of previous course attempts increases across all age groups, suggesting that repeated enrolments are associated with a greater likelihood of withdrawal. Students aged 35–55 with three or more previous attempts exhibit the highest withdrawal rates, while students with no previous attempts, particularly those aged 55 and above, have the lowest withdrawal rates. Overall, the findings indicate that previous course attempts have a stronger influence on student withdrawal than age, although the results for students with very high numbers of previous attempts should be interpreted cautiously because these groups contain very few observations.

Credit Group × Age Band — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/16dfa523-a55d-4b4e-9d3a-c4802fc12f85" />
Summary:
Withdrawal rates increase consistently as the academic workload (credit group) increases across all age groups, indicating that students enrolled in higher-credit courses are more likely to withdraw. Students taking Very High (>120 credits) courses exhibit the highest withdrawal rates across every age band, whereas students enrolled in Low (≤60 credits) courses consistently record the lowest withdrawal rates. Overall, the findings suggest that academic workload has a stronger influence on student withdrawal than age, although students aged 55 and above undertaking very high credit loads appear to be particularly

Credit Group × Highest Education — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/86f8384b-e3dc-44c2-aacf-31f80e7f5cd4" />
Summary:
Withdrawal rates are strongly influenced by the combination of academic workload (credit group) and highest education level, indicating that these factors jointly affect student retention. Students with No Formal Qualifications enrolled in Very High (>120 credits) courses exhibit the highest withdrawal rates, while students with Post Graduate Qualifications taking Low (≤60 credits) credit loads consistently record the lowest withdrawal rates. Overall, the findings suggest that both prior educational attainment and academic workload are important predictors of withdrawal, with heavier study loads having a much greater impact on students with lower educational qualifications.

VLE Engagement × Highest Education — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/a3818741-4712-4f67-b48a-97b9a37da405" />
Summary:
Withdrawal rates decrease substantially as VLE engagement increases, regardless of students' highest education level, indicating that active participation in the Virtual Learning Environment is strongly associated with improved student retention. Students with Low VLE engagement consistently exhibit the highest withdrawal rates across all education levels, while those with Very High VLE engagement have the lowest withdrawal rates, including students with No Formal Qualifications. Overall, the findings suggest that VLE engagement is a stronger predictor of student withdrawal than educational background, highlighting learner engagement as one of the most important behavioural factors influencing student retention.

VLE Engagement × Credit Group — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/278a9f41-791b-4b20-8c64-59df015b9d4d" />
Summary:
Withdrawal rates are strongly influenced by the combination of VLE engagement and academic workload (credit group), demonstrating a clear interaction between student engagement and course intensity. Students with Low VLE engagement consistently exhibit the highest withdrawal rates across all credit groups, particularly those enrolled in Very High (>120 credits) courses, whereas students with Very High VLE engagement record the lowest withdrawal rates regardless of academic workload. Overall, the findings suggest that VLE engagement is a stronger predictor of student withdrawal than academic workload, highlighting the importance of active participation in online learning for improving student retention, even in courses with heavier study loads.

Highest Education × Credit Group × IMD Band — Withdrawal Analysis 
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/9dbe71f2-17dc-40a8-81a3-e6767e1d73fb" />
Summary:
Withdrawal rates vary considerably across the combined effects of highest education level, credit group, and IMD band, indicating that student withdrawal is influenced by the interaction of academic preparedness, study workload, and socioeconomic background rather than any single factor alone. Students with No Formal Qualifications who are enrolled in higher credit groups frequently appear among the highest-risk groups, while combinations involving stronger educational backgrounds generally exhibit lower withdrawal rates. Although some combinations show withdrawal rates close to or equal to 100%, these values are likely based on very small sample sizes and should therefore be interpreted with caution. Overall, the analysis demonstrates that student withdrawal is driven by the combined influence of multiple demographic, academic, and socioeconomic factors, highlighting the importance of considering interaction effects when identifying high-risk students.

Highest Education × VLE Engagement × Credit Group — Withdrawal Analysis
Plot:
<img width="1590" height="790" alt="image" src="https://github.com/user-attachments/assets/20a89db3-1acd-4d3b-bb2b-ab6ca1797150" />
Summary:
Withdrawal rates vary substantially across the combined effects of highest education level, VLE engagement, and credit group, indicating that student withdrawal is influenced by the interaction of academic preparedness, learner engagement, and course workload. Students with Low VLE engagement consistently appear among the highest-risk groups across different education levels and credit groups, particularly when enrolled in High or Very High (>120 credits) courses. In contrast, combinations involving Medium or High VLE engagement generally exhibit lower withdrawal rates, regardless of educational background. Although some groups show withdrawal rates close to 100%, these results are likely based on very small sample sizes and should therefore be interpreted with caution. Overall, the analysis highlights that VLE engagement plays a critical role in student retention, and its combined effect with academic workload and educational background provides valuable insights for identifying students at the greatest risk of withdrawal.  

---
Add the latest resolved issue at the top.

---

### [Issue #NN - Issue title](https://github.com/adityamd/iiith-capstone-project/issues/NN)

**Evidence**

Add the result here. Evidence can be a short explanation, metric, output, code or notebook link, table, or plot.

Examples:

- [Notebook or code](path/to/file)
- Result: `metric = value`
- Plot: ![Plot description](path/to/plot.png)

---

## How to Update this README

1. Copy the issue entry above.
2. Paste it at the top of the **Issue Resolution Log**.
3. Replace the issue number, title, and link.
4. Replace the example evidence with the relevant result, file, output, table, or plot.
