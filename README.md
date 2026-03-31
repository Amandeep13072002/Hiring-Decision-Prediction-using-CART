# Hiring-Decision-Prediction-using-CART
An end-to-end Machine Learning project using Classification and Regression Trees (CART) to automate hiring decisions — covering EDA, hyperparameter tuning, cost complexity pruning, and a three-model comparison.

Business Problem
Imagine an HR Manager receiving hundreds of resumes every week. Manually screening each candidate — reading qualifications, checking scores, deciding who to shortlist — is slow, expensive, and prone to human bias.
This project builds a machine learning classifier that predicts whether a candidate should be hired (1) or not (0) based on their profile. The model learns patterns from historical hiring data and provides explainable, data-driven recommendations.
Impact: Automates initial screening, reduces bias, and cuts recruitment time significantly.

Dataset
File: recruitment_data.csv
Records: 1,500 candidates
Target Variable: HiringDecision (0 = Not Hired, 1 = Hired)

Feature Type  Description  
AgeNumericCandidate age (20–50)GenderBinary0 = Female, 1 = MaleEducationLevelOrdinal1 = Bachelor's → 4 = PhDExperienceYearsNumericTotal years of work experience (0–15)PreviousCompaniesNumericNumber of previous employers (1–5)DistanceFromCompanyNumericDistance in km from workplaceInterviewScoreNumericScore from interview panel (0–100)SkillScoreNumericTechnical skill assessment score (0–100)PersonalityScoreNumericPersonality/behavioural score (0–100)RecruitmentStrategyCategoricalChannel used (1 = Aggressive, 2 = Moderate, 3 = Conservative)HiringDecisionBinary0 = Not Hired, 1 = Hired (Target)

Statistical Summary:
Mean Age: 35.15 | Mean Experience: 7.69 years
Dataset is balanced with a reasonable Hired vs Not Hired split
No missing values — clean dataset ready for modelling

Project Workflow:
Data Loading → EDA → Preprocessing → Baseline CART → Evaluation
     ↓
GridSearchCV Tuning → Tuned Model → Cross-Validation → Overfitting Analysis
     ↓
Cost Complexity Pruning → Pruned Model → Model Comparison → Business Insights


Sections at a Glance:
SectionDescription
1. ImportsAll libraries loaded upfront
2. Problem StatementHR automation business context
3. Load DataCSV → DataFrame, shape & preview
4. EDAdescribe(), distributions, correlation heatmap, boxplots
5. PreprocessingFeature/target split, StandardScaler, 80/20 stratified split
6. Baseline CARTDecisionTreeClassifier(criterion='gini') — no restrictions
7. Evaluation (Baseline)Accuracy, ROC-AUC, confusion matrix, ROC curve
8. Tree VisualizationText + visual tree diagram (depth=3)
9. Feature ImportanceBar chart of feature contribution (Baseline)
10. Hyperparameter TuningGridSearchCV over 90 combinations, StratifiedKFold(5)
11. Tuned Model EvaluationConfusion matrix, classification report
12. Cross-Validation10-fold CV, per-fold bar chart
13. Overfitting AnalysisTrain vs Test accuracy across depths 1–20
14. Cost Complexity PruningCCP alpha path, pruned tree selection
15. Model ComparisonBaseline vs Tuned vs Pruned — full metrics table + ROC curves
16. Feature ImportanceTuned model feature ranking
17. Prediction Functionpredict_hiring(...) — reusable inference on new candidates
18. Final SummaryBusiness insights and model recommendation

Results:

Model Comparison Table
Model                     Accuracy    ROC-AUC    Avg Precision  Tree Depth      Num Leaves
Baseline CART              0.8933      0.8724    0.739213109       13            109
Tuned CART (GridSearchCV)  0.9167      0.9116    0.85061277        12             77
Pruned CART (CCP)          0.9400      0.9256    0.8869631         6              31
Best Model: Pruned CART

Accuracy: 94.00% — correctly classifies 94 out of 100 candidates
ROC-AUC: 0.9256 — excellent ability to separate hired vs not hired
Tree Depth: 6 — simple, interpretable, and generalizable
Leaves: 31 — drastically reduced from 109 (baseline) → much less overfitting

GridSearchCV — Best Parameters Found
criterion          = gini
max_depth          = None
min_samples_leaf   = 4
min_samples_split  = 2
Best CV ROC-AUC    = 0.9093
10-Fold Cross-Validation (Tuned Model)
Mean ROC-AUC  : 0.9143
Std Deviation : 0.1193
Min Score     : 0.5826  (Fold 10)
Max Score     : 0.9900  (Fold 2)

Key Visualizations
The notebook generates 15 plots across the pipeline:
PlotDescriptionPlot 1Target class distribution (Hired vs Not Hired)Plot 2Feature distributions (6 numerical variables)Plot 3Correlation heatmapPlot 4Boxplots by hiring decisionPlot 5Confusion matrix — Baseline CARTPlot 6ROC curve — Baseline CARTPlot 7Decision tree diagram (depth=3)Plot 8Feature importance — BaselinePlot 9Confusion matrix — Tuned CARTPlot 10Top-20 GridSearchCV parameter combinationsPlot 1110-fold cross-validation scores per foldPlot 12Overfitting analysis (Train vs Test accuracy)Plot 13CCP Alpha vs Accuracy curvePlot 14ROC curves — all 3 models overlaid + bar comparisonPlot 15Feature importance — Tuned model

 Business Insights
Top 3 Most Influential Features
RankFeatureImportance ScoreBusiness Meaning🥇 1RecruitmentStrategy0.3705The channel/strategy used to recruit drives hiring outcomes most strongly🥈 2ExperienceYears0.1390Work experience is the second strongest signal🥉 3SkillScore0.1344Technical skill assessment is a reliable predictor
Key Findings

Overfitting is real: The unconstrained baseline tree grew 13 levels deep with 109 leaves — clearly memorizing the training data, not learning generalizable patterns.
Pruning > Tuning: Cost Complexity Pruning (CCP) outperformed GridSearchCV tuning on all metrics. A simpler tree (depth=6, 31 leaves) generalized better than a complex one.
RecruitmentStrategy dominates: With 37% feature importance, the recruitment channel has the biggest influence on hiring outcomes — suggesting HR teams should invest in data-driven channel selection.
Gender has zero impact: Gender scored 0.0000 importance in the tuned model, confirming the model makes decisions without gender bias.
Interpretability advantage: Unlike black-box models, CART's decisions can be visualized as flowcharts — making it auditable and explainable to HR stakeholders.
