# Report Review

## 06. CUI Yiran

1. Summary: This report uses PCA, Kernel-PCA, MDS and random projections with PCA and MDS to explore SNPs data. The results are based on the continent-level data, and it shows a linear relationship with human's migration history.
2. Strength
   - Methods are explained in the theoretical background part.
   - The scripts are clear and starightforward.
   - Tested the effect of using different model settings.
   - Measured and compared the computation time of different methods.
3. Weakness
   - Infomation from the results are not extracted, i.e., only shows model setups and chart results, but didn't explain what these charts mean in human migration history.
   - Lack discussion and analysis on results from different models, i.e., the differences in the results from different models, the goodness and weakness of different models on this problem.
   - The dataset has different granularities, only region-level is analysed.
4. Evaluation on Clarity and quality of writing: 3
   - typo: mitigation --> migration
   - type: 2.3 kernel PCA --> 2.3 Kernel PCA
   - 3.1 didn't explain the impact of only taking 2 components.
   - Figure 1 shows the eigen values' value histogram, rather than the importance histogram. This figure is not stressed in text.
   - Part 3 mainly focuses on model settings, and how these settings will affect the chart results, but the author didn't connect these results to the human migration history. It is not confident to draw a conclusion like "Results show the SNPs inherit a linear relationship with people‘s mitigation history."
5. Evaluation on Technical Quality: 3
   - Codes are reproducible.
   - Strengths and weaknesses of different approaches are not discussed.
   - Analysis on results can be deepened and more related to human migration history.
6. Overall rating: 3
7. Assessment confidence: 3

## 08. MA Ruochen; Jihong TANG; Yuyan RUAN; Zhi HUANG

1. Summary: this project uses PCA, MDS, t-SNE, ISOMAP, LLE, UMAP, robust PCA and random projections to explore explore SNPs data. ARI score is used to compare different models.
2. Strength
   - 8 methods have been implemented in this project.
   - Used ARI score to evaluate the model prediction power.   -
3. Weakness
   - Lack model theory explanation.
   - Only uses region-level labels, while the dataset contains 4 granualarity levels.
   - The indication on the ancestry prediction from the results are not explained. This poster mainly focuses on model prediction wellness.
   - Lack explaination for the visualizations of the results.
4. Evaluation on Clarity and quality of writing: 3
5. Evaluation on Technical Quality: 3
6. Overall rating: 3
7. Assessment confidence: 2
   - Codes are complex to make a full check.

## 09. HUANG, Zhanmiao; Wencan XIA; Yuanhui LUO

1. Summary: this project uses PCA, MDS, random projections and Random Forest and Extra Trees to explore SNPs data. Dimension reduction methods are used to cluster data. random projections are used to explore their effects on PCA's performances. Random forests are used to identify the most important SNP. A case study is conducted to extract more hidden information.
2. Strength
   - Used 2D and 3D clustering in PCA and MDS.
   - Analysed the importance of SNPs using random forest.
   - Used a case study to explore the difference between China people and east Asia people.
3. Weakness
   - Only uses region-level labels, while the dataset contains 4 granualarity levels.
   - Methodology part can be explained in a more detailed way.
   - Did not analyse the geographic variations on the results from PCA/MDS and random projections.
   - Only uses random projections on PCA, but does not apply for MDS.
4. Evaluation on Clarity and quality of writing: 3
   - HKUST logo is almost hidden in the header part.
   - Poster is much more compact in information compared to report. In the result analysis part, more useful analysis should be included, rather than only explaining the findings on the surface. For example, the inner relationships among different continent data, how does the variation looks like geographically, etc.
5. Evaluation on Technical Quality: 4
   - Codes are clear and straightforward, but should remove unused comments, and include more section info.
6. Overall rating: 3.5
7. Assessment confidence: 3

## 14. CAI Bibi; QIU Zhenyu; WANG Zhiwei

1. Summary: this report uses LDA to extract topics from NIPs papers, and then uses K-Means, MDS and t-SNE to explore the data.
2. Strength
   - Methods are well explained.
3. Weakness
   - Abstract should not include questions. These questions should be described using declarative sentences.
   - No summary.
   - 4.3 results only illustrate what figure 5 and 6 shows, but does not make analysis, i.e., what these figures indicate.
4. Evaluation on Clarity and quality of writing: 4
   - Page 5 and page 6 results can be shown in a more compact way.
   - Brackets and space issue on page 8, i.e., Topic 6("probabilistic model & prediction model") --> Topic 6 ("probabilistic model & prediction model")
5. Evaluation on Technical Quality: 4
   - Codes are clear and straightforward, but should remove unused empty cells.
6. Overall rating: 4
7. Assessment confidence: 3

## 17. LI Qichao and HUANG Haohan

1. Summary: this report uses PCA, SPCA, KPCA and RF to explore handwritten digit data. Different cluster methods are implemented to classify the same dataset, and comparisons have been made in terms of computation cost and accuracies.
2. Strength
   - Methods are well explained.
3. Weakness
   - Random forest method should be explained in more details
   - Analysis are not enough. Should explain more on how and why problems, not only "what".
4. Evaluation on Clarity and quality of writing: 3
   - Title should be capitalized for each word.
   - typo: Methodology --> Methodologies
   - The topic is PCA family, but it also includes random forest.
5. Evaluation on Technical Quality: 4
   - Codes are clear and straightforward, but should remove unused empty cells.
6. Overall rating: 3.5
7. Assessment confidence: 3
