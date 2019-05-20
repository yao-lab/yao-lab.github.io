Agedata.mat is the pairwise comparison data we collected. 30 images from human age dataset are annotated by a group of volunteer users on ChinaCrowds platform. The annotator is presented with two images and given his choice of which one is older (or difficult to judge). Totally, we obtain 14,011 pairwise comparisons from 94 annotators. There are four columns. The first column is annotator_id, the second and third columns are individual_id, the last column is shows the annotator's choice:
1 indicates the second column is older than the third one,
-1 indicateds the second column is younger than the third one, 
0 indicates the second and third are difficult to judge.


For example, for one row (1,2, 3, 1)

It means annotator 1 provides individual 2 is older than individual 3. 


for one row (1,2, 3, -1)

It means annotator 1 provides individual 2 is younger than individual 3.


for one row (1,2, 3, 0)

It means annotator 1 provides individual 2 and individual 3 are difficult to judge.



Groundtruth.mat provides the groundtruth age of each individual. The first column is the individual_id, and the second column is the  groundtruth age of each individual.