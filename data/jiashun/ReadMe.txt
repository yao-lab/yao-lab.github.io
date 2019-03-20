

Data files for the coauthorship and citation networks for statisticians 


2014/10/13



authorList.txt - the list of 3607 author names, sorted alphabetically

paperDoiList.txt - the list of DOIs for 3248 papers 

authorPaperBiadj.txt - the 3607x3248 bipartite adjacency matrix for the authors and papers; the element (i,j) is 1 iff author i authored/coauthored paper j, and 0 otherwise; the authors are sorted as in authorList.txt and the papers are sorted as in paperDoiList.txt.

paperCitAdj.txt - the 3248x3248 asymmetric adjacency matrix for citations between the papers; the element (i,j) is 1 iff paper i cites paper j.


In the folder of "coauthorship":

authorListCoauthorGiant.txt  - list of 2263 authors in the giant component of the coauthorship network; sorted alphabetically

authorListCoauthorThreshGiant.txt - list of 236 authors in the giant component of the coauthorship network where each edge denotes at least t=2 papers coauthored

coauthorAdj.txt - 3607x3607 adjacency matrix of the coauthorship network

coauthorAdjGiant.txt - 2263x2263 adjacency matrix of the giant component of the coauthorship network where each edge denotes at least t=1 paper coauthored.

coauthorAdjThreshGiant.txt -  236x236 adjacency matrix of the giant component of the coauthorship network where each edge denotes at least t=2 papers coauthored  



In the folder of "citation":

authorCitAdj.txt - 3607x3607 adjacency matrix for the citations between authors; the element (i,j) is 1 iff author i cites author j at least once

authorCitAdjGiant.txt - 2654x2654 adjacency matrix for the weakly connected giant component of the citation network of authors; a submatrix of authorCitAdj.txt

authorListCitGiant.txt - list of 2654 authors in the weakly connected giant component of the citation network of authors; sorted alphabetically
