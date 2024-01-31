%%% a
%% First, load a pre-stored torus embedded in 4-dimensional space with 400 samples using the following commands:
load tutorial_examples/pointsTorusGrid.mat
scatter3(pointsTorusGrid(:,1), pointsTorusGrid(:,2), pointsTorusGrid(:,3))

%%% b
%% Next, we will construct a Vietoris-Rips complex using the dataset. 
max_dimension = 3;
max_filtration_value = 0.9;
num_divisions = 1000;
stream = api.Plex4.createVietorisRipsStream(pointsTorusGrid, max_dimension, max_filtration_value, num_divisions);
num_simplices = stream.getSize();

%%% c
%% We can now compute the persistent homology for the Vietoris-Rips complex.
persistence = api.Plex4.getModularSimplicialAlgorithm(max_dimension, 2);
intervals = persistence.computeIntervals(stream);

%%% d
%% Plot the barcode for the persistent Betti numbers 
options.filename = 'ripsTorus4.png';
options.max_filtration_value = max_filtration_value;
options.max_dimension = max_dimension - 1;
options.side_by_side = true;
plot_barcodes(intervals, options);

%%% e
% The resulting plot shows the barcode of persistent Betti numbers for the Vietoris-Rips complex of the torus. 
% The horizontal axis represents the filtration values, and the vertical axis represents the Betti numbers. 
% Each bar represents a homology class that exists for a range of filtration values. 
% The length of the bar indicates the range of filtration values for which the homology class persists. 
% In this case, we see several bars representing the 0th Betti number, indicating the number of connected components,
% and several bars representing the 1st Betti number, indicating the number of holes. 
% We also see a single bar representing the 2nd Betti number, indicating the number of voids.