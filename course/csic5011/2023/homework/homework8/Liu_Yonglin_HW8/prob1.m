load_javaplex
import edu.stanford.math.plex4.*;
api.Plex4.createExplicitSimplexStream()
ans = edu.stanford.math.plex4.streams.impl.ExplicitSimplexStream@16966ef

%%% a
% Create an explicit simplex stream
stream = api.Plex4.createExplicitSimplexStream();

% Add vertices
stream.addVertex(0, 0);
stream.addVertex(1, 1);
stream.addVertex(2, 2);
stream.addVertex(3, 3);

% Add edges
stream.addElement([0, 2], 4);
stream.addElement([0, 1], 5);
stream.addElement([2, 3], 6);
stream.addElement([1, 3], 7);
stream.addElement([1, 2], 8);

% Add triangles
stream.addElement([1, 2, 3], 9);
stream.addElement([0, 1, 2], 10);

% Finalize the stream
stream.finalizeStream();

% Check the number of simplices in the filtration (stream)
num_simplices = stream.getSize();

%This code should output the number of simplices in the filtration (stream) as 11, which matches the number of simplices in the example provided.

%%% b
%% Construct the filtration (stream) of the simplicial complex
stream = api.Plex4.createExplicitSimplexStream();
stream.addVertex(0,0);
stream.addVertex(1,1);
stream.addVertex(2,2);
stream.addVertex(3,3);
stream.addElement([0,2],4);
stream.addElement([0,1],5);
stream.addElement([2,3],6);
stream.addElement([1,3],7);
stream.addElement([1,2],8);
stream.addElement([1,2,3],9);
stream.addElement([0,1,2],10);
stream.finalizeStream();

%% Compute the persistent homology and plot the barcode
% Compute the Z/2Z persistence homology of dimension less than 3:
persistence = api.Plex4.getModularSimplicialAlgorithm(3, 2);
intervals = persistence.computeIntervals(stream);

options.filename = 'Persistent-Betti-Numbers';
options.max_filtration_value = 11;

% Plot the barcode of persistent Betti numbers:
plot_barcodes(intervals, options);
