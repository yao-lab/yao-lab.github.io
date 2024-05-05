function lda_scores = LDA( data , labels )


    [ ~ , ncols ] = size(data);
    
    retained_components = ncols;
    %Preprocess to ensure that covariance matrix is well conditioned,
    %retained inertia is 98%
    if( cond(cov(data)) > 100 )
        [ ~ , pca_scores , ~ , ~ , explained ]   = pca(data);
        while( sum(explained(1:retained_components)) > 98  )
            retained_components =  retained_components - 1;    
        end 
        final_data     = pca_scores(:,[1:retained_components]);
    else
        final_data = data;
    end

    

    %Compute Covariance Matrix and center for each group 
    %Also compute inter class covariance
    num_groups = max(labels);
    class_covariance_matrixes   = zeros( retained_components , retained_components , num_groups );
    class_centers               = zeros( num_groups , retained_components );
    inter_class_covariance      = zeros( retained_components , retained_components );
    for k=1:num_groups
        kgroup = final_data(labels == k,:);
        [mc,~] = size(kgroup);
        [mt,~] = size(final_data);
        class_covariance_matrixes(:,:,k)=cov(kgroup)*(mc/mt);
        class_centers(k,:) = mean(kgroup);
        inter_class_covariance = inter_class_covariance + (mc/mt)*(class_centers(k,:)'*class_centers(k,:));
    end

    intra_class_covariance = zeros(retained_components,retained_components);
    for k=1:retained_components
        for j=1:retained_components
            intra_class_covariance(j,k) = sum(class_covariance_matrixes(j,k,:));
        end
    end


    %Verify the covariance matrix calcule (A has to be almos zero matrix)
    %A = cov(final_data) - (intra_class_covariance + inter_class_covariance)

    %Compute of Product Inverse 
    lda_matrix = (inter_class_covariance/intra_class_covariance)';
    [ eigenvectors ,eigenvalues ]  =   eig(lda_matrix);
    eigenvectors             = real(eigenvectors);
    eigenvalues              = real(diag(eigenvalues));

    lda_scores = [];   
    for k=1:num_groups-1
        lda_scores(:,k) = final_data*(eigenvectors(:,k));
    end



end