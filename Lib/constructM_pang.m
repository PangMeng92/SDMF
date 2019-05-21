function W = constructM_pang(fea,options)
%	Usage:
%	W = constructM_pang(fea,options)
% 
%	fea: Rows of vectors of data points. Each row is x_i
%   options: Struct value in Matlab. The fields in options that can be set:
%                  
%           NeighborMode -  Indicates how to construct the graph. Choices
%                           are: 
%                'KNN'            -  Put an edge between two nodes if and
%                                    only if they are among the k nearst
%                                    neighbors of each other. You are
%                                    required to provide the parameter k in
%                                    the options. [Default One]
%               'Supervised'      -  Two variations:
%                                       1. k=0, Put an edge between two nodes 
%                                          if and only if they belong to same class. 
%                                       2. k>0, The distance between two nodes 
%                                          in the same class will be smaller than 
%                                          two nodes have diff. labels 
%                                    You are required to provide the label
%                                    information gnd in the options.
%                                              
%            k         -   The number of neighbors.
%            gnd       -   The parameter needed under 'Supervised'
%                          NeighborMode.  Column vector of the label
%                          information for each data point.
%

[nSmp, nFea] = size(fea);

switch lower(options.NeighborMode)
    case {lower('KNN')}
    	
		aa = sum(fea.*fea,2);
		ab = fea*fea';
	    Distance = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
	    Distance = max(Distance,Distance');
	    Distance = Distance - diag(diag(Distance));
	    
        if options.k <= 0
            error('k must be greater than 0!');
        end
    case {lower('Supervised')}
        if options.k > 0
        	
		    aa = sum(fea.*fea,2);
		    ab = fea*fea';
	        Distance = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
	        Distance = max(Distance,Distance');
	        Distance = Distance - diag(diag(Distance));
    
            WLDA = ones(nSmp,nSmp);
            Label = unique(options.gnd);
            nLabel = length(Label);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                WLDA(classIdx,classIdx) = 0;
            end
            Distance = Distance+WLDA*(max(max(Distance))+10);
        end
        
        case {lower('Supervised2')}
        if options.k > 0
        	
		    aa = sum(fea.*fea,2);
		    ab = fea*fea';
	        Distance = repmat(aa, 1, nSmp) + repmat(aa', nSmp, 1) - 2*ab;
	        Distance = max(Distance,Distance');
	        Distance = Distance - diag(diag(Distance));
    
            WLDA = zeros(nSmp,nSmp);
            Label = unique(options.gnd);
            nLabel = length(Label);
            for idx=1:nLabel
                classIdx = find(options.gnd==Label(idx));
                WLDA(classIdx,classIdx) = 1;
            end
            Distance = Distance+WLDA*(max(max(Distance))+10);
        end
        
    otherwise
        error('NeighborMode does not exist!');
end


if options.k == 0
    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        idx = find(options.gnd~=options.gnd(ii));
        idx(find(idx==ii)) = [];
        z = fea(idx,:)-repmat(fea(ii,:),length(idx),1); % shift ith pt to origin
        C = z*z';                                        % local covariance
        tW = C\ones(length(idx),1);                           % solve Cw=1
        tW = tW/sum(tW);                  % enforce sum(w)=1
        W(idx,ii) = tW;
    end
else %if options.k > 0
    [~,index] = sort(Distance,2);
    neighborhood = index(:,2:(1+options.k));

    %W = zeros(options.k,nSmp);
    W = zeros(nSmp,nSmp);
    for ii=1:nSmp
        z = fea(neighborhood(ii,:),:)-repmat(fea(ii,:),options.k,1); % shift ith pt to origin
        C = z*z';                                        % local covariance
%         W(:,ii) = C\ones(options.k,1);                           % solve Cw=1
%         W(:,ii) = W(:,ii)/sum(W(:,ii));                  % enforce sum(w)=1
        W(neighborhood(ii,:),ii) = C\ones(options.k,1);                           % solve Cw=1
        W(neighborhood(ii,:),ii) = W(neighborhood(ii,:),ii)/sum(W(neighborhood(ii,:),ii));                  % enforce sum(w)=1
    end
end

%save W 