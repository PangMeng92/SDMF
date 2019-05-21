function [eigvector, Y, elapse] = SR_caller2(options, data,Mw,Mb)
% SR: Spectral Regression
%
%       [eigvector, elapse] = SR_caller(options, data)
% 
%             Input:
%               data    - data matrix. Each row vector of data is a
%                         sample vector. 
%               options - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                        gnd   -  Colunm vector of the label information
%                                 for each data point. 
%                                 If gnd is provided, SR will give the
%                                 SRDA solution [See Ref 7,8]
%
%                        W     -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If gnd is not provided, W is required and
%                                 SR will give the RLPI (RLPP) solution
%                                 [See Ref 1] 
%
%                ReducedDim    -  The number of dimensions. If gnd is
%                                 provided, ReducedDim=c-1 where c is the number
%                                 of classes. Default ReducedDim = 30.
%
%                       Please see SR.m for other options. 
%
%
%             Output:
%               eigvector - Each column is an embedding function, for a new
%                           sample vector (row vector) x,  y = x*eigvector
%                           will be the embedding result of x.
%
%                           If 'Lasso' or 'RidgeLasso' regularization is
%                           used, the output eigvector will be a cell,
%                           please see 'lars.m' for the result format.
%
%               elapse    - Time spent on different steps 
%                           
% 
%===================================================================
%    Examples:
%           
%    (Supervised case with L2-norm (ridge) regularizer, SR-LDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguAlpha = 0.01;
%       options.ReguType = 'Ridge';
%       [eigvector] = SR_caller(options, fea);
%
%       [nSmp,nFea] = size(fea);
%       if size(eigvector,1) == nFea + 1
%           Y = [fea ones(nSmp,1)]*eigvector;
%       else
%           Y = fea*eigvector;
%       end
%-------------------------------------------------------------------
%    (Unsupervised case with L2-norm (ridge) regularizer, SR-LPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReguAlpha = 0.01;
%       options.ReguType = 'Ridge';
%       options.ReducedDim = 10;
%       [eigvector] = SR_caller(options, fea);
%
%       [nSmp,nFea] = size(fea);
%       if size(eigvector,1) == nFea + 1
%           Y = [fea ones(nSmp,1)]*eigvector;
%       else
%           Y = fea*eigvector;
%       end
%-------------------------------------------------------------------
%    (Supervised case with L1-norm (lasso) regularizer, SR-SpaseLDA)
%
%       fea = rand(50,70);
%       gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
%       options = [];
%       options.gnd = gnd;
%       options.ReguAlpha = 0.001;
%       options.ReguType = 'RidgeLasso';
%       options.LassoCardi = [10:5:60];
%       [eigvectorAll] = SR_caller(options, fea);
%       
%       [nSmp,nFea] = size(fea);
%       
%       for i = 1:length(options.LassoCardi)
%           eigvector = eigvectorAll{i};  %projective functions with cardinality options.LassoCardi(i)
%           
%           if size(eigvector,1) == nFea + 1
%               Y = [fea ones(nSmp,1)]*eigvector;
%           else
%               Y = fea*eigvector;
%           end
%       end
%
%-------------------------------------------------------------------
%    (Unsupervised case with L1-norm (lasso) regularizer, SR-SpaseLPP)
%
%       fea = rand(50,70);
%       options = [];
%       options.Metric = 'Euclidean';
%       options.NeighborMode = 'KNN';
%       options.k = 5;
%       options.WeightMode = 'HeatKernel';
%       options.t = 5;
%       W = constructW(fea,options);
%
%       options = [];
%       options.W = W;
%       options.ReguAlpha = 0.001;
%       options.ReguType = 'RidgeLasso';
%       options.LassoCardi = [10:5:60];
%       options.ReducedDim = 10;
%       [eigvector] = SR_caller(options, fea);
%
%       [nSmp,nFea] = size(fea);
%
%       for i = 1:length(options.LassoCardi)
%           eigvector = eigvectorAll{i};  %projective functions with cardinality options.LassoCardi(i)
%           
%           if size(eigvector,1) == nFea + 1
%               Y = [fea ones(nSmp,1)]*eigvector;
%           else
%               Y = fea*eigvector;
%           end
%       end
%
%===================================================================
%
%Reference:
%
%   1. Deng Cai, Xiaofei He, Jiawei Han, "Semi-Supervised Regression using
%   Spectral Techniques", Department of Computer Science
%   Technical Report No. 2749, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2749), July 2006.  
%
%   2. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for
%   Dimensionality Reduction", Department of Computer Science
%   Technical Report No. 2856, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2856), May 2007.  
%
%   3. Deng Cai, Xiaofei He, Jiawei Han, "SRDA: An Efficient Algorithm for
%   Large Scale Discriminant Analysis", Department of Computer Science
%   Technical Report No. 2857, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2857), May 2007.  
%
%   4. Deng Cai, Xiaofei He, and Jiawei Han. "Isometric Projection", Proc.
%   22nd Conference on Artifical Intelligence (AAAI'07), Vancouver, Canada,
%   July 2007.  
%
%   5. Deng Cai, Xiaofei He, Jiawei Han, "Efficient Kernel Discriminant
%   Analysis via Spectral Regression", Department of Computer Science
%   Technical Report No. 2888, University of Illinois at Urbana-Champaign
%   (UIUCDCS-R-2007-2888), August 2007.  
% 
%   6. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Subspace Learning Framework for Content-Based Image Retrieval", ACM
%   Multimedia 2007, Augsburg, Germany, Sep. 2007.
%
%   7. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression for Efficient
%   Regularized Subspace Learning", IEEE International Conference on
%   Computer Vision (ICCV), Rio de Janeiro, Brazil, Oct. 2007. 
%
%   8. Deng Cai, Xiaofei He, Jiawei Han, "Spectral Regression: A Unified
%   Approach for Sparse Subspace Learning", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   9. Deng Cai, Xiaofei He, Jiawei Han, "Efficient Kernel Discriminant
%   Analysis via Spectral Regression", Proc. 2007 Int. Conf. on Data
%   Mining (ICDM'07), Omaha, NE, Oct. 2007. 
%
%   10. Deng Cai, Xiaofei He, Wei Vivian Zhang, Jiawei Han, "Regularized
%   Locality Preserving Indexing via Spectral Regression", Proc. 2007 ACM
%   Int. Conf. on Information and Knowledge Management (CIKM'07), Lisboa,
%   Portugal, Nov. 2007.
%
%
%   version 2.0 --Aug/2007 
%   version 1.0 --May/2006 
%
%   Written by Deng Cai (dengcai2 AT cs.uiuc.edu)
%

ReducedDim = 30;
if isfield(options,'ReducedDim')
    ReducedDim = options.ReducedDim;
end



WPrime = Mw ;
DPrime = Mb ;


DPrime = max(DPrime,DPrime');
WPrime = max(WPrime,WPrime');
%ReducedDim = ReducedDim+1;


[nSmp,nFea] = size(data);


if ReducedDim > nSmp
    ReducedDim = nSmp; 
end


        [Y, eigvalue] = eig(DPrime,WPrime);
        eigvalue = diag(eigvalue);

        [junk, index] = sort(-eigvalue);
        eigvalue = eigvalue(index);
        Y = Y(:,index);
        
        if ReducedDim < length(eigvalue)
            Y = Y(:, 1:ReducedDim);
            eigvalue = eigvalue(1:ReducedDim);
        end
  
    
    eigIdx = find(abs(eigvalue) < 1e-6);
    eigvalue (eigIdx) = [];
    Y (:,eigIdx) = [];


[eigvector, elapse.timeReg] = SR(options, Y, data);

