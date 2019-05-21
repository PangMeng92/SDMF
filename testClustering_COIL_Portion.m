%Clustering on COIL20
clear;

load('COIL20.mat');	

nClass = length(unique(gnd));
gndOld=gnd;

%Normalize each data vector to have L2-norm equal to 1  
fea = NormalizeFea(fea);

feaNew=[];
gndNew=[];
index=[];
IterNumber=1;
Count = 0;

%par.pca=[1, 2, 4,  6, 8, 10];
%par.pca=[2,3,4,6,8,10];
%par.pcb=[0.01, 0.02, 0.05, 0.1, 0.5, 1];

%% Setting the Number of Clusters
ClassNumber =20;

%% Initialize the selected cluster number
% ClassMatrix=zeros(IterNumber,ClassNumber);
% for i=1:IterNumber
%     ClassMatrix(i,:)=int16(randnorepeat(ClassNumber,nClass));
% end

disp(['Begin iteration, please wait! The iteration number is ', num2str(IterNumber)]);
%for t= 1:size(par.pcb,2)
for i=1: IterNumber

%ClassVector=ClassMatrix(i,:);
ClassVector=[1:1:20];


for j=1:ClassNumber
    gnd=gndOld;
    index=find(gnd==ClassVector(j));
    feaNew=[feaNew;fea(index,:)];
    gnd(index,:)=j;
    gndNew=[gndNew;gnd(index,:)];
end

TotalNum=size(feaNew,1);


%% SDMF   KNN

% fea: Rows of vectors of data points. Each row is x_i
Itrain = feaNew;
trainlabels = gndNew;


%% Begin Sparse concept Discriminant Matrix Factorization algorithm
%% Construct within weight matrix Ww (Supervised and Unsupervised)
options = [];

%options.NeighborMode = 'Supervised';   % Supervised
%options.gnd = trainlabels;

options.NeighborMode = 'KNN';   % Unsupervised

options.k = 3;   %% The number of Neiborhood points with same class label  
Ww = constructM_LLE(Itrain,options);


%% Construct between weight matrix Wb (Supervised)
%{
interK = 0;     % The number of Neiborhood points  
[nSmp,nFea] = size(Itrain);
Label = unique(trainlabels);
nLabel = length(Label);

Wb = ones(nSmp,nSmp);

for idx=1:nLabel
    classIdx = find(trainlabels==Label(idx));
    Wb(classIdx,classIdx) = 0;
end

if interK > 0
    Dw = EuDist2(Itrain,[],0);
    [dump idx] = sort(Dw,2); % sort each row
    for i=1:nSmp
        G(i,idx(i,1:interK+1)) = 1;
    end
    Wb = Wb.*G;
end

%}

%% Construct  between weight matrix Wb (Unsupervised)
%

%para.k = par.pca(t);
    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 3;
   % options.WeightMode = 'HeatKernel';
    options.WeightMode = 'Binary';
    Wb = constructZ(Itrain,options);
 %%
Dw = diag(sum(Ww,2));
Db = diag(sum(Wb,2));

Mw = (Dw-Ww)'*(Dw-Ww);
Mb = (Db-Wb);

    options = [];
    options.Mw = Mw;
    options.Mb = Mb;
    %% L2
    options.ReguAlpha = 0.02;  
    options.ReguType = 'Ridge';
    %% L1
    %options.ReguType = 'RidgeLasso';
    % options.LassoCardi = [10:10:90];
    % options.LassoCardi = [80];
     
    options.ReducedDim = ClassNumber; %(40)      

    [WProj] = SR_caller2(options, Itrain,Mw,Mb);

    %% The projected feature in subspace (SCC)
    options.ReguType = 'RidgeLasso';
    %  options.Cardi = ceil(ClassNumber/2);  %(35)
      options.Cardi = ceil(ClassNumber*4/5);
     options.LassoCardi = [40];
     options.ReguParaType='LARs';
    [FeaTrain] = SparseCodingwithBasis(WProj, Itrain', options);
   


FeaTrain = cell2mat(FeaTrain);
FeaTrain = FeaTrain';

%%Clustering in the subspace
rand('twister',5489);
label = litekmeans(FeaTrain',ClassNumber,'Replicates',20);
MIsdmf(i) = MutualInfo(trainlabels,label)
feaNew=[];
gndNew=[];
end
%end
%% Calculate the average accuracy (AC), mutual information (MI) and corresponding standard error
disp(['The iteration number is ', num2str(IterNumber)]);
MIsdmf_average=sum(MIsdmf)./IterNumber
MIsdmf_std=std(MIsdmf)
%}


