function v=randnorepeat(m,N)
%  Generating nonredundant m  1-N integers 
p=randperm(N);
v=p(1:m);