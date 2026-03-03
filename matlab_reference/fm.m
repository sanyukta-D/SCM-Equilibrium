function [p0,m2,ActG,ActL,Alloc]=fm(U1,T,Y,p)
% New theory for markets. A general function which outputs next set of
% prices , wages, active goods and labour, and allocations given a set of
% initial prices and U,T,Y as usual. Deals with non invertible matrices as
% well.
U=U1;
[~,dim]=size(p);
Alloc=zeros(size(T));
fun= @(q) -p*q ; % the optimization function
q0=repmat(0.1,dim,1);
A=[T;-eye(dim)] ;
b1 = [Y;zeros(dim,1)];
Aeq=[];
beq=[];
lb=zeros(dim,1);
ub=[];
options=optimoptions('fmincon', 'Display','OFF');
q = fmincon(fun,q0,A,b1,Aeq,beq,lb,ub,[],options); % output:production
Diff=T*q-Y;
Diff=round(Diff,3);
ActL=find(abs(Diff)<0.01);
p*q;
Ind=find(abs(Diff)>0); % to find the dual variables i.e. wages
T(Ind,:)=[];
U1(Ind,:)=[];
q=round(q,2);
indices = find(abs(q)==0); % to find the total production
ActG= find(abs(q)>0);
q(indices) = [];  %reduced q vector
U1(:,indices)=[] ;  %reduced U1 matrix
T(:,indices)=[];
% [buyers,~]=size(U1);
% for i=1:1:buyers
%     U1(i,:,:)=U1(i,:,:)*Y(i,1);
% end
[b,~]=size(U1);
Y(Ind,:)=[];
p2=p;
p2(indices')=[];
T
m2=p2*inv(T) ;%computes new wages by using the reduced T and initial prices
for i=1:1:b
    m2(1,i)=m2(1,i)*Y(i,1); %class wages
end
m2=m2/sum(m2); 
if b==1
    m2; 
    p(1, ActG(1,1))=m2/q(1,1); % if only one class remains, prices are computed here
    Alloc(ActL,ActG)=m2;
else
    m2=round(m2,4);
    %U1,T,m2,Y;
    if b==2
        [~,p1,~,Alloc1]=forone(U1,T,m2,Y); % uses the special code for the 2D case
       
    else
        [~,p1,~,Alloc1]=fisherm(U1,T,m2,Y); % Fisher market using wages as computed before
    end
    Alloc(ActL,ActG)=Alloc1; % gives the total allocation including the discarded goods and labour
    p(ActG)=p1;
    
end
[y,~]=size(U1);
Ratio=zeros(1,y);
for j=ActL'
    a=[];
    for i = ActG
        
        a=[a U(j,i)/p(1,i)];
    end
    Ratio(1,j)= max(a);
end

for i=indices'  
    a=[];
    for j=ActL'
        
        a=[a U(j,i)/Ratio(1,j) ]; %Manipulating the price vector 
    end
    p(1,i)=max(a);
end
Alloc  ;
p0=p;
end
 %new prices
 