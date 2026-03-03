
function [m1,p1,t,Alloc]=fisherm(U,T,m,Y)
% input: Utility matrix, Tech matrix, money allocations, Availability of
% labour (for invertible T)
%Output: Next set of wages, prices, t: production, Allocation of goods
t=inv(T)*Y;
%t=[2;2];
%t=[0.7; 0.3376; 0.3];
[b,g]=size(U);
W=zeros(b+1,g+1,1);
L1=rand(b+1,g+1,1);
% to make the endowment matrix in required format
for i=1:1:b
    W(i,g+1,1) = m(1,i); 
end

for j=1:1:g
    W(b+1,j,1)= 1;
end
% Utility matrix for adplc in required format
U(:,g+1)=0;
U(b+1,:)=0;
U(b+1,g+1)=1;
for i=1:1:g
    U(:,i)=U(:,i)*t(i,1); %scaling the utility matrix 
end
[p,q]=adplc(U,W,L1); %Solves the market

s=sum(p);
for i=1:1:g
    p(1,i)=p(1,i)*sum(m)/(s*0.5*t(i,1)); % 
end
p1=p(1,1:g);
q=sum(m)*2*q./sum(q) ;% money allocation
Alloc=reshape(q,g+1,b+1)'; %goods allocation
Alloc(:,g+1)=[];
Alloc(b+1,:)=[];

for i= 1:1:g
    Alloc(:,i)=Alloc(:,i)/p(1,i);
end

m1=p1*inv(T);
%m1=p;
for i=1:1:b
    m1(1,i)=m1(1,i)*Y(i,1); %class wages
end

end
