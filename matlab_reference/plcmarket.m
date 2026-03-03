function [m1,p,t,Alloc]=plcmarket(U,T,m,Y,L1)
%U:Utility matrix , t: production as an input, m: money given , Div:
%division factor for the L1 matrix
% outputs prices, money & goods allocations, L1, Bang for the buck
t=inv(T)*Y;
[b,g,k]=size(L1);
W=zeros(b,g);

% div=2
% prod=[1/div;1/div];
% L1=[prod'; prod'];
% L1(:,:,k)=[t'; t'];
%L1=rand(b,g,k);
L2=zeros(b,g);
L2(:,:,2)=L1(:,:,2);
for j=1:1:b
    for i=1:1:g
        
        L2(j,i,1)= L1(j,i,1)/L1(j,i,2); %scaling L1 matrix
    end
end
for i=1:1:b
    for j=1:1:g %forming W 
        W(i,j)=m(1,i)/sum(m);       
    end
end

for i=1:1:g
    U(:,i,:)=U(:,i,:)*t(i,1); %scaling the utility matrix 
end
%U, W,L1
%W=rand(2,2);
[p,q]=adplc(U,W,L2);%Solves the market

s=sum(p);
for i=1:1:g
    p(1,i)=p(1,i)*sum(m)/(s*t(i,1)); % 
end
q=sum(m)*q./sum(q) ;
if k==2
    q0=reshape(q',b,g,k);% money allocation
    q1=permute(q0,[3,2,1]);
elseif k==1
    q1=reshape(q,b,g)';
end
Ratio=[];
for s=1:1:k
    for j=1:1:g
        
        for i =1:1:b
            
            Ratio=[Ratio U(i,j,s)/p(1,j)];
        end
        
    end
end
Ratio=reshape(Ratio,b,g,k);
Alloc=zeros(size(q1));   
for j=1:1:g
    Alloc(:,j,:)=q1(:,j,:)/p(1,j);
end
q=q1;
m1=p*inv(T);
for i=1:1:b
    m1(1,i)=m1(1,i)*Y(i,1); %class wages
end
end
