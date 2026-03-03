[p,q1,L1,Ratio,Alloc] = plc(rand(2,2,2),[2;2],rand(1,2),rand(2,2,2))


function [p,q1,L1,Ratio,Alloc]=plc(U,t,m,L1)
%U:Utility matrix , t: production as an input, m: money given , 
%Give input in fractions for the L1 matrix
% outputs prices, money & goods allocations, L1, Bang for the buck
[b,g,k]=size(L1);
W=zeros(b,g);
% div=2;
% prod=[1/div;1/div];
% L1=[prod'; prod'];
% L1(:,:,2)=[t'; t']
L2=zeros(size(L1));
%L2(:,:,2)=L1(:,:,2);
for j=1:1:b
    for i=1:1:g
        
        L2(j,i,1)= L1(j,i,1)/L1(j,i,2); %scaling L1 matrix
    end
end

for i=1:1:b
    
    for j=1:1:g
        W(i,j)=m(1,i)/sum(m);  % forming W   
    end
end

for i=1:1:g
    U(:,i,:)=U(:,i,:)*t(i,1); %scaling the utility matrix 
end
%U, W,L2
%W=rand(2,2);
[p,q]=adplc(U,W,L2); %Solves the market

s=sum(p);
for i=1:1:g
    p(1,i)=p(1,i)*sum(m)/(s*t(i,1)); % 
end

q=sum(m)*q./sum(q);

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
end
% U=rand(3,3,2);
% t=[2;2]
% m = rand(1,3);
% plc(U,t,m)
% T=[0.26 0 ; 0.25 1 ];
% Y=[2;4];
% U=[1 0.5;1.2 1];
%U(:,:,2)=[1 0.3;1.1 1];
% m=[6 14];
% plc(U,T,W,Y)
