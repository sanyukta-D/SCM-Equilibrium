function [p,m1,Ut]=plcequilibrium(U,T,m,Y,k,div)
for i=1:1:k
    [m1,p,~,~,Alloc]=plcmarket(U,T,m,Y,div);
    m=m1
    if m1(1,1)<0 || m1(1,2)<0 
        break
    end
    
end
[b,~,s]=size(U);
Ut=zeros(size(Y'));
for j=1:1:b
    Ut(1,j)=U(j,:,1)*Alloc(j,:,1)'+U(j,:,2)*Alloc(j,:,s)' ;
end
p,m1,Ut
end