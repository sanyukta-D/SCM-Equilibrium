function [m1,p1,q1,a1]=forone(U,T,m,Y)
%a special function for the 2 matrix case which deals with degenerate cases
%as well
i=m(1,1);
tm=sum(m);
try
    if rem(i,tm-i)==0 || rem(tm-i,i)==0
        n=[i-0.001 tm+0.001-i];
        p=[i+0.001 tm-i-0.001];
        [m0,p0,q0,a0]=fisherm(U,T,n,Y);
        [m2,p2,q2,a2]=fisherm(U,T,p,Y);
        m1=(m2-m0)/2+m0;
        p1=(p2-p0)/2+p0;
        q1=(q2-q0)/2+q0;
        a1=(a2-a0)/2+a0;
        return
    else
        [m1,p1,q1,a1]=fisherm(U,T,m,Y);
        
    end
catch
end
% if rem(i,20-i)==0 || rem(20-i,i)==0;
%     n=[i-0.001 20.001-i];
%     p=[i+0.001 20-i-0.001];
%     m0=fisherm(U,T,n,Y,b,g);
%     m2=fisherm(U,T,p,Y,b,g);
%     m1=(m2-m0)/2+m0;
% else;
%     m1=fisherm(U,T,m,Y,b,g);
% end
% end