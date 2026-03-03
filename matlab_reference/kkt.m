% First run 'testcases' to see the output of prices, prod, wages. 
% Then run [m,p,q]=kkt(U2,T2) (in the command window) to find m,p,q . Print 'c' to see how inequalities
% perform. U2 and T2 are as specified in 'testcases'.


function [m,p,q,y,z,veci,vec,con]=kkt(U,T)
% m p q y - wages, prices, prod, & y from ineq UijYi< Pj
% All these are variables
v=[0.01453  0.06478 0.0883 0.25294 0.47972  0.206 0.263 0.053 0.368 0.16863 0.43611  0.34265];
v=[0.086693 0.3865 0.53681 1.5091 2.8622 1.2266 0.26316 0.052632 0.36842 1.0061 2.6019 2.0444 0.086622 0 0 0.31027 0.076421 0 0 0.075276 0.45028];
%v=[0.22709  0.11902  0.65389 1.316  3.1032 1.3299  0.263 0.053 0.368 0.87735 2.269 2.2166 0.22714 0 0 0.11897 0 0 0 0.16447 0.4894];
%v=rand(1,12);
%v=[0.38462  0.61538  0 2.233  6.14706   0.77 0.33 0 0.33 1.4887 3.85 1];
% This is a starting point- constructed from our solution to the same
% system. v=[m p q y] Any other input can also be used, like - repmat(0.1,len,1)

global m p q y z len
[b,g]=size(U);
fun=@(x) myfun(x); % the maximization function- at the bottom
len=2*b+2*g+b*g;
%gradf = jacobian(fun,v)
nonlcon = @pnq; % nonlinear constraints- function given below
options = optimoptions('fmincon','Display','iter','Algorithm','active-set');
vector = fmincon(fun,v',[],[],[],[],zeros(len,1),[],nonlcon,options) ; %vector=[m p q y]
% x = fmincon(fun,x-init,A,b,Aeq,beq,LowerBound,UB,nonlcon,options)
veci=myfun(v');
vec=myfun(vector);
[con,~]=pnq(vector);
%con=round(con,3);
m=vector(1:b); % extracting the variables from 'vector'
p=vector(1+b:b+g);
q=vector(b+g+1:b+g+g);
y=vector(b+2*g+1:2*b+2*g);
z=vector(2*b+2*g+1:len);
z=reshape(z,3,3)';
function [c,ceq]=pnq(x)
    c=zeros(b*g+b+g,1);
    ceq=zeros(b+g+1);
   % c(b*g+b+g+1,1)=-p'*q+sum(m);  %equation (16)
   % c(b*g+b+g+1,1)=p'*q-1;
    %c(b*g+b+g+1,1)=-p'*q+m'*T*q;
    matrix=zeros(b,g);
    for i=1:1:b
        c(b*g+i)=T(i,:)*q-1;  %equation (19)
        ceq(i)=sum(z(i,:))-m(i);
        for j=1:1:g
            matrix(i,j)=U(i,j)*y(i)-p(j); %equation (15) 
        end
    end
    c(1:b*g)=reshape(matrix,1,b*g); %rearranging the matrix into a vector
    for j=1:1:g
        c(b*g+b+j,1)=-p(j)*q(j)+m'*T(:,j)*q(j);  %% ****
        ceq(b+j)=sum(z(:,j))-p(j)*q(j);
        %c(b*g+b+j)=m'*T(:,j)-p(j); %equation (20)
    end
     % print c to track the iterations
    
    c=round(c,2);
   % return;
    ceq(b+g+1)=sum(m)-1;
    ceq=round(ceq,2);
    %ceq=[];
end

function f=myfun(x)
sum=0;
m=x(1:b); % extracting the variables from 'vector'
p=x(1+b:b+g);
q=x(b+g+1:b+g+g);
y=x(b+2*g+1:2*b+2*g);
z=x(2*b+2*g+1:len);
z=reshape(z,g,b)';
q=round(q,3);
p=round(p,3);
m=round(m,3);
y=round(y,3);
z=round(z,3);
for i=1:1:length(m)
   %sum=sum+(m(i)*(T(i,:)*q)+(p(i)-m'*T(:,i))*q(i))*log(y(i));% maximizing function
    
   %sum=sum+(m(i)*log(y(i))); %maximizing function - 2 [new]  
   sum=sum+(m(i)*(T(i,:)*q))*log(y(i));
    % good 1 surplus goes to buyer 1 & so on
end
f=-sum ;%maximize
%f=m'*T*q-p'*q;
end
%0.01453  0.06478 0.0883
end
   

