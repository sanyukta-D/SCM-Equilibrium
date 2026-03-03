U1=[1 1 ; 1 1 ] ; %0.5 0.8
T=[0.2501 0 ; 0.25 1 ];
Y=[2;4];
p=[2 3];

% interesting case:  p1=[1 1.2 0.555]; Wages for class 1 initially zero,
% then become positive
%[PI,WI,ActG,ActL,Alloc,UT,WF]=iterations(U1,T,Y,p,5);
%[w,u]=Case2(U1,T,Y,p,5);

Datasave1=[];
R=[  0.5 0.75 1.001 1.5 1.7 ];
for alpha=R
    Y1=[2;4];
    datasave1=[];
    B=[];
    
   % R=[R ratio ratio ];
    for beta=0.21:0.2:1.9
        
        
        U=[1 alpha; beta 1];
        [wf,u,Alloc]=Case2(U,T,Y1,p,5);
        u1=zeros(size(Y'));
        for j=1:1:b
            u1(1,j)=U1(j,:)*Alloc(j,:)';
        end
        datasave1=[datasave1; u1(1,2)  wf ];
        B=[B;beta];
        
        
    end
    Datasave1=[Datasave1 datasave1];
end
Datasave1=[B Datasave1];

T1=array2table(Datasave1,'VariableNames',{'alpha' 'uDot5' 'WgFDot5' 'uDot75' 'WgDot75' 'u1' 'WgF1' 'u1Dot5' 'WgF1Dot5' 'u1Dot7' 'WgF1Dot7' })
%T, U, T1
%['beta' 'u1' 'u2' ]= Datasave1[1,:]
% Datasave1=[B;Datasave1];
% Datasave1=[R Datasave1];
% T1=array2table(Datasave1);
%T2 = mergevars(T1,[1 2],[3 4],[5 6],[7,8])
%disp(Datasave1);
    
        