%x=0:.1:1;
%y=0:.1:1;
x= [8,8,8,8,8,8,8,8,8,8,8,8,8,8,16,16,16,16,16,16,16,16,16,16,16,16,16,16,24,24,24,24,24,24,24,24,24,24,24,24,24,24];
y=[10,20,40,80,100,130,150,200,220,230,250,280,300,350,10,20,40,80,100,130,150,200,220,230,250,280,300,350,10,20,40,80,100,130,150,200,220,230,250,280,300,350];


z=[x;y];
%c=[1 2 3 4 5 6 7 8 9 10 11; 1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11;1 2 3 4 5 6 7 8 9 10 11];
%11 output=11 line
%net2 = newff(minmax(z),[3 11],{'logsig' 'purelin'},'traindiffevol1'); 
%v = [1 2 3 4 5 6 7 8 9 10 11];
c=[1 1 1 3 5 5 6 9 10 10 10 12 13 16 1 1 2 4 3 4 7 10 9 11 11 12 13 16 0 0 0 1 1 2 2 3 3 3 3 4 5 5 ];
net2 = newff(minmax(z),[3 1],{'logsig' 'purelin'},'traindiffevol1'); 
%net2 = newff(minmax(z),[3 1],{'purelin' 'purelin'},'traindiffevol1'); 

net2.trainParam.epochs = 15000; % these are smaller just for the demo_number of iteration
%msereg is a network performance function. It measures network performance as the weight sum of two factors: the mean squared error and the mean squared weight and bias values.
net2.performFcn = 'msereg';
net2.performParam.ratio = 0.995;
net2.trainParam.show = 25;
net2.trainParam.popsizew = 1;
%net2.trainParam.initw = 100;
net2.trainParam.cr = 0.5; %Crossover probability
net2.trainParam.f = 0.8;% DE stepsize

tic
net2 = train(net2,z,c);
toc
test2 = sim(net2, z);
%weight
weight=net2.IW{1,1};
%bias
bias=net2.b{1};
view(net2)

%net1 = newff(minmax(z),[3 1],{'logsig' 'purelin'},'trainscg'); 
%net1.trainParam.epochs = 3000; % these are smaller just for the demo
%net1.performFcn = 'msereg';
%net1.performParam.ratio = 0.995;
%net1.trainParam.show = 25;
%tic
%net1 = train(net1,z,c);
%toc
%test1 = sim(net1, z);


%[e,f] = meshgrid(x,y);
%mesh(e,f,test2)
%hold;
%mesh(e,f,c);