 function [net, tr, Ac, El, v5,v6,v7,v8] = traindiffevol1(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11,v12)
%function [net, tr, Ac, El] = traindiffevol1(net,Pd,Tl,Ai,Q,TS,VV,TV)
 %   function [net,tr,Ac,El] = trainscg(net,Pd,Tl,Ai,Q,TS,VV,TV)
 %function [net, tr, Ac, El, v5,v6,v7,v8] =
 %traindiffevol(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11,v12)
    %function [ output_args ] = Untitled( input_args )

%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
%function [net, tr, Ac, El, v5,v6,v7,v8] = traindiffevol(net,Pd,Tl,Ai,Q,TS,VV,TV,v9,v10,v11,v12);
 
%TRAINDIFFEVOL differetial evolution training
%
%	Syntax
%	
%	  [net, tr] = traindiffevol(net,Pd,Tl,Ai,Q,TS,VV);
%         info = traindiffevol(code)
%
%	Description
%
%	  TRAINDIFFEVOL is a network training function that updates weight and
%	  bias values according to differential evolution optimization.
%
%	  TRAINDIFFEVOL(NET,Pd,Tl,Ai,Q,TS,VV) takes these inputs,
%	    NET - Neural network.
%	    Pd  - Delayed input vectors.
%	    Tl  - Layer target vectors.
%	    Ai  - Initial input delay conditions.
%	    Q   - Batch size.
%	    TS  - Time steps.
%	    VV  - Either empty matrix [] or structure of validation vectors.
%	  and returns,
%	    NET - Trained network.
%	    TR  - Training record of various values over each epoch:
%	          TR.epoch - Epoch number.
%	          TR.perf  - Training performance.
%	          TR.vperf - Validation performance.
%	          TR.tperf - Test performance.
%	          TR.mu    - Adaptive mu value.
%
%	  Training occurs according to the TRAINDIFFEVOL's training parameters
%	  shown here with their default values:
%	    net.trainParam.epochs      10     Maximum number of epochs to train
%	    net.trainParam.goal         0     Performance goal
%           net.trainParam.popsizew     2     Population size as times of weights 
%           net.trainParam.popsize      0     Population size (can also be set explicitely popsizew = 0)
%           net.trainParam.maxvalue     inf   Maximum value of any population value
%	    net.trainParam.cr           0.5   Crossover probability
%           net.trainParam.f            0.9   DE-stepsize F from interval [0, 2]
%           net.trainParam.strategy     7     1-9 see options later
%	    net.trainParam.show         1     Epochs between showing progress
%	    net.trainParam.time       inf     Maximum time to train in seconds
%           net.trainParam.initw        0     Init vals (plus-minus) for weights (0 uses matlab initialisation)
%
%	  Strategy (as pairs)
%	    1 --> DE/best/1/exp           6 --> DE/best/1/bin
%           2 --> DE/rand/1/exp           7 --> DE/rand/1/bin
%           3 --> DE/rand-to-best/1/exp   8 --> DE/rand-to-best/1/bin
%           4 --> DE/best/2/exp           9 --> DE/best/2/bin
%           5 --> DE/rand/2/exp           else  DE/rand/2/bin
%          Experiments suggest that /bin likes to have a slightly larger CR than /exp.
%
%	  Dimensions for these variables are:
%	    Pd - NoxNixTS cell array, each element P{i,j,ts} is a DijxQ matrix.
%	    Tl - NlxTS cell array, each element P{i,ts} is a VixQ matrix.
%		Ai - NlxLD cell array, each element Ai{i,k} is an SixQ matrix.
%	  Where
%	    Ni = net.numInputs
%		Nl = net.numLayers
%		LD = net.numLayerDelays
%	    Ri = net.inputs{i}.size
%	    Si = net.layers{i}.size
%	    Vi = net.targets{i}.size
%	    Dij = Ri * length(net.inputWeights{i,j}.delays)
%
%	  If VV is not [], it must be a structure of validation vectors,
%	    VV.PD - Validation delayed inputs.
%	    VV.Tl - Validation layer targets.
%	    VV.Ai - Validation initial input conditions.
%	    VV.Q  - Validation batch size.
%	    VV.TS - Validation time steps.
%	  which is used to stop training early if the network performance
%	  on the validation vectors fails to improve or remains the same
%	  for MAX_FAIL epochs in a row.
%
%	  TRAINLM(CODE) return useful information for each CODE string:
%	    'pnames'    - Names of training parameters.
%	    'pdefaults' - Default training parameters.
%
%	Network Use
%
%	  You can create a standard network that uses TRAINDIFFEVOL with
%	  NEWFF, NEWCF, or NEWELM.
%
%	  To prepare a custom network to be trained with TRAINDIFFEVOL:
%	  1) Set NET.trainFcn to 'traindiffevol'.
%	     This will set NET.trainParam to TRAINDIFFEVOL's default parameters.
%	  2) Set NET.trainParam properties to desired values.
%
%	  In either case, calling TRAIN with the resulting network will
%	  train the network with TRAINDIFFEVOL.
%
%	  See NEWFF, NEWCF, and NEWELM for examples.
%
%	Algorithm
%
%	TRAINDIFFEVOL can be used as any other network training algorithms,
%       except that there is no more limitation set to transfer functions
%       (derivatives not needed). In particular, the algorithm seems to
%       work well only if the given values for variable minimums and maximums
%       ([XVmin,XVmax]) covers the region where the global minimum is expected.
%       DE is also somewhat sensitive to the choice of the stepsize F. A good
%       initial guess is to choose F from interval [0.5, 1], e.g. 0.8. CR, the
%       crossover probability constant from interval [0, 1] helps to maintain
%       the diversity of the population and is rather uncritical. The number of
%       population members NP is also not very critical. A good initial guess is
%       10*number_of_weights. Depending on the difficulty of the problem NP can be
%       lower than 10*number_of_weights or must be higher than 10*number_of:weights
%       to achieve convergence. If the parameters are correlated, high values of
%       CR work better. The reverse is true for no correlation.
%
%	TRAINDIFFEVOL is a vectorized variant of DE which, however, has a
%	propertiy which differs from the original version of DE:
%	1) The random selection of vectors is performed by shuffling the
%	population array. Hence a certain vector can't be chosen twice
%	in the same term of the perturbation expression.
%
%	  Training stops when any of these conditions occurs:
%	  1) The maximum number of EPOCHS (repetitions) is reached.
%	  2) The maximum amount of TIME has been exceeded.
%	  3) Performance has been minimized to the GOAL.
%	  4) Validation performance has increased more than MAX_FAIL times
%	     since the last time it decreased (when using validation).
%
%	See also NEWFF, NEWCF, TRAINLM, TRAINGD, TRAINGDM, TRAINGDA, TRAINGDX.

% TRAINDIFFEVOL $Revision: 1.10 $
% Copyright (C) 2000 Joni Kamarainen
% Lappeenranta University of Technology
% Box 20
% FIN-53851 LAPPEENRANTA
% Finland
% E-mail: joni.kamarainen@lut.fi
% WWW:  http://www.lut.fi/~jkamarai
%
% This program is free software; you can redistribute it and/or
% modify it under the terms of the GNU General Public License
% as published by the Free Software Foundation; either version 2
% of the License, or any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program; if not, write to the Free Software
% Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
% http://www.gnu.org/copyleft/gpl.html
%
% All author(s):
%  Jouni Lampinen  / Lappeenranta University of Technology
%  Miika Lindfors  / Lappeenranta University of Technology
%  Joni Kamarainen / Lappeenranta University of Technology
%  http://www.it.lut.fi/project/nngenetic
%
% Acknowledgements:
%  Authors would like to thank Mr. Rainer Storn for his
%  original Matlab source which can be downloaded from DE
%  homepage at:
%    www.icsi.berkeley.edu/~storn/code.html

%
% 1. INITIALISATION CALL
%
if ischar(net)
  switch (net)
  case 'pnames',
   net = fieldnames(traindiffevol1('pdefaults'));
   %case 'pdefaults',
   otherwise,
       
    trainParam.epochs = 3000;
    trainParam.goal = 0;
    trainParam.popsizew = 1;
    trainParam.popsize = 0;
    trainParam.maxvalue = inf;
    trainParam.cr = 0.3; % 0-1
    trainParam.f = 0.4; % 0-2
    trainParam.strategy = 6;
    trainParam.show = 100;
    trainParam.time = inf;
    trainParam.initw = 0;
    trainParam.mustinit = 1; % true
    trainParam.pop = []; % population
    trainParam.version=5;
    net = trainParam;
    
    
  % otherwise,
   % error('Unrecognized code.')
  end
  return
end



%
% 2. CALCULATION CALL - validation and test not supported yet
%

% set optimization constants
this = 'TRAINDIFFEVOL1';
epochs = net.trainParam.epochs;
goal = net.trainParam.goal;
NP = net.trainParam.popsize;
CR = net.trainParam.cr;
F = net.trainParam.f;
strategy = net.trainParam.strategy;
show = net.trainParam.show;
time = net.trainParam.time;


% Initialize
stop = ''; % text for stopping
startTime = clock;
X = getx(net); % initial values of weights
D = size(X,1); % Dimension of the population unit
if (net.trainParam.popsizew ~= 0)
  NP = net.trainParam.popsizew*D;
end;


% initialize population USES DEFAULT INITIALISATION OF MATLAB(min and max info should be used !!!)
if (net.trainParam.mustinit)
  net.trainParam.mustinit = 0; % not initialized next time
  initnet = net; % copy structure
  net.trainParam.pop = zeros(NP,D); % population of NP members
  for i=1:NP
    if (net.trainParam.initw == 0) % matlab default initialisation
      initnet = initlay(initnet); % matlab initialisation
      initX = getx(initnet); % initial values of weight
      net.trainParam.pop(i,:) = initX';
    else % user defined
      net.trainParam.pop(i,:) = -1*net.trainParam.initw+rand(1,D).*2*net.trainParam.initw; % initialisation from the article
    end;
  end
end;
pop = net.trainParam.pop;

% create other structures
popold    = zeros(size(pop));     % toggle population
val       = zeros(1,NP);          % create and reset the "cost array"
bestmem   = zeros(1,D);           % best population member ever
bestmemit = zeros(1,D);           % best population member in iteration

% Evaluate the best member after initialization
ibest   = inf;                      % start with first population member
bestval = inf;                 % best objective function value so far

for i=1:NP                        % check the remaining members
  net = setx(net,pop(i,:));
  [perf, El, Ac, N, Zb,Zi,Zl] = calcperf(net,pop(i,:),Pd,Tl,Ai,Q,TS);
  tr.perf(1) = perf;
  val(i) = perf;
  if (val(i) < bestval)           % if member is better
    ibest   = i;                 % save its location
    bestval = val(i);
  end   
end
bestmemit = pop(ibest,:);         % best member of current iteration
bestvalit = bestval;              % best value of current iteration
bestmem = bestmemit;              % best member ever

pm1 = zeros(NP,D);              % initialize population matrix 1
pm2 = zeros(NP,D);              % initialize population matrix 2
pm3 = zeros(NP,D);              % initialize population matrix 3
pm4 = zeros(NP,D);              % initialize population matrix 4
pm5 = zeros(NP,D);              % initialize population matrix 5
bm  = zeros(NP,D);              % initialize bestmember  matrix
ui  = zeros(NP,D);              % intermediate population of perturbed vectors
mui = zeros(NP,D);              % mask for intermediate population
mpo = zeros(NP,D);              % mask for old population
rot = (0:1:NP-1);               % rotating index array (size NP)
rotd= (0:1:D-1);                % rotating index array (size D)
rt  = zeros(NP);                % another rotating index array
rtd = zeros(D);                 % rotating index array for exponential crossover
a1  = zeros(NP);                % index array
a2  = zeros(NP);                % index array
a3  = zeros(NP);                % index array
a4  = zeros(NP);                % index array
a5  = zeros(NP);                % index array
ind = zeros(4);

tr = newtr(epochs,'perf','vperf','tperf','mu'); % new training record

%
% Train
%
for epoch = 0:epochs
  
  % Training Record
  net = setx(net, bestmem);
  [perf,El,Ac,N,Zb,Zi,Zl] = calcperf(net,bestmem,Pd,Tl,Ai,Q,TS);
  epochPlus = epoch+1;
  tr.perf(epoch+1) = perf;

  % Stopping Criteria
  currentTime = etime(clock,startTime);
  if (perf <= goal)
    stop = 'Performance goal met.';
  elseif (epoch == epochs)
    stop = 'Maximum epoch reached, performance goal was not met.';
  elseif (currentTime > time)
    stop = 'Maximum time elapsed, performance goal was not met.';
  end
  
  % Progress
  if ~rem(epoch, show) | length(stop)
    fprintf(this);
    if isfinite(epochs) fprintf(', Epoch %g/%g',epoch, epochs); end
    if isfinite(time) fprintf(', Time %4.1f%%',currentTime/time*100); end
    if isfinite(goal) fprintf(', %s %g/%g',upper(net.performFcn),perf,goal); end
    fprintf('\n')
    plotperf(tr,goal,this,epoch)
    if length(stop) fprintf('%s, %s\n\n',this,stop); break; end
  end
  
  % DIFFERENTIAL EVOLUTION ( this is fast way to calculate )
  popold = pop;                   % save the old population
  ind = randperm(4);              % index pointer array
  a1  = randperm(NP);             % shuffle locations of vectors
  rt = rem(rot+ind(1),NP);        % rotate indices by ind(1) positions
  a2  = a1(rt+1);                 % rotate vector locations
  rt = rem(rot+ind(2),NP);
  a3  = a2(rt+1);                
  rt = rem(rot+ind(3),NP);
  a4  = a3(rt+1);               
  rt = rem(rot+ind(4),NP);
  a5  = a4(rt+1);                
  pm1 = popold(a1,:);             % shuffled population 1
  pm2 = popold(a2,:);             % shuffled population 2
  pm3 = popold(a3,:);             % shuffled population 3
  pm4 = popold(a4,:);             % shuffled population 4
  pm5 = popold(a5,:);             % shuffled population 5
  
  for i=1:NP                      % population filled with the best member
    bm(i,:) = bestmemit;        % of the last iteration
  end
  
  mui = rand(NP,D) < CR;          % all random numbers < CR are 1, 0 otherwise
  
  if (strategy > 5) % EXP
    st = strategy-5;		  % binomial crossover
  else % BIN
    st = strategy;		  % exponential crossover
    mui=sort(mui');	          % transpose, collect 1's in each column
    for i=1:NP
      n=floor(rand*D);
      if n > 0
	rtd = rem(rotd+n,D);
	mui(:,i) = mui(rtd+1,i); %rotate column i by n
      end
    end
    mui = mui';			  % transpose back
  end
  mpo = mui < 0.5;                % inverse mask to mui
  
  % use strategy
  if (st == 1)                      % DE/best/1
    ui = bm + F*(pm1 - pm2);        % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 2)                  % DE/rand/1
    ui = pm3 + F*(pm1 - pm2);       % differential variation
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 3)                  % DE/rand-to-best/1
    ui = popold + F*(bm-popold) + F*(pm1 - pm2);        
    ui = popold.*mpo + ui.*mui;     % crossover
  elseif (st == 4)                  % DE/best/2
    ui = bm + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;           % crossover
  elseif (st == 5)                  % DE/rand/2
    ui = pm5 + F*(pm1 - pm2 + pm3 - pm4);  % differential variation
    ui = popold.*mpo + ui.*mui;            % crossover
  end
  
  % Select which vectors are allowed to enter the new population
  for i=1:NP
%    for j=1:D
%      if abs(ui(i,j)) > 10000 % if some absolute value of the weights is greater than 10 000,
%	ui(i,j) = -10000+rand*20000; % it is evaluated randomly between -10 000 and 10 000
%      end
%    end
    net = setx(net,ui(i,:));
    [perf, El, Ac, N, Zb,Zi,Zl] = calcperf(net,ui(i,:),Pd,Tl,Ai,Q,TS);
    tempval = perf;
    if (tempval <= val(i) & max(abs(ui(i,:)))<net.trainParam.maxvalue)  % if competitor is better than value in "cost array"
      pop(i,:) = ui(i,:);  % replace old vector with new one (for new iteration)
      val(i)   = tempval;  % save value in "cost array"
			   % we update bestval only in case of success to save time
      if (tempval < bestval)     % if competitor better than the best one ever
	bestval = tempval;      % new best value
	bestmem = ui(i,:);      % new best parameter vector ever          
      end
    end     
  end
  bestmemit = bestmem;       % freeze the best member of this iteration for the coming 
			     % iteration. This is needed for some of the strategies.
  fprintf(1,'mean MSE %f, mean popvalue %f\n',mean(val),mean(abs(pop(:))));
end % for epoch = 0:epochs

% Finish
net = setx(net,bestmem);
tr = cliptr(tr,epoch);

% Save population
net.trainParam.pop = pop;














end

