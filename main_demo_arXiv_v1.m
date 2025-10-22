
% LDPC decoder using Hidden Markov Model: https://arxiv.org/abs/2509.21872
% Demo code based on paper 
% 
% 260 bits frame, rate 1/2 demo.  
% 
% See Fig 5, IEEE TRANSACTIONS ON WIRELESS COMMUNICATIONS, VOL. 24, NO.
% 4, APRIL 2025, page 3386
%
% The regular random LDPC code under a HMM decoder outperforms:
% 1) Polar code (results Fig 5)
% 2) 5G optimized LDPC + Tanner BP
% 3) SO-GRAND
%
% code prints BER and FER and the frame counter





clear all

M = 130;  % information bits
weight = 3; %parity check weight
EbN0 = 3; % dB
nr_frames = 10^10;  % will terminate after 100 frames did not decode
iterations = 250; % number of BP iterations for Tanner
data_choice = 2;  % 1 is all zeros, 2 is random
DBN_tries = 50; % Statsitics attemps first iteration HMM
state_sz = 4;  % two bits
HMM_iterations = 5;  % how many HMM iterations
retries = 50; % number of times the HMM is retried with a random walk change
llr_limit = 1e-10; % limit the LLR magnitude
stage_two = 0; % 0 => stage 2 disabled here

%=======================================================
% main Tanner graph H: 

load G 
load H
range = M/65*[1:30]; % for statistical ranking stage

H_dummy = H';
H_dag(1:M,:) = H_dummy(M+1:end,:); % matlab format
H_dag(1+M:2*M,:) = H_dummy(1:M,:);
clear H_dummy;

% define matlab object for Tanner decoder
 global hDec1;
 hDec1 = comm.LDPCDecoder(H_dag');
 hDec1.MaximumIterationCount = iterations;
 hDec1.OutputValue =  'Whole codeword';
 hDec1.DecisionMethod =  'Soft decision';


%=======================================
% make the transitional table
%=======================================
% transitional probability table  P(Xk=xi|Xk-1=xj) 
   size_par=6;  % number of parity checks per column of H
   Q = 2^size_par;  % number of states 
  
 T = [0.5 0.5 0   0 
      0   0   0.5 0.5
      0.5 0.5 0   0
      0   0   0.5 0.5];


%sum(sum(mod(G*H_dag,2)))
%=======================
 % make 6 bit matrix
 for loop=1:Q % go over Q states, probability distribution for zero
        a = dec2bin(loop-1,size_par);           
        kandidaat = a-'0';
        bit_matrix(loop,:) = kandidaat;
 end


%========================================
%simulation
%========================================

total_samples = 0; 
total_errors = 0;
Rc = M/(2*M); % rate 1/2 fixed

kn_tx2rx = sqrt(1./(10.^(EbN0./10)*2*Rc)); %noise std dev
frame_error_cnt = 1; errors_stats = zeros(1,M*2);
FER = 0;
rng('default')
rng('shuffle');  % Seed the RNG based on the current time   
counter_difficult = 1;
while (frame_error_cnt < nr_frames)  % monte Carlo loop
   
   if data_choice == 1
      data = zeros(1,M); % 
   else
      data = (rand(1,M)>0.5); % random data
   end
   encoded = mod(data*G,2); % encode
   rx = 2*encoded-1 + kn_tx2rx.*randn(1,2*M);  %noisy received data
   std_rx = std(rx); % for use in statistics
   f1=1./(1+exp(-2*rx/kn_tx2rx^2));        % likelihoods
   f0=1-f1;  
   llr = log((1-f0)./(f0-1e-30)); %LLR
   prob_to_modify = [f0' f1'];  % for input to HMM

%==============================================
% try Tanner first, if it fails, then HMM will kick in 
   Tanner = step(hDec1, llr'); % factor graph
   hard_bits_tanner = (sign(Tanner)+1)/2;
   errors = sum(abs(hard_bits_tanner(1:M)' - data));
   class(frame_error_cnt) = 0;  % so 0 will mean did not decode
   if errors < 3
      errors = 0;  % 2 or less errors are easy to fix for these short frames
      class(frame_error_cnt) = 1; % decoded stage one
   end

%==============================================
% if Tanner failed, rest below is HMM stages

   % Stage 1
   % HMM iterrative, passing to Tanner after each iteration, simple
   % emissions
   if errors > 0 % iterative HMM and Tanner combined attempt
    herhalers = 0;
    while herhalers <= retries & errors > 0   % HMM ran many times, each time new walk
     rng('shuffle');  % Seed the RNG based on the current time 
     walk = random_walk_create(size_par,H_dag,M); % generates the random walk
     [llr_hmm,hard_est,flag,which_decoder] = HMM_iter_v2(llr_limit,walk,prob_to_modify,HMM_iterations,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,encoded); % if flag == 0 it decoded  
     if flag == 0 
        errors = flag; 
        class(frame_error_cnt) = 10;
     end
     herhalers = herhalers + 1;
    end
   end

 % Stage two
 if stage_two == 1   % higher order emissions.  When 0 stage two bypassed to speed up simulations
   prob_to_modify = [f0' f1'];  % input to HMM
   if errors > 0 
    herhalers = 0;
    while herhalers <= retries & errors > 0   % HMM ran many times, each time new walk
     rng('shuffle');  % Seed the RNG based on the current time 
     walk = random_walk_create(size_par,H_dag,M); % generates the random walk
     [llr_hmm,hard_est,flag,which_decoder] = HMM_iter_v2_emis(llr_limit,walk,prob_to_modify,HMM_iterations,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,encoded); % if flag == 0 it decoded  
     if flag == 0 
        errors = flag; 
        class(frame_error_cnt) = 2;
     end
     herhalers = herhalers + 1;
     ['stage_2']
    end
   end
end

  
%Stage 3: deploy statistics and set uncertain bits LLR=0, simple emissions
   if errors > 0
     [hard_errors_amount,flag] = statistics(llr_limit,std_rx,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,DBN_tries,range,data);
     if flag == 0 
        errors = 0;
        class(frame_error_cnt) = 3;
     end
   end

%Stage 4: all else failed, high order emissions, statistics and set uncertain bits LLR=0
   if errors > 0
     [hard_errors_amount,flag] = statistics_emis(llr_limit,std_rx,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,DBN_tries,range,data);
     if flag == 0 
        errors = 0;
        class(frame_error_cnt) = 4;
     end
   end


   if errors > 0
      difficult_frames(counter_difficult,:) = rx;  % store hard ones
      counter_difficult = counter_difficult + 1;
   end

   

   total_errors = total_errors + errors;
   total_samples = total_samples + M;
   FER = FER + sign(errors);
   [total_errors/total_samples FER/frame_error_cnt]  % BER and FER
   frame_error_cnt = frame_error_cnt + 1;
   frame_count = frame_error_cnt;
   [frame_count]

   if FER == 100
      break;
   end

end % end of monte carlo frames 







%=======================================================================%
%      Functions
%=======================================================================%

function [llr,prob_for_zero] = HMM(llr_limit,walk_state,prob,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz)
   
% prob => vector
% llr => log likelyhood vector
% kn_tx2rx is noise std dev


prob_to_modify = prob; % this is a local matrix, historical reasons

walk_sequence = walk_state(1,[1:end]); % this is how forward walked
steps = length(walk_sequence);

% now do MAP
% forward 
f = [0.5 0.5 0 0]'; % starts in state with bit 0
sm_s = kn_tx2rx^2; % var(noise);  
%walk_state = []; 
n = 1; teller1 = 1; 
indeks1 = find(H_dag(:,n)==1);
dominant_bit_pos = indeks1(1); % default start at top
teller1 =1;
for n=1:steps % only stop if all bits visited
     
    indeks1 = find(H_dag(:,walk_sequence(n))==1); % positions in this column where H == 1
    I1 = find(indeks1 == walk_state(2,n));  % dominant bit
    I2 = find(indeks1 == walk_state(3,n));  %These were the bits used for this check


    %indeks1 = find(H_dag(:,n)==1);
    %I1 = find(indeks1 == dominant_bit_pos);
    %dummy = [1:6];
    %dummy(I1) = [];
    %dummy(randperm(size_par-1)) = dummy;  % random permutation of indeks

    %I2 = dummy(1);  
    %walk_state = [walk_state [n indeks1(I1) indeks1(I2)]'];  % parity check and bit positions
   
    akkumeleer1 = 0; akkumeleer2 = 0;  akkumeleer3 = 0; akkumeleer4 = 0;
    for party1=1:Q  % parity one Q states
          kandidaat1 = bit_matrix(party1,:);  % candidate for this state
          pariteit1 = mod(sum(kandidaat1),2); % parity for check
          
        % this is for state [0,0]:  first bit is main state, second bit to
        % be main state in next ceck
        if pariteit1 == 0
          if kandidaat1(I1) == 0  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
          % this is for state [0,1]
          if kandidaat1(I1) == 0  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer2 = akkumeleer2 + probab_combo1;
          end
          % this is for state [1,0]
          if kandidaat1(I1) == 1  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer3 = akkumeleer3 + probab_combo1;
          end
          % this is for state [1,1]
          if kandidaat1(I1) == 1  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer4 = akkumeleer4 + probab_combo1;
          end
        end
       end % end of parity 1 states

       observe_prob = [akkumeleer1 akkumeleer2 akkumeleer3 akkumeleer4];  % state un-norm
       observe_prob = observe_prob/sum(observe_prob);  % normalized for numerical reasons

%if ismember(n,kill_it)
 %   observe_prob = [1 1 1 1]/4; 
%end

       evidence_vec = observe_prob;  % its a probability!
       O = diag(evidence_vec); 
       f = O*T'*f; f = f/sum(f);
       F(:,teller1) = f; % state for n
       
       % bepaal sommer nou die volgende stap/pariteit toets
       %dummy = find(H_dag(indeks1(I2),:)==1);
       %pointer = find(dummy == n);  % which one is current time-stamp
       %indeks_next = dummy;
       %indeks_next(pointer) = []; % remove current time-stamp indeks 
       %indeks_next(randperm(2)) = indeks_next;  
       %n = indeks_next(1); % this is the next step of the random walk.
       %dominant_bit_pos = indeks1(I2); % this must be the dominant state of next time stamp
       %termination_test = length(setdiff(1:2*M, walk_state(2,:)));
     %  [teller1 termination_test]
       teller1 = teller1 + 1;
    end

% backward msg

ep = 1e-8;
f = 1/state_sz*ones(1,state_sz)'; % all equally likely, not observed
walk_sequence = walk_state(1,[1:end]); % this is how forward walked
steps = length(walk_sequence);
B(:,steps+1) = f;
%f = [1 0 0 0]'; 
sm_s = kn_tx2rx^2; % var(noise);  % complex noise
teller1 = 1;
for n=steps:-1:1 
    indeks1 = find(H_dag(:,walk_sequence(n))==1); % positions in this column where H == 1
    I1 = find(indeks1 == walk_state(2,n));  % dominant bit
    I2 = find(indeks1 == walk_state(3,n));  %These were the bits used for this check

    akkumeleer1 = 0; akkumeleer2 = 0;  akkumeleer3 = 0; akkumeleer4 = 0;
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
          
        % this is for state [0,0]:  first bit is main state, second bit to
        % be main state in next ceck
        if pariteit1 == 0
         if kandidaat1(I1) == 0  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
         end
          % this is for state [0,1]
         if kandidaat1(I1) == 0  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer2 = akkumeleer2 + probab_combo1;
         end
          % this is for state [1,0]
         if kandidaat1(I1) == 1  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer3 = akkumeleer3 + probab_combo1;
         end
        % this is for state [1,1]
         if kandidaat1(I1) == 1  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer4 = akkumeleer4 + probab_combo1;
         end
        end
       end % end of parity 1 states

       observe_prob = [akkumeleer1 akkumeleer2 akkumeleer3 akkumeleer4];  % state un-norm
       observe_prob = observe_prob/sum(observe_prob);  % normalized for numerical reasons
       
%if ismember(n,kill_it)
 %   observe_prob = [1 1 1 1]/4; 
%end

       evidence_vec = observe_prob;
       O = diag(evidence_vec); 
       %f = O*T'*f; f = f/sum(f);
       f = T*O*f; f = f/sum(f); % formalize to prevent underflow
       B(:,n) = f; % state for n
       teller1 = teller1 + 1;
end

% combine forward and backward for map
for n=1:steps
    combine = F(:,n).*B(:,n+1);
    map_est(:,n) = combine/sum(combine);
end

% compute posterior bit prob and llr using map_est
for loop=1:2*M  % all 2*M bits
    JJ = find(walk_state(2,:) == loop);
    Pr_0_dom = sum(map_est(1:2,[JJ]));
    JJ = find(walk_state(3,:) == loop);
    Pr_0_sec = sum(map_est(1:2:3,[JJ]));
    a = [Pr_0_dom Pr_0_sec]; % probability for being a zero
    soft_random_walk = log((1-a+llr_limit)./(a+llr_limit)); 
    llr(loop) = mean(soft_random_walk); % this is a output
    prob_for_zero(loop) = mean(a); % this is a output
    %if prob_for_zero == 1
    %   prob_for_zero = 0.99999999;
    %end
    %prob_for_zero(loop) = mean(a); % this is a output
    %if prob_for_zero == 0
    %   prob_for_zero = 0.00000001;
    %end

end

end %end of function


% =================================================================

function [llr,prob_for_zero] = HMM_emis(llr_limit,walk_state,prob,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz)

% better emission model, more complex
   
% prob => vector
% llr => log likelyhood vector
% kn_tx2rx is noise std dev


prob_to_modify = prob; % this is a local matrix, historical reasons

walk_sequence = walk_state(1,[1:end]); % this is how forward walked
steps = length(walk_sequence);

% now do MAP
% forward 
f = [0.5 0.5 0 0]'; % starts in state with bit 0
sm_s = kn_tx2rx^2; % var(noise);  
%walk_state = []; 
n = 1; teller1 = 1; 
indeks1 = find(H_dag(:,n)==1);
dominant_bit_pos = indeks1(1); % default start at top
teller1 =1;
for n=1:steps % only stop if all bits visited
     
    indeks1 = find(H_dag(:,walk_sequence(n))==1); % positions in this column where H == 1
    I1 = find(indeks1 == walk_state(2,n));  % dominant bit
    I2 = find(indeks1 == walk_state(3,n));  %These were the bits used for this check

% compute prior Ps first 
    rows_for_I1 = find(H_dag(indeks1(I1),:) == 1);
    rows_for_I2 = find(H_dag(indeks1(I2),:) == 1);
    rows_for_I1(rows_for_I1 == walk_sequence(n)) = [];  % remove current check
    rows_for_I2(rows_for_I2 == walk_sequence(n)) = [];  % remove current check

    kolom1 = H_dag(:,rows_for_I1(1));
    kolom2 = H_dag(:,rows_for_I1(2));
    kolom3 = H_dag(:,rows_for_I2(1));
    kolom4 = H_dag(:,rows_for_I2(2));

    akkumeleer1 = 0;  % P \propto dominant bit, first check = 0
    I = find(kolom1 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check1(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto dominant bit first check = 1
    I = find(kolom1 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check1(2) = akkumeleer1;  

    

    akkumeleer1 = 0;  % P \propto dominant bit, second check = 0
    I = find(kolom2 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check2(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto dominant bit second check = 1
    I = find(kolom2 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check2(2) = akkumeleer1;  

    %-----------------------------------
    % secondary bit
    akkumeleer1 = 0;  % P \propto secondary bit, first check = 0
    I = find(kolom3 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check1(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto secondary bit first check = 1
    I = find(kolom3 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check1(2) = akkumeleer1;  

    

    akkumeleer1 = 0;  % P \propto seconday bit, second check = 0
    I = find(kolom4 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check2(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto secondary bit second check = 1
    I = find(kolom4 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check2(2) = akkumeleer1;  

 
    % resume old chain now
    akkumeleer1 = 0; akkumeleer2 = 0;  akkumeleer3 = 0; akkumeleer4 = 0;
    for party1=1:Q  % parity one Q states
          kandidaat1 = bit_matrix(party1,:);  % candidate for this state
          pariteit1 = mod(sum(kandidaat1),2); % parity for check
          
        % this is for state [0,0]:  first bit is main state, second bit to
        % be main state in next ceck
        if pariteit1 == 0
          if kandidaat1(I1) == 0  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
          % this is for state [0,1]
          if kandidaat1(I1) == 0  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer2 = akkumeleer2 + probab_combo1;
          end
          % this is for state [1,0]
          if kandidaat1(I1) == 1  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer3 = akkumeleer3 + probab_combo1;
          end
          % this is for state [1,1]
          if kandidaat1(I1) == 1  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer4 = akkumeleer4 + probab_combo1;
          end
        end
       end % end of parity 1 states

       % normalize side info
       %P_domin_check1 = P_domin_check1/sum(P_domin_check1);
       %P_domin_check2 = P_domin_check2/sum(P_domin_check2);
       %P_secondary_check1 = P_secondary_check1/sum(P_secondary_check1);
       %P_secondary_check2 = P_secondary_check2/sum(P_secondary_check2);

       observe_prob = [akkumeleer1 akkumeleer2 akkumeleer3 akkumeleer4];  % state un-norm
       observe_prob(1) = observe_prob(1)*P_domin_check1(1)*P_domin_check2(1)*...
                                         P_secondary_check1(1)*P_secondary_check2(1);
       observe_prob(2) = observe_prob(2)*P_domin_check1(1)*P_domin_check2(1)*...
                                         P_secondary_check1(2)*P_secondary_check2(2);
       observe_prob(3) = observe_prob(3)*P_domin_check1(2)*P_domin_check2(2)*...
                                         P_secondary_check1(1)*P_secondary_check2(1);
       observe_prob(4) = observe_prob(4)*P_domin_check1(2)*P_domin_check2(2)*...
                                         P_secondary_check1(2)*P_secondary_check2(2);

     %  observe_prob(1) = observe_prob(1)*P_domin_check1(1)*P_secondary_check2(1);
     %  observe_prob(2) = observe_prob(2)*P_domin_check1(1)*P_secondary_check2(2);
     %  observe_prob(3) = observe_prob(3)*P_domin_check1(2)*P_secondary_check2(1);
     %  observe_prob(4) = observe_prob(4)*P_domin_check1(2)*P_secondary_check2(2);


       observe_prob = observe_prob/sum(observe_prob);  % normalized for numerical reasons

%if ismember(n,kill_it)
 %   observe_prob = [1 1 1 1]/4; 
%end

       evidence_vec = observe_prob;  % its a probability!
       O = diag(evidence_vec); 
       f = O*T'*f; f = f/sum(f);
       F(:,teller1) = f; % state for n
       
       % bepaal sommer nou die volgende stap/pariteit toets
       %dummy = find(H_dag(indeks1(I2),:)==1);
       %pointer = find(dummy == n);  % which one is current time-stamp
       %indeks_next = dummy;
       %indeks_next(pointer) = []; % remove current time-stamp indeks 
       %indeks_next(randperm(2)) = indeks_next;  
       %n = indeks_next(1); % this is the next step of the random walk.
       %dominant_bit_pos = indeks1(I2); % this must be the dominant state of next time stamp
       %termination_test = length(setdiff(1:2*M, walk_state(2,:)));
     %  [teller1 termination_test]
       teller1 = teller1 + 1;
    end

% backward msg

ep = 1e-8;
f = 1/state_sz*ones(1,state_sz)'; % all equally likely, not observed
walk_sequence = walk_state(1,[1:end]); % this is how forward walked
steps = length(walk_sequence);
B(:,steps+1) = f;
%f = [1 0 0 0]'; 
sm_s = kn_tx2rx^2; % var(noise);  % complex noise
teller1 = 1;
for n=steps:-1:1 
    indeks1 = find(H_dag(:,walk_sequence(n))==1); % positions in this column where H == 1
    I1 = find(indeks1 == walk_state(2,n));  % dominant bit
    I2 = find(indeks1 == walk_state(3,n));  %These were the bits used for this check

    % compute prior Ps first 
    rows_for_I1 = find(H_dag(indeks1(I1),:) == 1);
    rows_for_I2 = find(H_dag(indeks1(I2),:) == 1);
    rows_for_I1(rows_for_I1 == walk_sequence(n)) = [];  % remove current check
    rows_for_I2(rows_for_I2 == walk_sequence(n)) = [];  % remove current check

    kolom1 = H_dag(:,rows_for_I1(1));
    kolom2 = H_dag(:,rows_for_I1(2));
    kolom3 = H_dag(:,rows_for_I2(1));
    kolom4 = H_dag(:,rows_for_I2(2));

    akkumeleer1 = 0;  % P \propto dominant bit, first check = 0
    I = find(kolom1 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check1(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto dominant bit first check = 1
    I = find(kolom1 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check1(2) = akkumeleer1;  

    

    akkumeleer1 = 0;  % P \propto dominant bit, second check = 0
    I = find(kolom2 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check2(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto dominant bit second check = 1
    I = find(kolom2 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_domin_check2(2) = akkumeleer1;  

    %-----------------------------------
    % secondary bit
    akkumeleer1 = 0;  % P \propto secondary bit, first check = 0
    I = find(kolom3 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check1(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto secondary bit first check = 1
    I = find(kolom3 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check1(2) = akkumeleer1;  

    

    akkumeleer1 = 0;  % P \propto seconday bit, second check = 0
    I = find(kolom4 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check2(1) = akkumeleer1;  

    akkumeleer1 = 0;  % P \propto secondary bit second check = 1
    I = find(kolom4 == 1);
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
        if pariteit1 == 0
          if kandidaat1(I1) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(I(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
          end
        end
    end
    P_secondary_check2(2) = akkumeleer1;  

% resume old chain 
    akkumeleer1 = 0; akkumeleer2 = 0;  akkumeleer3 = 0; akkumeleer4 = 0;
    for party1=1:Q  % parity one Q states
        kandidaat1 = bit_matrix(party1,:);  % candidate for this state
        pariteit1 = mod(sum(kandidaat1),2); % parity for check
          
        % this is for state [0,0]:  first bit is main state, second bit to
        % be main state in next ceck
        if pariteit1 == 0
         if kandidaat1(I1) == 0  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer1 = akkumeleer1 + probab_combo1;
         end
          % this is for state [0,1]
         if kandidaat1(I1) == 0  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer2 = akkumeleer2 + probab_combo1;
         end
          % this is for state [1,0]
         if kandidaat1(I1) == 1  & kandidaat1(I2) == 0
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer3 = akkumeleer3 + probab_combo1;
         end
        % this is for state [1,1]
         if kandidaat1(I1) == 1  & kandidaat1(I2) == 1
             for lopie=1:size_par
                 vektor1(lopie) = prob_to_modify(indeks1(lopie),kandidaat1(lopie)+1);
             end
             probab_combo1 = prod(vektor1); % probability for this combination
             akkumeleer4 = akkumeleer4 + probab_combo1;
         end
        end
       end % end of parity 1 states

       % normalize side info
      % P_domin_check1 = P_domin_check1/sum(P_domin_check1);
      % P_domin_check2 = P_domin_check2/sum(P_domin_check2);
      % P_secondary_check1 = P_secondary_check1/sum(P_secondary_check1);
      % P_secondary_check2 = P_secondary_check2/sum(P_secondary_check2);

       observe_prob = [akkumeleer1 akkumeleer2 akkumeleer3 akkumeleer4];  % state un-norm
       observe_prob(1) = observe_prob(1)*P_domin_check1(1)*P_domin_check2(1)*...
                                         P_secondary_check1(1)*P_secondary_check2(1);
       observe_prob(2) = observe_prob(2)*P_domin_check1(1)*P_domin_check2(1)*...
                                         P_secondary_check1(2)*P_secondary_check2(2);
       observe_prob(3) = observe_prob(3)*P_domin_check1(2)*P_domin_check2(2)*...
                                         P_secondary_check1(1)*P_secondary_check2(1);
       observe_prob(4) = observe_prob(4)*P_domin_check1(2)*P_domin_check2(2)*...
                                         P_secondary_check1(2)*P_secondary_check2(2);

       %observe_prob(1) = observe_prob(1)*P_domin_check1(1)*P_secondary_check2(1);
       %observe_prob(2) = observe_prob(2)*P_domin_check1(1)*P_secondary_check2(2);
       %observe_prob(3) = observe_prob(3)*P_domin_check1(2)*P_secondary_check2(1);
       %observe_prob(4) = observe_prob(4)*P_domin_check1(2)*P_secondary_check2(2);


       observe_prob = observe_prob/sum(observe_prob);  % normalized for numerical reasons



       %observe_prob = [akkumeleer1 akkumeleer2 akkumeleer3 akkumeleer4];  % state un-norm
       %observe_prob = observe_prob/sum(observe_prob);  % normalized for numerical reasons
       
%if ismember(n,kill_it)
 %   observe_prob = [1 1 1 1]/4; 
%end

       evidence_vec = observe_prob;
       O = diag(evidence_vec); 
       %f = O*T'*f; f = f/sum(f);
       f = T*O*f; f = f/sum(f); % formalize to prevent underflow
       B(:,n) = f; % state for n
       teller1 = teller1 + 1;
end

% combine forward and backward for map
for n=1:steps
    combine = F(:,n).*B(:,n+1);
    map_est(:,n) = combine/sum(combine);
end

% compute posterior bit prob and llr using map_est
for loop=1:2*M  % all 2*M bits
    JJ = find(walk_state(2,:) == loop);
    Pr_0_dom = sum(map_est(1:2,[JJ]));
    JJ = find(walk_state(3,:) == loop);
    Pr_0_sec = sum(map_est(1:2:3,[JJ]));
    a = [Pr_0_dom Pr_0_sec]; % probability for being a zero
    soft_random_walk = log((1-a+llr_limit)./(a+llr_limit)); 
    llr(loop) = mean(soft_random_walk); % this is a output
    prob_for_zero(loop) = mean(a); % this is a output
    %if prob_for_zero == 1
    %   prob_for_zero = 0.99999999;
    %end
    %prob_for_zero(loop) = mean(a); % this is a output
    %if prob_for_zero == 0
    %   prob_for_zero = 0.00000001;
    %end

end

end %end of function



%==================================================================

function [llr_from_hmm,hard_hmm,errors,which_decoder] = HMM_iter_v2(llr_limit,walk,prob_to_modify,HMM_iterations,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,encoded)

global hDec1;
errors = 100;
which_decoder = 2;

doen_weer = 0;
while doen_weer < HMM_iterations  
    
    [llr_from_hmm,prob_zero] = HMM(llr_limit,walk,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz);  %call HMM, it ceates a random walk by itself 
    f1 = 1./(1+exp(-llr_from_hmm));
    f0=1-f1;
    prob_to_modify = [f0' f1'];
    %prob_to_modify = [prob_zero' (1-prob_zero)']; % feedback
    %stem(llr_from_hmm)
    hard_hmm = (sign(llr_from_hmm)+1)/2;
    error_hmm = sum(abs(hard_hmm(1:M)-encoded(1:M)));
    if error_hmm < 3
       errors = 0;
       doen_weer = 10^3; % disable this stage
       which_decoder = 0; 
    end

    tanner_try = step(hDec1, (llr_from_hmm'));  % try Tanner flooding
    hard_hmm = ((sign(tanner_try)+1)/2)';
    error_hmm = sum(abs(hard_hmm(1:M)-encoded(1:M)));
    if error_hmm < 3
       errors = 0;
       doen_weer = 10^3; % disable this stage
       which_decoder = 1;
    end  
    doen_weer = doen_weer + 1;
end

end % end of function

%==================================================================

function [llr_from_hmm,hard_hmm,errors,which_decoder] = HMM_iter_v2_emis(llr_limit,walk,prob_to_modify,HMM_iterations,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,encoded)

% better emissions

global hDec1;
errors = 100;
which_decoder = 2;

doen_weer = 0;
while doen_weer < HMM_iterations  
    
    [llr_from_hmm,prob_zero] = HMM_emis(llr_limit,walk,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz);  %call HMM, it ceates a random walk by itself 
    f1 = 1./(1+exp(-llr_from_hmm));
    f0=1-f1;
    prob_to_modify = [f0' f1'];
    %prob_to_modify = [prob_zero' (1-prob_zero)']; % feedback
    %stem(llr_from_hmm)
    hard_hmm = (sign(llr_from_hmm)+1)/2;
    error_hmm = sum(abs(hard_hmm(1:M)-encoded(1:M)));
    if error_hmm < 3
       errors = 0;
       doen_weer = 10^3; % disable this stage
       which_decoder = 0; 
    end

    tanner_try = step(hDec1, (llr_from_hmm'));  % try Tanner flooding
    hard_hmm = ((sign(tanner_try)+1)/2)';
    error_hmm = sum(abs(hard_hmm(1:M)-encoded(1:M)));
    if error_hmm < 3
       errors = 0;
       doen_weer = 10^3; % disable this stage
       which_decoder = 1;
    end  
    doen_weer = doen_weer + 1;
end

end % end of function


%==================================================================

% statistical data exploit

function [hard_bits_errors,errors] = statistics(llr_limit,std_rx,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,DBN_tries,range,data)

  %global hDec2; % 1 iteration
  global hDec1; % lots iterations
  errors = 100;
  try_again = 1;   

  while try_again <= DBN_tries & errors > 0 %  if HMM iterative failed, statistics sorting attempt   
    rng('shuffle');  % Seed the RNG based on the current time 
    walk = random_walk_create(size_par,H_dag,M); % generates the random walk
    [llr_from_hmm,prob_zero] = HMM(llr_limit,walk,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz);  %call HMM, it ceates a random walk by itself 
    DBN_tanner = step(hDec1, (llr_from_hmm')); % try decoding on plain HMM output
    hard_bits = (sign(DBN_tanner)+1)/2;
    errors_HMM_tanner_cleaned = sum(abs(hard_bits(1:M)' - data));
    if errors_HMM_tanner_cleaned < 3
       errors = 0;
    end
    llr_HMM = llr_from_hmm;
    if errors > 0 % continue to make stats data
       llr_HMM_array(:,try_again) = DBN_tanner; % /std(DBN_tanner);  % build array
       std_vector = std(llr_HMM_array,1,2)./abs(mean(llr_HMM_array,2));  % std of rows 
       [waarde,posisie] = sort(std_vector); % least changing first

       llr_HMM_array_no_tanner(:,try_again) = llr_HMM'; % /std(DBN_tanner);  % build array
       std_vector = std(llr_HMM_array_no_tanner,1,2)./abs(mean(llr_HMM_array_no_tanner,2));  % std of rows 
       [waarde_no_tanner,posisie_no_tanner] = sort(std_vector); % least changing first
    end

   if try_again > 2 % if larger, sort 
     dummy = DBN_tanner(posisie); 
     lopertjie = 1;
     while lopertjie <= length(range) & errors > 0
           dummy(end-range(lopertjie):end) = 0; % these are most likely junk
           cleaned(posisie) = dummy;
           cleaned = cleaned/std(cleaned)*std_rx;
           f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
           f0_c=1-f1_c;  
           llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
           DBN_tanner_cleaned = step(hDec1,llr_cleaned');
           hard_bits_try_cleaned = (sign(DBN_tanner_cleaned)+1)/2;
           errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
           if errors_HMM_tanner_cleaned < 3
                errors = 0;
           end
           lopertjie = lopertjie + 1;
    end

    dummy = llr_HMM(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
            dummy(end-range(lopertjie):end) = 0; % these are most likely junk
            cleaned(posisie) = dummy;
            cleaned = cleaned/std(cleaned)*std_rx;
            f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
            f0_c=1-f1_c;  
            llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
            HMM_cleaned = step(hDec1,llr_cleaned');
            hard_bits_try_cleaned = (sign(HMM_cleaned)+1)/2;
            errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
            if errors_HMM_tanner_cleaned < 3
                errors = 0;
            end
            lopertjie = lopertjie + 1;
    end

% now position not using Tanner 
    posisie = posisie_no_tanner; 
    dummy = DBN_tanner(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
        dummy(end-range(lopertjie):end) = 0; % these are most likely junk
        cleaned(posisie) = dummy;
        cleaned = cleaned/std(cleaned)*std_rx;
        f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
        f0_c=1-f1_c;  
        llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
        DBN_tanner_cleaned = step(hDec1,llr_cleaned');
        hard_bits_try_cleaned = (sign(DBN_tanner_cleaned)+1)/2;
        errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
        if errors_HMM_tanner_cleaned < 3
            errors = 0;
        end
        lopertjie = lopertjie + 1;
    end

    dummy = llr_HMM(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
        dummy(end-range(lopertjie):end) = 0; % these are most likely junk
        cleaned(posisie) = dummy;
        cleaned = cleaned/std(cleaned)*std_rx;
        f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
        f0_c=1-f1_c;  
        llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
        HMM_cleaned = step(hDec1,llr_cleaned');
        hard_bits_try_cleaned = (sign(HMM_cleaned)+1)/2;
        errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
        if errors_HMM_tanner_cleaned < 3
            errors = 0;
        end     
        lopertjie = lopertjie + 1;
    end
 %   [try_again 1]
   end
    try_again = try_again + 1;
 end % end for this random walk

 hard_bits_errors = errors_HMM_tanner_cleaned;

end


%==================================================================

% statistical data exploit

function [hard_bits_errors,errors] = statistics_emis(llr_limit,std_rx,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,DBN_tries,range,data)

% better emissions

  %global hDec2; % 1 iteration
  global hDec1; % lots iterations
  errors = 100;
  try_again = 1;   

  while try_again <= DBN_tries & errors > 0 %  if HMM iterative failed, statistics sorting attempt   
    rng('shuffle');  % Seed the RNG based on the current time 
    walk = random_walk_create(size_par,H_dag,M); % generates the random walk
    [llr_from_hmm,prob_zero] = HMM_emis(llr_limit,walk,prob_to_modify,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz);  %call HMM, it ceates a random walk by itself 
    DBN_tanner = step(hDec1, (llr_from_hmm')); % try decoding on plain HMM output
    hard_bits = (sign(DBN_tanner)+1)/2;
    errors_HMM_tanner_cleaned = sum(abs(hard_bits(1:M)' - data));
    if errors_HMM_tanner_cleaned < 3
       errors = 0;
    end
    llr_HMM = llr_from_hmm;
    if errors > 0 % continue to make stats data
       llr_HMM_array(:,try_again) = DBN_tanner; % /std(DBN_tanner);  % build array
       std_vector = std(llr_HMM_array,1,2)./abs(mean(llr_HMM_array,2));  % std of rows 
       [waarde,posisie] = sort(std_vector); % least changing first

       llr_HMM_array_no_tanner(:,try_again) = llr_HMM'; % /std(DBN_tanner);  % build array
       std_vector = std(llr_HMM_array_no_tanner,1,2)./abs(mean(llr_HMM_array_no_tanner,2));  % std of rows 
       [waarde_no_tanner,posisie_no_tanner] = sort(std_vector); % least changing first
    end

   if try_again > 2 % if larger, sort 
     dummy = DBN_tanner(posisie); 
     lopertjie = 1;
     while lopertjie <= length(range) & errors > 0
           dummy(end-range(lopertjie):end) = 0; % these are most likely junk
           cleaned(posisie) = dummy;
           cleaned = cleaned/std(cleaned)*std_rx;
           f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
           f0_c=1-f1_c;  
           llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
           DBN_tanner_cleaned = step(hDec1,llr_cleaned');
           hard_bits_try_cleaned = (sign(DBN_tanner_cleaned)+1)/2;
           errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
           if errors_HMM_tanner_cleaned < 3
                errors = 0;
           end
           lopertjie = lopertjie + 1;
    end

    dummy = llr_HMM(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
            dummy(end-range(lopertjie):end) = 0; % these are most likely junk
            cleaned(posisie) = dummy;
            cleaned = cleaned/std(cleaned)*std_rx;
            f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
            f0_c=1-f1_c;  
            llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
            HMM_cleaned = step(hDec1,llr_cleaned');
            hard_bits_try_cleaned = (sign(HMM_cleaned)+1)/2;
            errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
            if errors_HMM_tanner_cleaned < 3
                errors = 0;
            end
            lopertjie = lopertjie + 1;
    end

% now position not using Tanner 
    posisie = posisie_no_tanner; 
    dummy = DBN_tanner(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
        dummy(end-range(lopertjie):end) = 0; % these are most likely junk
        cleaned(posisie) = dummy;
        cleaned = cleaned/std(cleaned)*std_rx;
        f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
        f0_c=1-f1_c;  
        llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
        DBN_tanner_cleaned = step(hDec1,llr_cleaned');
        hard_bits_try_cleaned = (sign(DBN_tanner_cleaned)+1)/2;
        errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
        if errors_HMM_tanner_cleaned < 3
            errors = 0;
        end
        lopertjie = lopertjie + 1;
    end

    dummy = llr_HMM(posisie); 
    lopertjie = 1;
    while lopertjie <= length(range) & errors > 0
        dummy(end-range(lopertjie):end) = 0; % these are most likely junk
        cleaned(posisie) = dummy;
        cleaned = cleaned/std(cleaned)*std_rx;
        f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
        f0_c=1-f1_c;  
        llr_cleaned = log((f1_c)./(f0_c-1e-30)); 
        HMM_cleaned = step(hDec1,llr_cleaned');
        hard_bits_try_cleaned = (sign(HMM_cleaned)+1)/2;
        errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
        if errors_HMM_tanner_cleaned < 3
            errors = 0;
        end     
        lopertjie = lopertjie + 1;
    end
  %  [try_again 2]
   end
    try_again = try_again + 1;
 end % end for this random walk

if 0%  try_again == DBN_tries+1 & errors > 0
posisie = posisie_no_tanner; 
dummy = llr_HMM(posisie); 
lopertjie = 1;
while lopertjie <= length(range) & errors > 0
        dummy(end-range(lopertjie):end) = 0; % these are most likely junk
        cleaned(posisie) = dummy;
        cleaned = cleaned/std(cleaned)*std_rx;
        f1_c=1./(1+exp(-2*cleaned/kn_tx2rx^2));        % likelihoods
        f0_c=1-f1_c;  
        prob_cleaned = [f0_c' f1_c'];  % input to HMM
        rng('shuffle');  % Seed the RNG based on the current time 
        walk = random_walk_create(size_par,H_dag,M); % generates the random walk
        [HMM_cleaned,xx,flag,which_decoder] = HMM_iter_v2(walk,prob_cleaned,5,size_par,Q,kn_tx2rx,H_dag,bit_matrix,T,M,state_sz,data); % if flag == 0 it decoded  
        hard_bits_try_cleaned = (sign(HMM_cleaned)+1)/2;
        errors_HMM_tanner_cleaned = sum(abs(hard_bits_try_cleaned(1:M)' - data));
        if errors_HMM_tanner_cleaned < 3
            errors = 0;
        end     
   lopertjie = lopertjie + 1;
end
end



 hard_bits_errors = errors_HMM_tanner_cleaned;

end



% =================================
function [walk] = random_walk_create(size_par,H_dag,M)
   
 rng('shuffle');  % Seed the RNG based on the current time 

 walk_state = []; n = 1; teller1 = 1; 
 indeks1 = find(H_dag(:,n)==1);
 dominant_bit_pos = indeks1(1); % default start at top
 termination_test = 1000;
 while termination_test > 0 % only stop if all bits visited
     
    indeks1 = find(H_dag(:,n)==1);
    I1 = find(indeks1 == dominant_bit_pos);
    dummy = [1:size_par];
    dummy(I1) = [];
    dummy(randperm(size_par-1)) = dummy;  % random permutation of indeks

    I2 = dummy(1);  
    walk_state = [walk_state [n indeks1(I1) indeks1(I2)]'];  % parity check and bit positions
       
% bepaal sommer nou die volgende stap/pariteit toets
    dummy = find(H_dag(indeks1(I2),:)==1);
    pointer = find(dummy == n);  % which one is current time-stamp
    indeks_next = dummy;
    indeks_next(pointer) = []; % remove current time-stamp indeks 
    indeks_next(randperm(2)) = indeks_next;  
    n = indeks_next(1); % this is the next step of the random walk.
    dominant_bit_pos = indeks1(I2); % this must be the dominant state of next time stamp
    termination_test = length(setdiff(1:2*M, walk_state(2,:)));
     %  [teller1 termination_test]
    teller1 = teller1 + 1;
 end
 walk = walk_state;
end %end of function

