function [A, B, P, PI] = initialize(B_init, D, K, M)
%function [A,B,P,PI,Vk,MO,K]=hsmmInitialize(MO,M,D,K,MT)
% 
% Author: Shun-Zheng Yu
% Available: http://sist.sysu.edu.cn/~syu/Publications/hsmmInitialize.m.txt
%
% To initialize the matrixes of A,B,P,PI for hsmm_new.m, to get 
% the observable values and to transform the observations O
% from values to their indexes.
%
% Usage: [A,B,P,PI,Vk,O,K]=hsmmInitialize(O,M,D,K)
% 
%  N:  Number of observation sequences
%  MT: Lengths of the observation sequences: MT=(T_1,...,T_N)
%  MO: Set of the observation sequences: MO=[O^1,...,O^N], O^n is the n'th obs seq.
%  M:  total number of states
%  D:  maximum duration of states
%  K:  total number of observable values
%
%  A: initial values of state transition probability matrix
%  B: initial values of observation probability matrix
%  P: initial values of state duration probability matrix
%  PI: initial values of starting state probability vector
%  Vk: the set of observable values
%
%  
%  Updated Nov.2014
%
%

% initial sub-state probabilities
PI=rand(M,1)+0.001;
A=rand(M)+0.001;
B = B_init;

% normalize
PI=PI./sum(PI);              %starting state probabilities
A=A./(sum(A')'*ones(1,M));
B=B./(sum(B')'*ones(1,K));      
P=repmat((1:D).^2,M,1);         % let long duration have higher init prob
P=double(P)./(sum(P')'*ones(1,D));
