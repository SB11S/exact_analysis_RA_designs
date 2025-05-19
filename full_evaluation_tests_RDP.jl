# this code generates the rejection rates and UX critical values for the RDP design from Section 4 
# to get these results, run full_evaluation_tests_RDP(i, false , false) for i = 60, 240, 960
# after this, run the python file generate_results_comparison_tests.py with setting policy = 'RDP' at the top to generate the figure + table
using TickTock, Pickle, Distributions, SpecialFunctions, JLD# packages

function right_tail_fn(vals, w, signif)
    # find the right tail at level signif of the empirical distribution given by vals x weights 
   if all(w.==0) || length(vals)==1
       return(Inf)
   else
       if sum(w[vals.== maximum(vals)])/sum(w) > signif
           return(Inf)
       else
          idx = sortperm(vals, alg=QuickSort)
          vals_interest = vals[idx][(cumsum(w[idx])./sum(w).>=1-signif)]
          x = vals_interest[1]
          if sum(w[vals.>= x])/sum(w) > signif
              x = vals_interest[vals_interest.>x][1]
          end
        end
       @assert sum(w[vals.>= x])/sum(w) <= signif
       @assert (sum(w[vals.>= maximum(vals[vals.<x])])/sum(w) > signif)
       return x
  
   end
end
  
  function left_tail_fn(vals, w, signif)
    # find the left tail at level signif of the empirical distribution given by vals x weights 
   if all(w.==0) || length(vals)==1
       return(-Inf)
   else
       if sum(w[vals.== minimum(vals)])/sum(w) > signif
           return(-Inf)
       else
          idx = sortperm(vals, alg=QuickSort)
          vals_interest = vals[idx][(cumsum(w[idx])./sum(w).<=signif)]
          x = last(vals_interest)
          if sum(w[vals.<= x])/sum(w) > signif
              x = last(vals_interest[vals_interest.<x])
          end
        end
       return x
   end
end

function DP_2_lin_index( number_of_allocations , number_of_successes_arm_1 , number_of_failures_arm_1 , number_of_successes_arm_2 , number_of_remaining_allocations )
    # code from ''Jacko, P. (2019). BinaryBandit: An efficient Julia package for optimization and evaluation of the finite-horizon bandit problem with binary responses.''
    # This converts a 4D state to linear index
    # number_of_successes_arm_1 , number_of_failures_arm_1 , number_of_successes_arm_2 \ge 0
    # number_of_allocations , number_of_remaining_allocations \ge 1
    # number_of_successes_arm_1 + number_of_failures_arm_1 + number_of_successes_arm_2 + number_of_remaining_allocations \le number_of_allocations
        return Int64(div( number_of_allocations * ( number_of_allocations + 1 ) * ( number_of_allocations + 2 ) * ( number_of_allocations + 3 ) - ( number_of_allocations - number_of_remaining_allocations + 1 ) * ( number_of_allocations - number_of_remaining_allocations + 2 ) * ( number_of_allocations - number_of_remaining_allocations + 3 ) * ( number_of_allocations - number_of_remaining_allocations + 4 ) , 24 ) + div( ( number_of_allocations - number_of_remaining_allocations + 1 ) * ( number_of_allocations - number_of_remaining_allocations + 2 ) * ( number_of_allocations - number_of_remaining_allocations + 3 ) - ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 2 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 3 ) , 6 ) + div( ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 2 ) - ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 - number_of_failures_arm_1 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 - number_of_failures_arm_1 + 2 ) , 2 ) + number_of_successes_arm_1 + 1)
end

function apply_func_states(N::Int64, func::Function)
    # applies a given function to all states 
    # N is the trial horizion and func is the function to be applied
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   # # number of entries of g-vector 
    v:: Array{ Float64 , 1 } = zeros(num_states_N)
    for N_1::Int64 = 0 : N , S_1::Int64 = 0 :  N_1  , S_2::Int64 = 0 :  N - N_1
        @inbounds begin    
            v[idx_map(S_1, N_1, S_2, N)] = func(S_1, N_1, S_2, N) 
        end
    end
    return(v)
end

calc_RR = function(N,p_1, p_2, mults)
    # calculate the rejection rate for a vector mults of rejection indicators times policy coefficients
    # N is the trial horizon, p_1 the experimental and p_2 the control success probability, mults is the vector of g coefficients times indicator of rejection of a test for each state
    pval_func = function(S_1, N_1, S_2, N)
        return(p_1^S_1*(1-p_1)^(N_1-S_1)*p_2^S_2*(1-p_2)^(N - N_1 - S_2))
    end
  
    prob_coefs = apply_func_states(N, pval_func)
     return(sum(prob_coefs.*mults))
end

function idx_map(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # an index mapping less efficient but a bit easier to work with than the one in ''Jacko, P. (2019). BinaryBandit: An efficient
    #  Julia package for optimization and evaluation of the finite-horizon bandit problem with binary responses. can be made more efficient.''
    # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
    return(1 + S_1 + N_1*(N+1) + (1+(N + 2)*N)*S_2)
  end

function inv_idx_map(idx::Int64, N::Int64)
    # inverse of idx_map, idx is the index and N is the trial horizon
    S_2 = Int64(floor(idx/(N*(N+2)+1)))
    N_1 = Int64(floor((idx - (1+(N + 2)*N)*S_2)/(N+1)))
    S_1 = Int64(idx - 1 - (1+(N + 2)*N)*S_2 - N_1*(N+1))
      return S_1, N_1, S_2
end

function determine_policy(N::Int64, delta ::Float64, ell ::Int64, psi_0::Array{Float64} = Float64[1,1], kappa_0::Array{Float64} = Float64[2,2])
    # determine optimal policy + value for CRDP
    # N is the trial horizon, delta the probability of action adherence, ell the minimum treatment group size for all arms, psi_0 is the 
    # prior number of successes and kappa_0 is the prior number of allocations
    num_states_N= 1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2     # number of entries of value vector 
    num_states::Int64 = DP_2_lin_index(N, 0, 0, 0, N)    # number of entries policy vector
    Policy:: Vector{ UInt8}  = zeros(UInt8, num_states) # init policy vector 
    Values:: Array{ Float64,1} = [-N*((N_1 < ell) || (N - N_1 < ell)) for i = 1:num_states_N for N_1 =  inv_idx_map(i,N)[2]]  # init value vector
  
    for t in N-1:-1:0
      for  N_1::Int64 = 0 :  t, S_1::Int64 = 0 : N_1 , S_2::Int64 = 0 :  t - N_1
        @inbounds begin     
  
        #determining the probability of success in the arms
        Ep_1::Float64 = (S_1 + psi_0[1])/( N_1 + kappa_0[1])
        Ep_2::Float64 = (S_2 + psi_0[2])/( t-N_1 + kappa_0[2])
  
        # determining values of both arms
        V1 = Ep_1 * ( 1.0 + Values[ idx_map(S_1 + 1, N_1+1, S_2, N)] ) + ( 1.0 - Ep_1 ) * Values[ idx_map(S_1 , N_1+1, S_2, N)] 
        V2 = Ep_2 * ( 1.0 + Values[ idx_map(S_1 , N_1, S_2+1, N)] ) + ( 1.0 - Ep_2 ) * Values[ idx_map(S_1, N_1, S_2, N)]  
  
        # storing the optimal action and corresponding value
        if V1>V2 
          Values[idx_map(S_1,N_1,S_2,N) ] = delta*V1 +(1.0-delta)*V2
          Policy[DP_2_lin_index(N, S_1, N_1-S_1, S_2, N-t)] = 2
        elseif V1<V2
          Values[ idx_map(S_1,N_1,S_2,N) ] = delta*V2 +(1.0-delta)*V1
          Policy[DP_2_lin_index(N, S_1, N_1-S_1, S_2, N-t)] = 0
        else 
          Values[ idx_map(S_1,N_1,S_2,N) ] = V2
          Policy[DP_2_lin_index(N, S_1, N_1-S_1, S_2, N-t)] = 1
        end
      end
      end
    end
   return Policy, Values[1]
  end

function get_prob(A::UInt8, delta::Float64)
    # A is the action encoding in CRDP policy vector and delta the prob of adherence to the action
    # get the probability of assigning treatment 1 from the CRDP encoding
    if A == 0
        return(1-delta)
    elseif A == 2
        return(delta)
    else
        return(1/2)
    end
end

function determine_policy_coefs_CRDP(N::Int64, pol_vec::Array{UInt8}, delta::Float64) 
    #N is the trial horizon , pol_vec is the CRDP policy vector , delta is the probability of action adherence
    # this function determines policy-based coefficients g that occur in the distribution over endstates
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector
    a:: Array{ Float64 , 1 }  = zeros( Float64 , num_states_N)
    a[1] = 1 # probabiltiy of s0 = 1 is 1 initially
    for t in 1:1:N
        println(t)
        for N_1::Int64 = t :-1: 0 , S_1::Int64 = N_1 :-1:  0  , S_2::Int64 = t-N_1:-1:  0
            @inbounds begin  
            # for each state for next decision epoch, sum the coefs for possible previous states and multiply with the coef of the corresponding action
            a_val = 0
            if((S_2 <t-N_1)  & (N_1<t)) # failure on arm 2
                a_val += a[idx_map(S_1, N_1, S_2, N)]*(1-get_prob(pol_vec[DP_2_lin_index(N, S_1, N_1-S_1, S_2, N-t+1)], delta))
            end
            if(S_1 > 0) # success on arm 1
                a_val += a[idx_map(S_1-1, N_1-1, S_2, N)]*get_prob(pol_vec[DP_2_lin_index(N, S_1-1, N_1-S_1, S_2, N-t+1)], delta)
            end
            if((N_1 > 0) & (S_1 < N_1)) # failure on arm 1
                a_val += a[idx_map(S_1, N_1-1, S_2, N)]*get_prob(pol_vec[DP_2_lin_index(N, S_1, N_1-1-S_1, S_2, N-t+1)], delta)
            end
            if(S_2 > 0) # success on arm 2 
                a_val += a[idx_map(S_1, N_1, S_2-1, N)]*(1-get_prob(pol_vec[DP_2_lin_index(N, S_1, N_1-S_1, S_2-1, N-t+1)], delta))
            end
            a[idx_map(S_1, N_1, S_2, N) ]   = a_val # assign the coef value to the right place in the vector
            end
        end
    end
   return a
end

WS = function(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # determines the Agresti-Caffo adjusted Wald statistic for testing H_1: p_1, p_2 free vs. H_0: p_1 = p_2
    # S_a is the number of successes for arm a, N_1 is the number of patients allocated to arm 1.
    phat_1 = (S_1+1)/(N_1+2)
    phat_2 = (S_2+1)/(N-N_1+2)
    return (phat_1 - phat_2)/sqrt(phat_1*(1-phat_1)/(N_1+2) + phat_2*(1-phat_2)/(N-N_1+2) )   
end
  
function P_FET(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # calculate the hypergeometric probabilities used in Fisher's exact test
    # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
    return prod(factorial.(big.([S_1 + S_2, N - S_1 - S_2, N_1, N-N_1])))/prod(factorial.(big.([S_1, S_2, N_1-S_1, N-N_1-S_2, N])))
end
  
perc_N1 = function(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # percentage of participants allocated to group 1
    return(N_1/N)
end 

function determine_unconditional_CV(N::Int64, coefs::Array{Float64}, outc_stat::Array{Float64}, epsilon::Float64,
     signif_level::Float64, M::Float64  = Inf, lower = true)
    # determine an unconditional exact threshold  by using the Lipschitz property of the rejection rate function
    # N is the trial horizon , coefs are the g coefficients of a policy, outc_stat is a vector of test statistic values per state, epsilon is an absolute tolerance
    # signif_level is the targeted significance level, M is a maximum trial horizon after which we just determine the UX critical value on a grid

    #determine initial critical value and rejection rate for p = 0.5 
    p = 0.5
    pval_func = function(S_1, N_1, S_2, N)
        return(p^(S_1+ S_2)*(1-p)^(N-S_1-S_2))
    end
    prob_coefs = apply_func_states(N, pval_func) # calculate the probability coefs
    z = right_tail_fn(outc_stat, (prob_coefs.*coefs), signif_level ) # calculate the critical value
    RR = sum(prob_coefs.*coefs.*(outc_stat .>= z)) # calculate the rejection rate 
    idx_outc_rej = (1:length(outc_stat))[outc_stat .>= z] # get the indices of outc stat which lead to rejection

    if N <= M
      #determine a grid such that the maximum critical value for each point on this grid is the UX critical value
      dOP = 0.05 # distance between points where the lipschitz constant changes
      outer_points = dOP:dOP:1 # points where the lipschitz constant changes
  
      num_states_N=  1 + N + N*(N+1) + (1+(N + 2)*N)*N # number of states
  
      K = zeros(length(outer_points)) # Lipschitz constant for each of the outer points

      for j in 1:length(outer_points) 
          println(j/length(outer_points)) # print progress
          for i in idx_outc_rej
                inv_idx =  inv_idx_map(mod(i, num_states_N),N) # calculate the state corresponding to the index, where mod is taken to accound for more complex trial designs (e.g., real life examples)
                S =inv_idx[1]  +inv_idx[3]  # total sum of successes
                if (S>=1) & (N>=S+1)
                    est = (S-1)/(N-2)
                    if est > outer_points[j]
                        h = (outer_points[j] )^(S-1)*(1-(outer_points[j]))^(N-S-1)
                    elseif est < outer_points[j] - dOP
                        h = (outer_points[j] - dOP)^(S-1)*(1-(outer_points[j] - dOP))^(N-S-1)
                    else
                        h = est^(S-1)*(1-est)^(N-S-1)
                    end
                else
                    h = 1
                end
                K[j] += coefs[i]*h*max(abs(S - N*(outer_points[j] - dOP)), abs(S - N*outer_points[j]))
          end
        end
  
        dp = epsilon./K  # the grid spacing between outer points as determined by the Lipschitz bound
        println(dp)
        grid = []
        for j in 1:length(outer_points) 
            grid = vcat(grid, outer_points[j]-dOP:dp[j]:outer_points[j]) # add points to the Grid
        end
      #determine z value
    else # if N is too large, use equidistant grid
      grid = 0:0.01:1
    end

    # loop through the grid and determine the largest critical value
    k = 1 #index of the null probability
    p_argmax = 0.5
    RR_max = deepcopy(RR)
    while k <= length(grid)
        p = grid[k]
        println(p)
        pval_func = function(S_1, N_1, S_2, N)
            return(p^(S_1+ S_2)*(1-p)^(N-S_1-S_2))
        end
        prob_coefs = apply_func_states(N, pval_func)
        RR = sum(prob_coefs.*coefs.*(outc_stat .>= z)) # calculate the rejection rate for the null probability in the grid
        println(RR)
        println(z)
        if RR > RR_max 
            p_argmax = p
            RR_max = deepcopy(RR)
        end
        if  RR > signif_level-epsilon*(N<=M)*lower
            println("increase z")
            z = right_tail_fn(outc_stat, (prob_coefs.*coefs), signif_level-epsilon*(N<=M)*lower )
            RR = sum(prob_coefs.*coefs.*(outc_stat .>= z))
            RR_max = deepcopy(RR)
        end

        if N<=M
            k+= max(1,Int64(floor((signif_level-epsilon*(N<=M) -RR)/epsilon))) # if the current difference between RR and the target is larger than epsilon, we can skip some grid points
        else
            k+=1
        end
    end
    return([z, p_argmax, RR_max])
end
  

function get_rejection_rates_FET(N::Int64, policy_coefs::Array{Float64}, outc_FET::Array{Float64}, thr_exact_FET::Float64)
    # evaluates the rejection rate for an UX FET under a policy and UX threshold 
    # N is the trial horizon, policy_coefs the vector of g coefficients, outc_FET is the vector of FET p-values per state, thr_exact_FET is 
    # the UX threshold

    Nvals = [N]
    range_p2_small =vcat([0.0, 0.01, 0.05], 0.1:0.1:0.9, [0.95,0.99, 1.0])  # range of p2 (control prob) under the alternative (for power plots)

    range_p1 =  reduce(vcat, [x:0.01:1 for x in range_p2_small]) # range of p1 (experimental prob)
    range_p2 = reduce(vcat, [repeat([x], Int64(round((1-x)/0.01) + 1)) for x in range_p2_small]) # range p2 of same length of range p1 where each value of range_p2_small is repeated

    prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)] # keys for df for power plots
    res_dict_plot_alt = Dict([(((Nval, range_p1[i], range_p2[i]),  ("RR")), 0.0) for Nval in Nvals for i in 1:length(range_p1)  ]) # power plot dict

    range_p2 = 0:0.01:1 # range of control prob under the null
    prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)] # keys for df for null plots
    res_dict_plot_null = Dict([(((Nval, range_p2[i], range_p2[i]),  ("RR")), 0.0) for Nval in Nvals for i in 1:length(range_p2)  ]) # null plot dict

    for i in 1:length(Nvals)
        println(Nvals[i])

        vec_rejection_FET = policy_coefs.*( (outc_FET.<= thr_exact_FET)) # determine vector containing the indicator of rejection times the policy coefs
        outc_FET = nothing # remove as not needed anymore
 

        # determine operating characteristics under the alternative
        for idx_p in 1:length(prob_keys_alt) # loop over keys
            p_1 = prob_keys_alt[idx_p][1] # determine experimental prob succ
            p_2 = prob_keys_alt[idx_p][2] # determine control prob succ
            println("evaluation plots alt p_1 = "*string(p_1)*" p_2 = "*string(p_2)) # print something, so we know where we are
    
            res_dict_plot_alt[((N,p_1, p_2), ("RR" ))] =calc_RR(N,p_1, p_2, vec_rejection_FET) # calculate the rejection rate and put in dict for alternative
        end

        # determine operating characteristics under the null
        for idx_p in 1:length(prob_keys_null) # loop over keys
            p = prob_keys_null[idx_p][1] # probability of success under the null
            println("evaluation plots null p = "*string(p)) # print something, so we know where we are
            
            res_dict_plot_null[((N,p, p), ("RR" ))] =calc_RR(N,p, p, vec_rejection_FET) # calculate the rejection rate and put in dict for null
        end
    end

    res_eval_statistics = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)]) # put all dicts in one dict
    return(res_eval_statistics)
end

function evaluate_tests(N::Int64, policy_coefs::Array{Float64}, outc_stat::Array{Float64}, thr_exact_stat::Float64, signif_level = 0.05, store_intermediate = false)
    # evaluates the asymptotic, UX, CX-S and CX-SA tests given a vector of g coefficients, outcome statistics and UX threshold 
    # N is the trial horizon, policy_coefs the vector of g coefficients, outc_stat is the vector of test statistic outcomes per state, thr_exact_stat is 
    # the UX threshold, signif_level the significance level of the tests and store_intermediate is a Boolean indicating whether we want to have intermediate storage in case the code breaks due to memory issues.

    Nvals = [N]
    meas_keys = ["Asymp.", "Exact Uncond.","Exact Cond. (1M)" ,"Exact Cond. (2M)"] # different tests to evaluate
    range_p2_small =vcat([0.00, 0.01, 0.05], 0.1:0.1:0.9, [0.95,0.99,1.0])  # range of p2 (control prob) under the alternative (for power plots)

    range_p1 =  reduce(vcat, [x:0.01:1 for x in range_p2_small]) # range of p1 (experimental prob)
    range_p2 = reduce(vcat, [repeat([x], Int64(round((1-x)/0.01) + 1)) for x in range_p2_small]) # range p2 of same length of range p1 where each value of range_p2_small is repeated

    prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)] # keys for df for power plots
    res_dict_plot_alt = Dict([(((Nval, range_p1[i], range_p2[i]),  meas_key), 0.0) for Nval in Nvals for i in 1:length(range_p1)  for meas_key in meas_keys]) # power plot dict

    range_p2 = 0:0.01:1 # range of control prob under the null
    prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)] # keys for df for null plots
    res_dict_plot_null = Dict([(((Nval, range_p2[i], range_p2[i]),  meas_key), 0.0) for Nval in Nvals for i in 1:length(range_p2)  for meas_key in meas_keys]) # null plot dict

    tick()
    # determine the state indices for each total number of successes
    idx_S = []
    for S in 0:N
        idx_S = push!(idx_S , [idx_map(S_1, N_1, S - S_1, N) for N_1 = 0 : N for S_1 = max(S-N+N_1, 0) :  min(S, N_1) ])
    end

    # determine the exact conditional test threshold for each total number of successes
    num_states   = length(outc_stat)
    thr_1M_upper = zeros(num_states)
    thr_1M_lower = zeros(num_states)
    for S in 0:N
        thr_1M_upper[idx_S[S+1]] .=  right_tail_fn(outc_stat[idx_S[S+1]], ((policy_coefs)[idx_S[S+1]]), signif_level/2)
        thr_1M_lower[idx_S[S+1]]  .=  left_tail_fn(outc_stat[idx_S[S+1]], ((policy_coefs)[idx_S[S+1]]), signif_level/2)
    end

    # determine the state indices for each total number of successes and allocations to group 1
    idx_S_N = []
    for S in 0:N
        for N_1 in 0:N
            idx_S_N = push!(idx_S_N , [idx_map(S_1, N_1, S - S_1, N) for S_1 = max(S-N+N_1, 0) :  min(S, N_1) ])
        end
    end

    # determine the exact conditional test threshold for each total number of successes
    thr_2M_upper = zeros(num_states)
    thr_2M_lower = zeros(num_states)
    for S in 0:N
        for N_1 in 0:N
            idx = idx_S_N[S*(N+1)+N_1+1]
            thr_2M_upper[idx] .=  right_tail_fn(outc_stat[idx], (policy_coefs)[idx], signif_level/2)
            thr_2M_lower[idx] .=  left_tail_fn(outc_stat[idx], (policy_coefs)[idx], signif_level/2)
        end
    end
    println("done determining exact conditional thresholds")

    # determine for each state and test whether it rejects, and multiply by policy coefs
    vec_rejection_cond1M = policy_coefs.*((outc_stat.>=thr_1M_upper).|| (outc_stat.<=thr_1M_lower)  )
    vec_rejection_cond2M = policy_coefs.*((outc_stat.>=thr_2M_upper).|| (outc_stat.<=thr_2M_lower)  )
    vec_rejection_asymp = policy_coefs.*((outc_stat.>=quantile(Normal(),  1-signif_level/2)) .|| (outc_stat.<=quantile(Normal(),  signif_level/2)))
    vec_rejection_uncond = policy_coefs.*((outc_stat.>=thr_exact_stat) .|| (outc_stat.<= -1 .*thr_exact_stat))
    vec_EPASA = policy_coefs.*apply_func_states(N, perc_N1)
    dict_tests = Dict([# put in dict for iteration
        ("Asymp.",vec_rejection_asymp),
        ( "Exact Uncond.",vec_rejection_uncond),
        ( "Exact Cond. (1M)",vec_rejection_cond1M),
        ( "Exact Cond. (2M)",vec_rejection_cond2M)
        ])

    policy_coefs = nothing# remove as not needed anymore
    thr_1M_upper = nothing# remove as not needed anymore
    thr_1M_lower = nothing# remove as not needed anymore
    thr_2M_upper = nothing# remove as not needed anymore
    thr_2M_lower = nothing# remove as not needed anymore
    
    # determine RR under alternative
    for idx_p in 1:length(prob_keys_alt)
        p_1 = prob_keys_alt[idx_p][1] # success probability experimental
        p_2 = prob_keys_alt[idx_p][2] # success probability null 
        println("evaluation alt p_1 = "*string(p_1)*" p_2 = "*string(p_2))
        
        #calculate rejection rates of asymptotic, UX, CX-S and CX-SA tests under alternative
        for test in collect(keys(dict_tests))
            res_dict_plot_alt[((N,p_1, p_2), test)]   = calc_RR(N,p_1, p_2, dict_tests[test])
        end

        #add EPASA
        res_dict_plot_alt[((N,p_1, p_2), "EPASA")]   = calc_RR(N,p_1, p_2, vec_EPASA)

        if (mod(idx_p,25)==0) & store_intermediate # store intermediate (safety guard for if code crashes)
            res_eval_tests = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
            Pickle.store("eval_tests_RDP_N="*string(N)*".pkl", res_eval_tests)
        end
    end

    # determine RR under null
     for idx_p in 1:length(prob_keys_null)
            p = prob_keys_null[idx_p][1]
            println("evaluation null p_= "*string(p))
            
            #calculate rejection rates of asymptotic, UX, CX-S and CX-SA tests under null
            for test in collect(keys(dict_tests))
                res_dict_plot_null[((N,p, p), test)]   = calc_RR(N,p, p, dict_tests[test])
            end

            if (mod(idx_p,25)==0) & store_intermediate # store intermediate (safety guard for if code crashes)
                res_eval_tests = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
                Pickle.store("eval_tests_RDP_N="*string(N)*".pkl", res_eval_tests)
            end
    end

    return( Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)]))
end

full_evaluation_tests_RDP = function(N, save_intermediate, read_UXCV = false) # N is the trial horizon, save_intermediate stores intermediate results
    #### Constants ####
    psi_0   = [1.,1.] # prior number of successes for both arms
    kappa_0 = [2.,2.]  # prior number of pulls for both arms
    delta   = 0.9  # probability of adherence to arm assignment
    ellfreq = 0.0
    ell     = Int64(ceil(ellfreq*N)) # minimum number of assignments
    ##########################

    # calculate RDP policy 
    tick()
    res_RDP = determine_policy(N, delta, ell , psi_0, kappa_0)
    time_policy = tok()
    polvec_RDP = res_RDP[1]
    res_RDP = nothing  # remove, not needed anymore


    # calculate policy coefs
    tick()
    coefs_RDP = determine_policy_coefs_CRDP(N, polvec_RDP, delta) 
    time_calc_coefs = tok()
    polvec_RDP = nothing # remove, not needed anymore

    tick()
    res_WS = @timed apply_func_states(N, WS) # calculate the wald statistic for every state
    time_calc_WS = tok()
    outc_WS = res_WS[1]
   
    tick()
    vec_probs_states = apply_func_states(N, P_FET) 
    function FET(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
        # calculate the P-value for Fisher's exact test 
        # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
        prob_cur = vec_probs_states[idx_map(S_1, N_1, S_2, N)] # get the hypergeometric probability for current state
        prob_other = [vec_probs_states[idx_map(succ1, N_1, S_1 + S_2 - succ1, N)] for succ1 = max(0, N_1 + S_1 + S_2 - N) : min(N_1, S_1 + S_2) ]# get the hypergeometric probability for other states with same amount of successes
        sum(prob_other[prob_other .<= prob_cur]) # obtain the p - value 
    end
    outc_FET = apply_func_states(N, FET) # calculate the FET p-values for every state
    time_calc_FET_stat = tok()

    # determine the unconditional thresholds for the Wald test and FET
    sig_level = 0.05
    
    if read_UXCV # if we read the ux crit. val
        res_thr_exact_WS = JLD.load("thresholds/res_UX_CV_WS_RDP_N="*string(N)*".jld")["results"]   # get uconditional exact threshold for the wald statistic, remove 600 if we want to determine true UX thr for trial sizes >= 600
        time_calc_uxcv_WS = Inf
        thr_exact_WS = res_thr_exact_WS[1][1]

        res_thr_exact_FET = JLD.load("thresholds/res_UX_CV_FET_RDP_N="*string(N)*".jld")["results"]  
        time_calc_uxcv_FET = Inf
        thr_exact_FET = min(0.05, -res_thr_exact_FET[1][1]) 
    else
        tick()
        res_thr_exact_WS =  @timed determine_unconditional_CV(N, coefs_RDP, outc_WS, 0.00005, sig_level/2, 600.0) # get uconditional exact threshold for the wald statistic, remove 600 if we want to determine true UX thr for trial sizes >= 600
        time_calc_uxcv_WS = tok()
        thr_exact_WS = res_thr_exact_WS[1][1]
        JLD.save("thresholds/res_UX_CV_WS_RDP_N="*string(N)*".jld", "results", res_thr_exact_WS)    

        tick()
        res_thr_exact_FET =  @timed determine_unconditional_CV(N, coefs_RDP, -outc_FET, 0.00005, sig_level, 600.0) # get uconditional exact threshold for the FET, remove 600 if we want to determine true UX thr for trial sizes >= 600
        thr_exact_FET = min(0.05, -res_thr_exact_FET[1][1])# take minimum so that FET is only corrected for type I error 
        time_calc_uxcv_FET = tok()
        JLD.save("thresholds/res_UX_CV_FET_RDP_N="*string(N)*".jld", "results", res_thr_exact_FET)    
    end
   

    tick()
    res_RR_FET = get_rejection_rates_FET(N, coefs_RDP, outc_FET, thr_exact_FET)
    time_calc_res_FET = tok()
    Pickle.store("data/RR_FET_RDP_N="*string(N)*".pkl", res_RR_FET) # save dict of results for FET under RDP

    tick()
    res_eval_tests = evaluate_tests(N, coefs_RDP, outc_WS, thr_exact_WS, sig_level, save_intermediate)
    time_calc_res_WS = tok()
    res_eval_tests = Dict([("res_mats_null", res_eval_tests["res_mats_null"]), ("res_mats_alt", res_eval_tests["res_mats_alt"]),
     ("times", [time_policy,time_calc_coefs,time_calc_WS,time_calc_FET_stat,time_calc_uxcv_WS,time_calc_uxcv_FET,time_calc_res_FET,time_calc_res_WS])])
    Pickle.store("data/eval_tests_RDP_N="*string(N)*".pkl", res_eval_tests) # save dict of results for WS under RDP 
end

full_evaluation_tests_RDP(60, false, false) # argument is the trial size, set third element to true to load the UX crit. val for FET and Wald (faster), note that this should be in the same folder

