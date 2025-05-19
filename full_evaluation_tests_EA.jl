# this code generates the rejection rates and UX critical values for the EA design shown in the supplement 
# to get these results, run full_evaluation_tests_EA(i, false , false) for i = 60, 240, 960
# after this, run the python file generate_results_comparison_tests.py with setting policy = 'EA' at the top to generate the figure + table
using TickTock, Pickle, Distributions, SpecialFunctions, JLD# packages

function right_tail_fn(vals, w, signif)
    # find the right tail at level signif of the empirical distribution given by vals x weights 
   if all(w.==0) || length(vals)==1
       return(Inf)
   else
       if sum(w[vals.== maximum(vals)])/sum(w) > signif
           return(Inf)
       else
          idx = sortperm(vals)
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
          idx = sortperm(vals)
          vals_interest = vals[idx][(cumsum(w[idx])./sum(w).<=signif)]
          x = last(vals_interest)
          if sum(w[vals.<= x])/sum(w) > signif
              x = last(vals_interest[vals_interest.<x])
          end
        end
       return x
   end
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
  

function determine_unconditional_CV(N::Int64, coefs::Array{Float64}, outc_stat::Array{Float64}, epsilon::Float64, signif_level::Float64, M::Float64  = Inf)
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
        if  RR > signif_level-epsilon*(N<=M)
            println("increase z")
            z = right_tail_fn(outc_stat, (prob_coefs.*coefs), signif_level-epsilon*(N<=M) )
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
    range_p2_small =vcat([0.0, 0.01, 0.05], 0.1:0.1:0.9, [0.95,0.99, 1.0]) # range of p2 (control prob) under the alternative (for power plots)

    range_p1 =  reduce(vcat, [x:0.01:1 for x in range_p2_small]) # range of p1 (experimental prob)
    range_p2 = reduce(vcat, [repeat([x], Int64(round((1-x)/0.01) + 1)) for x in range_p2_small]) # range p2 of same length of range p1 where each value of range_p2_small is repeated

    prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)] # keys for df for power plots
    res_dict_plot_alt = Dict([(((Nval, range_p1[i], range_p2[i]),  ("RR")), 0.0) for Nval in Nvals for i in 1:length(range_p1)  ]) # power plot dict

    range_p2 = 0:0.01:1 # range of control prob under the null
    prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)] # keys for df for null plots
    res_dict_plot_null = Dict([(((Nval, range_p2[i], range_p2[i]),  ("RR")), 0.0) for Nval in Nvals for i in 1:length(range_p2)  ]) # null plot dict

    for i in 1:length(Nvals)
        println(Nvals[i])

        vec_rejection_FET = policy_coefs.*( (outc_FET.<= 0.05)) # determine vector containing the indicator of rejection times the policy coefs
        vec_rejection_BT = policy_coefs.*( (outc_FET.<= thr_exact_FET)) # determine vector containing the indicator of rejection times the policy coefs
        outc_FET = nothing # remove as not needed anymore
 

        # determine operating characteristics under the alternative
        for idx_p in 1:length(prob_keys_alt) # loop over keys
            p_1 = prob_keys_alt[idx_p][1] # determine experimental prob succ
            p_2 = prob_keys_alt[idx_p][2] # determine control prob succ
            println("evaluation plots alt p_1 = "*string(p_1)*" p_2 = "*string(p_2)) # print something, so we know where we are
    
            res_dict_plot_alt[((N,p_1, p_2), ("RR" ))] =calc_RR(N,p_1, p_2, vec_rejection_FET) # calculate the rejection rate and put in dict for alternative
            res_dict_plot_alt[((N,p_1, p_2), ("RR BT" ))] =calc_RR(N,p_1, p_2, vec_rejection_BT) # calculate the rejection rate BT and put in dict for alternative

        end

        # determine operating characteristics under the null
        for idx_p in 1:length(prob_keys_null) # loop over keys
            p = prob_keys_null[idx_p][1] # probability of success under the null
            println("evaluation plots null p = "*string(p)) # print something, so we know where we are
             
            res_dict_plot_null[((N,p, p), ("RR" ))] =calc_RR(N,p, p, vec_rejection_FET) # calculate the rejection rate and put in dict for null
            res_dict_plot_null[((N,p, p), ("RR BT" ))] =calc_RR(N,p, p, vec_rejection_BT) # calculate the rejection rate BT and put in dict for alternative

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
    range_p2_small =vcat([0.0, 0.01, 0.05], 0.1:0.1:0.9, [0.95,0.99,1.0])  # range of p2 (control prob) under the alternative (for power plots)

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
        res_dict_plot_alt[((N,p_1, p_2), ("Asymp."))]            = calc_RR(N,p_1, p_2, vec_rejection_asymp)
        res_dict_plot_alt[((N,p_1, p_2), ( "Exact Uncond." ))]   = calc_RR(N,p_1, p_2, vec_rejection_uncond)
        res_dict_plot_alt[((N,p_1, p_2), ( "Exact Cond. (1M)"))] = calc_RR(N,p_1, p_2, vec_rejection_cond1M)
        res_dict_plot_alt[((N,p_1, p_2), ("Exact Cond. (2M)"))]  = calc_RR(N,p_1, p_2, vec_rejection_cond2M)

        if (mod(idx_p,25)==0) & store_intermediate # store intermediate (safety guard for if code crashes)
            res_eval_tests = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
            Pickle.store("eval_tests_EA_N="*string(N)*".pkl", res_eval_tests)
        end
    end

    # determine RR under null
    for idx_p in 1:length(prob_keys_null)
            p = prob_keys_null[idx_p][1]
            println("evaluation null p_= "*string(p))
            
            #calculate rejection rates of asymptotic, UX, CX-S and CX-SA tests under null
            res_dict_plot_null[((N,p, p), ("Asymp."))] = calc_RR(N,p, p, vec_rejection_asymp)
            res_dict_plot_null[((N,p, p), ( "Exact Uncond." ))] =calc_RR(N,p, p, vec_rejection_uncond)
            res_dict_plot_null[((N,p, p), ( "Exact Cond. (1M)"))] =calc_RR(N,p, p, vec_rejection_cond1M)
            res_dict_plot_null[((N,p, p), ("Exact Cond. (2M)"))] = calc_RR(N,p, p, vec_rejection_cond2M)

            if (mod(idx_p,25)==0) & store_intermediate # store intermediate (safety guard for if code crashes)
                res_eval_tests = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
                Pickle.store("eval_tests_EA_N="*string(N)*".pkl", res_eval_tests)
            end
    end

    return( Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)]))
end

full_evaluation_tests_EA = function(N, save_intermediate, load_UXCV = false) # N is the trial horizon, save_intermediate stores intermediate results
 
    tick()
    # calculate policy coefs for EA
    coefs_EA =  zeros( 1 + N + N*(N+1) + (1+(N + 2)*N)*N )
    for S_1 = Int64(floor(N/2)) :-1:  0  , S_2 = Int64(ceil(N/2)):-1:  0
        N_1 = Int64(floor(N/2))
        coefs_EA[idx_map(S_1, N_1, S_2, N)] = exp(loggamma(N_1+1)-loggamma(S_1+1)-loggamma(N_1-S_1+1) +
        loggamma(N-N_1+1)- loggamma(S_2+1) - loggamma(N-N_1-S_2+1))
    end
    if !iseven(N) 
        coefs_EA = coefs_EA./2
        for S_1 = Int64(ceil(N/2)) :-1:  0  , S_2 = Int64(floor(N/2)):-1:  0
            N_1 = Int64(ceil(N/2))
            coefs_EA[idx_map(S_1, N_1, S_2, N)] = exp(loggamma(N_1+1)-loggamma(S_1+1)-loggamma(N_1-S_1+1) +
            loggamma(N-N_1+1)- loggamma(S_2+1) - loggamma(N-N_1-S_2+1))/2
        end
    end
    time_calc_coefs = tok()

    tick()
    res_WS = @timed apply_func_states(N, WS) # calculate the wald statistic for every state, complexity O(N^3)
    time_calc_WS =tok()
    outc_WS = res_WS[1]

    tick()
    vec_probs_states = apply_func_states(N, P_FET) #complexity O(N^3)
    function FET(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
        # complexity O(min(N_1, S_1 + S_2) - max(0, N_1 + S_1 + S_2 - N))
        # calculate the P-value for Fisher's exact test 
        # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
        prob_cur = vec_probs_states[idx_map(S_1, N_1, S_2, N)] # get the hypergeometric probability for current state
        prob_other = [vec_probs_states[idx_map(succ1, N_1, S_1 + S_2 - succ1, N)] for succ1 = max(0, N_1 + S_1 + S_2 - N) : min(N_1, S_1 + S_2) ]# get the hypergeometric probability for other states with same amount of successes
        sum(prob_other[prob_other .<= prob_cur]) # obtain the p - value 
    end
    outc_FET = apply_func_states(N, FET) # calculate the FET p-values for every state,  #complexity O(N^4)
    time_calc_FET_stat = tok()

    # determine the unconditional thresholds for the Wald test and FET
    sig_level = 0.05
    if load_UXCV
        res_thr_exact_WS = JLD.load("thresholds/res_UX_CV_WS_EA_N="*string(N)*".jld")["results"]   # get uconditional exact threshold for the wald statistic, remove 600 if we want to determine true UX thr for trial sizes >= 600
        time_calc_uxcv_WS = Inf
        thr_exact_WS = res_thr_exact_WS[1][1]

        res_thr_exact_FET = JLD.load("thresholds/res_UX_CV_FET_EA_N="*string(N)*".jld")["results"]   # get uconditional exact threshold for the wald statistic, remove 600 if we want to determine true UX thr for trial sizes >= 600
        time_calc_uxcv_FET = Inf
        thr_exact_FET = -res_thr_exact_FET[1][1]
    else
        tick()
        res_thr_exact_WS =  @timed determine_unconditional_CV(N, coefs_EA, outc_WS, 0.00005, sig_level/2, 250.0) # get uconditional exact threshold for the wald statistic, remove 600 if we want to determine true UX thr for trial sizes >= 600
        time_calc_uxcv_WS = tok()
        thr_exact_WS = res_thr_exact_WS[1][1]
        JLD.save("thresholds/res_UX_CV_WS_EA_N="*string(N)*".jld", "results", res_thr_exact_WS)

        tick()
        res_thr_exact_FET =  @timed determine_unconditional_CV(N, coefs_EA, -outc_FET, 0.00005, sig_level, 250.0) # get uconditional exact threshold for the FET, remove 600 if we want to determine true UX thr for trial sizes >= 600
        thr_exact_FET = -res_thr_exact_FET[1][1]
        time_calc_uxcv_FET = tok()
        JLD.save("thresholds/res_UX_CV_FET_EA_N="*string(N)*".jld", "results", res_thr_exact_FET)    
    end

    tick()
    res_RR_FET = get_rejection_rates_FET(N, coefs_EA, outc_FET, thr_exact_FET)
    time_calc_res_FET = tok()
    Pickle.store("data/RR_FET_EA_N="*string(N)*".pkl", res_RR_FET) # save dict of results for FET under EA
  

    tick()
    res_eval_tests = evaluate_tests(N, coefs_EA, outc_WS, thr_exact_WS, sig_level, save_intermediate)
    time_calc_res_WS = tok()
    res_eval_tests = Dict([("res_mats_null", res_eval_tests["res_mats_null"]), ("res_mats_alt", res_eval_tests["res_mats_alt"]),
    ("times", [time_calc_coefs,time_calc_WS,time_calc_FET_stat,time_calc_uxcv_WS,time_calc_res_FET,time_calc_res_WS])])
    Pickle.store("data/eval_tests_EA_N="*string(N)*".pkl", res_eval_tests) # save dict of results for FET under EA
end

full_evaluation_tests_EA(60, false , false) # first argument is the trial size,  set third element to true to load the UX crit. val for Wald and FET (faster), note that this should be in the same folder


