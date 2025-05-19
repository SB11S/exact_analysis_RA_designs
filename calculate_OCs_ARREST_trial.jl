# generate the results for the ARREST trial described in the paper
# after running this script, run the python file generate_results_arrest.py
using  DSP, RCall, QuadGK, JLD, Distributions, StatsBase, TickTock, SpecialFunctions

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

function right_tail_fn(vals, w, signif)
    # find the right tail at level signif of the empirical distribution given by vals x weights 
   if all(w.==0) || length(vals)==1
       return(Inf)
   else
       # x = minimum(vals[vals .>= -quantile(-vals, weights(w), signif)])
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

function idx_map(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # an index mapping less efficient but a bit easier to work with than the one in ''Jacko, P. (2019). BinaryBandit: An efficient
    #  Julia package for optimization and evaluation of the finite-horizon bandit problem with binary responses. can be made more efficient.''
    # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
    return(1 + S_1 + N_1*(N+1) + (1+(N + 2)*N)*S_2)
  end

function floorfn(x)
  return(ceil(x)-1)
end
function determine_unconditional_CV(N::Int64, coefs_max::Array{Float64}, discr_max::Array{Float64}, epsilon::Float64,
    signif_level::Float64, grid = [])

    num_states_N = 1 + N + N*(N+1) + (1+(N + 2)*N)*N
    maxvals = []
    for i in 1:length(discr_max)
        maxvals = vcat(maxvals, ones(num_states_N)*discr_max[i])
    end

    # determine initial crit. val at p=0.5
    if length(grid) == 0
        p = 0.5
    else
        p = median(grid)
    end
    outc_stat = maxvals
    pval_func = function(S_1, N_1, S_2, N)
        return(p^(S_1+ S_2)*(1-p)^(N-S_1-S_2))
    end
    prob_coefs = apply_func_states(N, pval_func)
    prob_coefs = repeat(prob_coefs, length(discr_max))
    z = right_tail_fn(outc_stat, (prob_coefs.*coefs_max), signif_level/2 )
    RR = sum(prob_coefs.*coefs_max.*(maxvals .>= z))

    # determine which coefs to take into account by only restricting to those where the maximum is less than z
    i_cur_thr = length(discr_max[discr_max.<z])
    coefs_thr = reduce(+,[coefs_max[Int64(1 +(k-1)*num_states_N) : Int64(k*num_states_N)]  for k in 1:i_cur_thr])
    coefs_not_thr = reduce(+,[coefs_max[Int64(1 +(k-1)*num_states_N) : Int64(k*num_states_N)]  for k in i_cur_thr+1:length(discr_max)])

    #determine a grid such that the maximum critical value for each point on this grid is the UX critical value
    if length(grid) == 0
        dOP = 0.05 # distance between points where the lipschitz constant changes
        outer_points = dOP:dOP:1 # points where the lipschitz constant changes
        res = zeros(length(outer_points))
        derivbd = function(S_1,N_1,S_2, N, ub)
            S = S_1  +S_2
            if (S>=1) & (N>=S+1)
                est = (S-1)/(N-2)
                if est > ub
                    h = (ub )^(S-1)*(1-(ub))^(N-S-1)
                elseif est < ub - dOP
                    h = (ub - 0.1)^(S-1)*(1-(ub- dOP))^(N-S-1)
                else
                    return(est^(S-1)*(1-est)^(N-S-1)*max(abs(S - N*(ub- dOP)), abs(S - N*ub)))
                end
            else
                return(max(abs(S - N*(ub- dOP)), abs(S - N*ub)))
            end
        end
        for j in 1:length(outer_points)
            derivbd_spec = function(S_1,N_1,S_2,N)
                return(derivbd(S_1,N_1,S_2,N, outer_points[j]))
            end

            DB = apply_func_states(N, derivbd_spec)
            res[j] = sum(coefs_not_thr.*DB)
        end

        dp = epsilon./res
        println(dp)
        grid = []
        for j in 1:length(outer_points) 
            grid = vcat(grid, outer_points[j]-dOP:dp[j]:outer_points[j])
        end
    end

    # loop through the grid and determine the largest critical value
    k = 1 #index of the null probability
    p_argmax = 0.5
    RR_max = deepcopy(RR)
    p_argmax = p
    while k <= length(grid)
        p = grid[k]
        println(p)
        pval_func = function(S_1, N_1, S_2, N)
            return(p^(S_1+ S_2)*(1-p)^(N-S_1-S_2))
        end
        prob_coefs = apply_func_states(N, pval_func)
        RR = 1-sum(prob_coefs.*coefs_thr)
        if RR> RR_max
            p_argmax = p
        end
        RR_max = max(RR, RR_max)

        println(RR)
        println(z)
        if  RR > signif_level/2-epsilon
            println("increase z")
            while RR> signif_level/2-epsilon
                z = discr_max[i_cur_thr+2]
                i_cur_thr = length(discr_max[discr_max.<z])
                coefs_thr = reduce(+,[coefs_max[Int64(1 +(k-1)*num_states_N) : Int64(k*num_states_N)]  for k in 1:i_cur_thr])
                RR = 1-sum(prob_coefs.*coefs_thr)
            end
        end
        k+= max(1,Int64(floor((signif_level/2-epsilon -RR)/epsilon)))
    end
    return([z, p_argmax, RR_max])
end

function determine_policy_coefs_TS_OS(N,decision_epochs, OSB, policy::Array{Float64}) 
    # determine policy-based coefficients that occur in the distribution over endstates
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
    a:: Array{ Float64 , 1 }  = zeros( Float64 , (length(decision_epochs))*num_states_N)
    a[1] = 1 # probabiltiy of s0 = 1 initially
    for i in 1:length(decision_epochs)-1
        t = decision_epochs[i]
        println(i/(length(decision_epochs)-1))
        dDE =  decision_epochs[i+1]-t
        for N_1::Int64 = 0 :1: t , S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:t-N_1 
            if a[idx_map(S_1, N_1, S_2, N)+ (i-1)*num_states_N]>0 
                prob_superior = policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1-t)]
                if !((prob_superior >= OSB) || (1-prob_superior >= OSB)  )
                    allocprob = min(0.75, max(0.25, prob_superior))
                    vec_dN1 = [Int64(floorfn(allocprob*dDE)), Int64(ceil(allocprob*dDE))]
                    probs_dN1 = [ceil(allocprob*dDE)- allocprob*dDE   , allocprob*dDE - floorfn(allocprob*dDE)]  
                    for i_dN_1::Int64 = [1,2] , dS_1::Int64 = 0:1:vec_dN1[i_dN_1]  , dS_2::Int64 = 0:1:dDE-vec_dN1[i_dN_1]      
                        prob_dN1 = probs_dN1[i_dN_1]
                        if prob_dN1>0 
                            dN_1 = vec_dN1[i_dN_1]
                            a[idx_map(S_1+dS_1, N_1+dN_1, S_2+dS_2, N)+i*num_states_N] += binomial(dN_1, dS_1)*binomial(dDE-dN_1,dS_2)*a[idx_map(S_1, N_1, S_2, N)+(i-1)*num_states_N ]*prob_dN1                           
                        end
                    end
                end
            end
        end
    end
   return a
end

function determine_policy_coefs_TS_max(N,decision_epochs, discr_max, policy::Array{Float64}) 
    # determine policy-based coefficients that occur in the distribution over endstates
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
    a:: Array{ Float64 , 1 }  = zeros( Float64 , length(discr_max)*num_states_N)
    a[1] = 1 # probabiltiy of s0 = 1 initially
    for i in 1:length(decision_epochs)-1
        t = decision_epochs[i]
        println(i/(length(decision_epochs)-1))
        dDE =  decision_epochs[i+1]-t
        a_copy = deepcopy(a)
        a*=0
        for N_1::Int64 = 0 :1: t 
            println(N_1/t)
            for S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:t-N_1, idx_max = 1:length(discr_max)
            if a_copy[idx_map(S_1, N_1, S_2, N)+ (idx_max-1)*num_states_N]>0 
                prob_superior = policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1-t)]
                allocprob = min(0.75, max(0.25, prob_superior))
                vec_dN1 = [Int64(floorfn(allocprob*dDE)), Int64(ceil(allocprob*dDE))]
                probs_dN1 = [ceil(allocprob*dDE)- allocprob*dDE   , allocprob*dDE - floorfn(allocprob*dDE)]  
                for i_dN_1::Int64 = [1,2] , dS_1::Int64 = 0:1:vec_dN1[i_dN_1]  , dS_2::Int64 = 0:1:dDE-vec_dN1[i_dN_1]      
                    prob_dN1 = probs_dN1[i_dN_1]
                    if prob_dN1>0 
                        dN_1 = vec_dN1[i_dN_1]
                        new_prob_superior = max(discr_max[idx_max],policy[DP_2_lin_index(N+1, S_1+dS_1, N_1+dN_1 - S_1 - dS_1, S_2+dS_2, N+1-t-dDE)])
                        idx_max_new = last(findall(new_prob_superior .>= discr_max))
                        a[idx_map(S_1+dS_1, N_1+dN_1, S_2+dS_2, N) + (idx_max_new-1)*num_states_N ] += binomial(dN_1, dS_1)*binomial(dDE-dN_1,dS_2)*a_copy[idx_map(S_1, N_1, S_2, N) + (idx_max-1)*num_states_N]*prob_dN1             
                    end
                end
            end
            end
        end
    end
   return a
end


function determine_policy_coefs_TS_min(N,decision_epochs, discr_min, policy::Array{Float64}) 
    # determine policy-based coefficients that occur in the distribution over endstates
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
    a:: Array{ Float64 , 1 }  = zeros( Float64 , length(discr_min)*num_states_N)
    a[1 + (length(discr_min)-1)*num_states_N] = 1 # probabiltiy of s0 = 1 initially
    for i in 1:length(decision_epochs)-1
        t = decision_epochs[i]
        println(i/(length(decision_epochs)-1))
        dDE =  decision_epochs[i+1]-t
        a_copy = deepcopy(a)
        a*=0
        for N_1::Int64 = 0 :1: t 
            println(N_1/t)
            for S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:t-N_1, idx_min = 1:length(discr_min)
            if a_copy[idx_map(S_1, N_1, S_2, N)+ (idx_min-1)*num_states_N]>0 
                prob_superior = policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1-t)]
                allocprob = min(0.75, max(0.25, prob_superior))
                vec_dN1 = [Int64(floorfn(allocprob*dDE)), Int64(ceil(allocprob*dDE))]
                probs_dN1 = [ceil(allocprob*dDE)- allocprob*dDE   , allocprob*dDE - floorfn(allocprob*dDE)]  
                for i_dN_1::Int64 = [1,2] , dS_1::Int64 = 0:1:vec_dN1[i_dN_1]  , dS_2::Int64 = 0:1:dDE-vec_dN1[i_dN_1]      
                    prob_dN1 = probs_dN1[i_dN_1]
                    if prob_dN1>0 
                        dN_1 = vec_dN1[i_dN_1]
                        new_prob_superior = min(discr_min[idx_min], policy[DP_2_lin_index(N+1, S_1+dS_1, N_1+dN_1 - S_1 - dS_1, S_2+dS_2, N+1-t-dDE)])
                        idx_min_new = sort(findall(new_prob_superior .<= discr_min))[1]
                        a[idx_map(S_1+dS_1, N_1+dN_1, S_2+dS_2, N) + (idx_min_new-1)*num_states_N ] += binomial(dN_1, dS_1)*binomial(dDE-dN_1,dS_2)*a_copy[idx_map(S_1, N_1, S_2, N) + (idx_min-1)*num_states_N]*prob_dN1             
                    end
                end
            end
            end
        end
    end
   return a
end
function DP_2_lin_index( number_of_allocations , number_of_successes_arm_1 , number_of_failures_arm_1 , number_of_successes_arm_2 , number_of_remaining_allocations )
    # code from ''Jacko, P. (2019). BinaryBandit: An efficient Julia package for optimization and evaluation of the finite-horizon bandit problem with binary responses.''
    # This converts a 4D state to linear index
    # number_of_successes_arm_1 , number_of_failures_arm_1 , number_of_successes_arm_2 \ge 0
    # number_of_allocations , number_of_remaining_allocations \ge 1
    # number_of_successes_arm_1 + number_of_failures_arm_1 + number_of_successes_arm_2 + number_of_remaining_allocations \le number_of_allocations
        return Int64(div( number_of_allocations * ( number_of_allocations + 1 ) * ( number_of_allocations + 2 ) * ( number_of_allocations + 3 ) - ( number_of_allocations - number_of_remaining_allocations + 1 ) * ( number_of_allocations - number_of_remaining_allocations + 2 ) * ( number_of_allocations - number_of_remaining_allocations + 3 ) * ( number_of_allocations - number_of_remaining_allocations + 4 ) , 24 ) + div( ( number_of_allocations - number_of_remaining_allocations + 1 ) * ( number_of_allocations - number_of_remaining_allocations + 2 ) * ( number_of_allocations - number_of_remaining_allocations + 3 ) - ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 2 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 3 ) , 6 ) + div( ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 + 2 ) - ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 - number_of_failures_arm_1 + 1 ) * ( number_of_allocations - number_of_remaining_allocations - number_of_successes_arm_2 - number_of_failures_arm_1 + 2 ) , 2 ) + number_of_successes_arm_1 + 1)
end
function determine_policy_coefs_TS_OS_cond(N,decision_epochs, FPR, policy::Array{Float64}, boundary = "Bonferroni") 
    # determine policy-based coefficients that occur in the distribution over endstates
    num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
    a:: Array{ Float64 , 1 }  = zeros( Float64 , (length(decision_epochs))*num_states_N)
    thr_lower:: Array{ Float64 , 1 }  = zeros( Float64 , (length(decision_epochs))*num_states_N)
    thr_upper:: Array{ Float64 , 1 }  = zeros( Float64 , (length(decision_epochs))*num_states_N)

    if boundary == "Pocock"# we also allow for other boundaries (but no uniformly better method)
        levels = diff((FPR .* log.(1 .+ (exp(1) - 1).*(0:(length(decision_epochs)-1))./(length(decision_epochs)-1))./2))
    elseif boundary == "Bonferroni"
        levels = ones(length(decision_epochs)-1).*FPR./(2*(length(decision_epochs)-1))
    elseif boundary == "OBF"
        levels = diff((2 .- 2*cdf.(Normal(0,1), quantile(Normal(0,1), 1-FPR/2)./sqrt.((0:(length(decision_epochs)-1))./(length(decision_epochs)-1))))/2)
    end
    a[1] = 1 # probabiltiy of s0 = 1 initially
    for i in 1:length(decision_epochs)-1
        t = decision_epochs[i]

        # calculate indices corresponding to total sum successes
        idx_S = []
        for S in 0:t
            idx_S = push!(idx_S , [idx_map(S_1, N_1, S - S_1, N) for N_1 = 0 : t for S_1 = max(S-t+N_1, 0) :  min(S, N_1) ])
        end

        # calculate TS probabilities per possible state
        outc_stat = zeros(num_states_N)
        for N_1 in 0:t, S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:t-N_1
            outc_stat[idx_map(S_1, N_1, S_2, N)] = policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1-t)]
        end

        #calculate margins 
        thr_1M_TS_lower = zeros(num_states_N)
        thr_1M_TS_upper = zeros(num_states_N)
        for S in 0:t
            if i == 1
                thr_1M_TS_upper[idx_S[S+1]] .=  right_tail_fn(outc_stat[idx_S[S+1]], a[(i-1)*num_states_N .+ idx_S[S+1]], 0)
                thr_1M_TS_lower[idx_S[S+1]] .=  left_tail_fn(outc_stat[idx_S[S+1]], a[(i-1)*num_states_N .+ idx_S[S+1]], 0)    
            else
                thr_1M_TS_upper[idx_S[S+1]] .=  right_tail_fn(outc_stat[idx_S[S+1]], a[(i-1)*num_states_N .+ idx_S[S+1]], levels[i-1])
                thr_1M_TS_lower[idx_S[S+1]] .=  left_tail_fn(outc_stat[idx_S[S+1]], a[(i-1)*num_states_N .+ idx_S[S+1]], levels[i-1])    
            end
        end
        thr_lower[(i-1)*num_states_N+1:i*num_states_N] = thr_1M_TS_lower
        thr_upper[(i-1)*num_states_N+1:i*num_states_N] = thr_1M_TS_upper

        indc_not_reject = .!(outc_stat .>= thr_1M_TS_upper .|| outc_stat .<= thr_1M_TS_lower)
        println(i/(length(decision_epochs)-1))
        dDE =  decision_epochs[i+1]-t
        for N_1::Int64 = 0 :1: t , S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:t-N_1 
            if a[idx_map(S_1, N_1, S_2, N)+ (i-1)*num_states_N]>0 
                if indc_not_reject[idx_map(S_1, N_1, S_2, N)]
                    prob_superior = outc_stat[idx_map(S_1, N_1, S_2, N)]
                    allocprob = min(0.75, max(0.25, prob_superior))
                    vec_dN1 = [Int64(floorfn(allocprob*dDE)), Int64(ceil(allocprob*dDE))]
                    probs_dN1 = [ceil(allocprob*dDE)- allocprob*dDE   , allocprob*dDE - floorfn(allocprob*dDE)]  
                    for i_dN_1::Int64 = [1,2] , dS_1::Int64 = 0:1:vec_dN1[i_dN_1]  , dS_2::Int64 = 0:1:dDE-vec_dN1[i_dN_1]      
                        prob_dN1 = probs_dN1[i_dN_1]
                        if prob_dN1>0 
                            dN_1 = vec_dN1[i_dN_1]
                            a[idx_map(S_1+dS_1, N_1+dN_1, S_2+dS_2, N)+i*num_states_N] += binomial(dN_1, dS_1)*binomial(dDE-dN_1,dS_2)*a[idx_map(S_1, N_1, S_2, N)+(i-1)*num_states_N ]*prob_dN1                           
                        end
                    end
                end
            end
        end
    end

    t= nothing
    # calculate indices corresponding to total sum successes
    idx_S = []
    for S in 0:N
        idx_S = push!(idx_S , [idx_map(S_1, N_1, S - S_1, N) for N_1 = 0 : N for S_1 = max(S-N+N_1, 0) :  min(S, N_1) ])
    end

    # calculate TS probabilities per possible state
    outc_stat = zeros(num_states_N)
    for N_1 in 0:N, S_1::Int64 = 0 :1:  N_1  , S_2::Int64 = 0:1:N-N_1
        outc_stat[idx_map(S_1, N_1, S_2, N)] = policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1-N)]
    end

    #calculate margins 
    thr_1M_TS_lower = zeros(num_states_N)
    thr_1M_TS_upper = zeros(num_states_N)
    for S in 0:N
        thr_1M_TS_upper[idx_S[S+1]] .=  right_tail_fn(outc_stat[idx_S[S+1]], a[(length(decision_epochs)-1)*num_states_N .+ idx_S[S+1]], levels[length(decision_epochs)-1])
        thr_1M_TS_lower[idx_S[S+1]] .=  left_tail_fn(outc_stat[idx_S[S+1]], a[(length(decision_epochs)-1)*num_states_N .+ idx_S[S+1]], levels[length(decision_epochs)-1])    
    end

    thr_lower[(length(decision_epochs)-1)*num_states_N+1:length(decision_epochs)*num_states_N] = thr_1M_TS_lower
    thr_upper[(length(decision_epochs)-1)*num_states_N+1:length(decision_epochs)*num_states_N] = thr_1M_TS_upper

   return a, thr_lower, thr_upper
end

WS_onesided = function(S_1, N_1, S_2, N)
    # determines the Agresti-Caffo adjusted Wald statistic for testing H_1: p_1, p_2 free vs. H_0: p_1 = p_2
    phat_1 = (S_1+1)/(N_1+2)
    phat_2 = (S_2+1)/(N-N_1+2)
    return (phat_1 - phat_2)/sqrt(phat_1*(1-phat_1)/(N_1+2) + phat_2*(1-phat_2)/(N-N_1+2) )   
end

prob_TS = function(S_1, N_1, S_2, N, psi_0= [1,1], kappa_0= [2,2])
    alpha_2 = psi_0[2] + S_2
    alpha_1 = psi_0[1] + S_1
    beta_1 = kappa_0[1] - psi_0[1] + N_1 - S_1
    beta_2 = kappa_0[2] - psi_0[2] + N-N_1 - S_2

        dens = function(p)
            return( (1-cdf( Beta(alpha_1, beta_1), p))*pdf(Beta(alpha_2, beta_2), p))
        end
        result = max(0, min(1,quadgk(dens, 0, 1; rtol = Inf, atol = 0.001)[1] ))
            
        return( result)
end

function determine_policy_myopic_batches(N::Int64, decision_epochs, myopicrule ::Function)
    # this function is needed to determine the TS policy vector
    num_states::Int64 = DP_2_lin_index(N+1, 0, 0, 0, N+1)    # number of entries policy vector
    Policy:: Vector{ Float64}  = zeros(Float64, num_states) # init policy vector 
     for t in decision_epochs
        println(t)
         for  N_1::Int64 = 0 :  t# 
        for S_1::Int64 = 0 : N_1 , S_2::Int64 = 0 :  t - N_1
               Policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N-t+1)] = myopicrule(S_1, N_1, S_2 ,t)
        end
      end
    end
   return Policy
  end

# determine allocation probabilities using above functions
policy_path = "data//policies//"
input_path = "data//"
N=150
d = 30
decision_epochs = 0:d:N

if isfile(policy_path*"policy_TS_N=150_uniform.jld")
    # if the policy vector exists (after first run), load allocation probabilities (faster than the above)
    policy = JLD.load(policy_path*"policy_TS_N=150_uniform.jld")["policy"]
else
    # else just calculate it
    policy = determine_policy_myopic_batches(N, decision_epochs, prob_TS) # calculate thompson sampling policy (note this is not g^pi but pi)
    save(policy_path*"policy_TS_N=150_uniform.jld", "policy", policy)
end

#get SB OST 
thr_fixed = 0.986
discr_max = vcat([0.5,0.6,0.7,0.8,0.9,0.95],LinRange(0.986,0.999,24))

if isfile(input_path*"polcoefs_TS_uniform_max.jld")
    # if file exists load policy coefs (faster than the above)
    coefs_max = JLD.load(input_path*"polcoefs_TS_uniform_max.jld")["coefs"]
else
    # else get policy coefs g^pi for UX OST Markov chain to determine the UX critical value using functions above
    coefs_max = determine_policy_coefs_TS_max(N,decision_epochs, discr_max, policy) 
    save(input_path*"polcoefs_TS_uniform_max.jld","coefs", coefs_max)
end

# get first UX critical value using functions above
if isfile(input_path*"res_exact_ARREST.jld")
    # if file exists (after first run) load first UX critical value (faster than the above)
    res_exact = JLD.load(input_path*"res_exact_ARREST.jld")[ "results"]
else
    res_exact = determine_unconditional_CV(N, coefs_max, discr_max, 0.00004, 0.05)
    JLD.save(input_path*"res_exact_ARREST.jld", "results", res_exact)
end

discr_max_2 = vcat([0.5], LinRange(maximum(discr_max[discr_max .< res_exact[1]]), res_exact[1], 29))

if isfile(input_path*"polcoefs_TS_uniform_max_2.jld")
    # #load policy coefs (faster than the above)
    coefs_max_2 = JLD.load(input_path*"polcoefs_TS_uniform_max_2.jld")["coefs"]
else
    # get policy coefs g^pi for UX OST Markov chain to determine the UX critical value in the second iteration (refining M) using functions above
    coefs_max_2 = determine_policy_coefs_TS_max(N, decision_epochs, discr_max_2, policy) 
    save(input_path*"polcoefs_TS_uniform_max_2.jld","coefs", coefs_max_2)
end

if isfile(input_path*"res_exact_ARREST_2.jld")
    # if file exists (e.g., after first run), uncomment to load UX critical value (faster than the above)
    res_exact_2 = JLD.load(input_path*"res_exact_ARREST_2.jld")[ "results"]
else
    # get second UX critical value using functions above
    res_exact_2 = determine_unconditional_CV(N, coefs_max_2, discr_max_2, 0.00004, 0.05)
    JLD.save(policy_path*"res_exact_ARREST_2.jld", "results", res_exact_2)
end
thr_exact = res_exact_2[1]

# determine g^pi for the markov chain as defined in example 3 of the paper 
coefs_trialsize_fixed= determine_policy_coefs_TS_OS(N,decision_epochs, 0.986, policy)
coefs_trialsize_uncond = determine_policy_coefs_TS_OS(N,decision_epochs, thr_exact, policy)
coefs_trialsize_cond, lower_thr_cond, upper_thr_cond = determine_policy_coefs_TS_OS_cond(N,decision_epochs, 0.05, policy)

# determine vectors to calculate OCs
probs_superior = zeros(length(coefs_trialsize_fixed))
trialsize_long = zeros(length(coefs_trialsize_fixed))
num_states_N::Int64 =  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
  
for i in 1:length(decision_epochs)
    t = decision_epochs[i]
    function get_prob_superior(S_1,N_1,S_2,N)
        if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
            return(policy[DP_2_lin_index(N+1, S_1, N_1-S_1, S_2, N+1 - t)])
        else
            return(0.5)
        end
    end
    probs_superior[(i-1)*num_states_N + 1:i*num_states_N] = apply_func_states(N, get_prob_superior)
    trialsize_long[(i-1)*num_states_N+1:i*num_states_N] .= t
end

cap1 = function(x) min(x,1.)end
indc_endstate_uncond = cap1.(Float64.(probs_superior.>= thr_exact).+Float64.(1 .-probs_superior.>= thr_exact) .+vcat(zeros(num_states_N*(length(decision_epochs)-1)), ones(num_states_N)))
indc_endstate_fixed = cap1.(Float64.(probs_superior.>= 0.986).+Float64.(1 .-probs_superior.>= 0.986) .+vcat(zeros(num_states_N*(length(decision_epochs)-1)), ones(num_states_N)))
indc_endstate_cond = cap1.(Float64.(probs_superior.>= upper_thr_cond).+Float64.(probs_superior.<= lower_thr_cond) .+vcat(zeros(num_states_N*(length(decision_epochs)-1)), ones(num_states_N)))

# evaluate type I error, power, expected participant outcomes, expected trial size
range_p1 =  0:0.01:1.0
range_p2 = 0.12*ones(length(range_p1))

prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)]
prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)]

methods = ["CX-S (adj.)"]
measures = ["RR", "trial size", "benefit", "benefit_in"]
res_dict_plot_null = Dict([((( range_p1[i], range_p1[i]),  ( measure, method )), 0.0) for i in 1:length(range_p1)  for  measure in measures for method in methods])
res_dict_plot_alt = Dict([(((range_p1[i], range_p2[i]),   ( measure, method )), 0.0)  for i in 1:length(range_p1)  for   measure in measures for method in methods])

for idx_p in 1:length(prob_keys_alt)
    p_1 = prob_keys_alt[idx_p][1]
    p_2 = prob_keys_alt[idx_p][2]
    println("evaluation plots p_1 = "*string(p_1)*" p_2 = "*string(p_2))

    benefit_long_fixed = zeros(length(coefs_trialsize_fixed))
    benefit_long_uncond = zeros(length(coefs_trialsize_fixed))
    benefit_long_cond = zeros(length(coefs_trialsize_fixed))
    benefit_long2= zeros(length(coefs_trialsize_fixed))
    benefit_long3= zeros(length(coefs_trialsize_fixed))
    prob_coefs_long = zeros(length(coefs_trialsize_fixed))
     for i in 1:length(decision_epochs)
        t = decision_epochs[i]
        pval_func = function(S_1, N_1, S_2, N)
            return((t>=N_1 + S_2)*p_1^S_1*(1-p_1)^(N_1-S_1)*p_2^S_2*(1-p_2)^(t - N_1 - S_2))
        end
        prob_coefs_long[(i-1)*num_states_N+1:i*num_states_N] = apply_func_states(N, pval_func)

        # determine participant outcomes for the SB OST
        benefit_func_fixed = function(S_1, N_1, S_2, N)
            if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
                if p_1 > p_2
                    return(N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)]>=0.986)*(N-t))
                elseif p_1 == p_2
                    return(N/2)
                else
                    return(N-N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)]<=1-0.986)*(N-t))
                end
            else
                return(0.0)
            end
        end
        benefit_long_fixed[(i-1)*num_states_N+1:i*num_states_N] =apply_func_states(N, benefit_func_fixed)
    
        # determine expected patient outcomes for UX OST
        benefit_func_uncond = function(S_1, N_1, S_2, N)
            if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
                if p_1 > p_2
                    return(N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)] >= thr_exact)*(N-t))
                elseif p_1 == p_2
                    return(N/2)
                else
                    return(N-N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)]<=1-thr_exact)*(N-t))
                end
            else
                return(0.0)
            end
        end
        benefit_long_uncond[(i-1)*num_states_N+1:i*num_states_N] =apply_func_states(N, benefit_func_uncond)

        # determine expected participant outcomes for CX-S OST
        benefit_func_cond = function(S_1, N_1, S_2, N)
            if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
                if p_1 > p_2
                    return(N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)]>=upper_thr_cond[idx_map(S_1, N_1, S_2, N) + (i-1)*num_states_N])*(N-t))
                elseif p_1 == p_2
                    return(N/2)
                else
                    return(N-N_1 + (policy[DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)]<=lower_thr_cond[idx_map(S_1, N_1, S_2, N) + (i-1)*num_states_N])*(N-t))
                end
            else
                return(0.0)
            end
        end
        benefit_long_cond[(i-1)*num_states_N+1:i*num_states_N] =apply_func_states(N, benefit_func_cond)

        benefit_func2 = function(S_1, N_1, S_2, N)
            if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
                if p_1 > p_2
                    return(N_1 )
                elseif p_1 == p_2
                    return(N/2)
                else
                    return(N-N_1 )
                end
            else
                return(0.0)
            end
        end
        benefit_long2[(i-1)*num_states_N+1:i*num_states_N] =apply_func_states(N, benefit_func2)

        benefit_func3 = function(S_1, N_1, S_2, N)
            if  DP_2_lin_index(N+1, S_1, N_1 - S_1, S_2, N+1-t)<=length(policy)
                if p_1 > p_2
                    return(N_1/(t + (t==0)))
                elseif p_1 == p_2
                    return(1/2)
                else
                    return((N-N_1)/(t + (t==0) ))
                end
            else
                return(0.0)
            end
        end
        benefit_long3[(i-1)*num_states_N+1:i*num_states_N] =apply_func_states(N, benefit_func3)
    end

    # determine OCs under alternative
    res_dict_plot_alt[((p_1, p_2), ("RR","UX"))] =sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*(probs_superior.>=thr_exact .||probs_superior.<=1-thr_exact))
    res_dict_plot_alt[((p_1, p_2), ("RR","SB"))] =sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*(probs_superior.>=thr_fixed .||probs_superior.<=1-thr_fixed))
    res_dict_plot_alt[((p_1, p_2), ("RR","CX-S"))] =sum(prob_coefs_long.*(probs_superior.>=upper_thr_cond .||probs_superior .<= lower_thr_cond).*coefs_trialsize_cond.*indc_endstate_cond)

    res_dict_plot_alt[((p_1, p_2), ("trial size","UX"))] =    sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*trialsize_long)
    res_dict_plot_alt[((p_1, p_2), ("trial size","SB"))] =   sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*trialsize_long)
    res_dict_plot_alt[((p_1, p_2), ("trial size","CX-S"))] =   sum(prob_coefs_long.*coefs_trialsize_cond.*indc_endstate_cond.*trialsize_long)

    res_dict_plot_alt[((p_1, p_2), ("Esucc","SB"))] =sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*benefit_long_fixed)
    res_dict_plot_alt[((p_1, p_2), ("Esucc","UX"))] =sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*benefit_long_uncond)
    res_dict_plot_alt[((p_1, p_2), ("Esucc","CX-S"))] =sum(prob_coefs_long.*coefs_trialsize_cond.*indc_endstate_cond.*benefit_long_cond)

    res_dict_plot_alt[((p_1, p_2), ("Esucc_in","UX"))] =sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*benefit_long2)
    res_dict_plot_alt[((p_1, p_2), ("Esucc_in","SB"))] =sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*benefit_long2)
    res_dict_plot_alt[((p_1, p_2), ("Esucc_in","CX-S"))] =sum(prob_coefs_long.*coefs_trialsize_cond.*indc_endstate_cond.*benefit_long2)

   res_dict_plot_alt[((p_1, p_2), ("benefit","SB"))] =sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*benefit_long_fixed)/N
   res_dict_plot_alt[((p_1, p_2), ("benefit","UX"))] =sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*benefit_long_uncond)/N
   res_dict_plot_alt[((p_1, p_2), ("benefit","CX-S"))] =sum(prob_coefs_long.*coefs_trialsize_cond.*indc_endstate_cond.*benefit_long_cond)/N

   res_dict_plot_alt[((p_1, p_2), ("benefit_in","UX"))] =sum(prob_coefs_long.*coefs_trialsize_uncond.*indc_endstate_uncond.*benefit_long3)
   res_dict_plot_alt[((p_1, p_2), ("benefit_in","SB"))] =sum(prob_coefs_long.*coefs_trialsize_fixed.*indc_endstate_fixed.*benefit_long3)
  res_dict_plot_alt[((p_1, p_2), ("benefit_in","CX-S"))] =sum(prob_coefs_long.*coefs_trialsize_cond.*indc_endstate_cond.*benefit_long3)

   # determine type I error
    prob_coefs_long_null = zeros(length(coefs_trialsize_fixed))
     for i in 1:length(decision_epochs)
        t = decision_epochs[i]
        pval_func = function(S_1, N_1, S_2, N)
            if t>=N_1 + S_2
                result =p_1^S_1*(1.0-p_1)^(N_1-S_1)*p_1^S_2*(1.0-p_1)^(t - N_1 - S_2)
            else
                result = 0
            end
            return(result)
        end
        prob_coefs_long_null[(i-1)*num_states_N+1:i*num_states_N] = apply_func_states(N, pval_func)
    end

    res_dict_plot_null[((p_1, p_1), ("RR","UX"))] =sum(prob_coefs_long_null.*coefs_trialsize_uncond.*indc_endstate_uncond.*(probs_superior.>=thr_exact .||probs_superior.<=1-thr_exact))
    res_dict_plot_null[((p_1, p_1), ("RR","SB"))] =sum(prob_coefs_long_null.*coefs_trialsize_fixed.*indc_endstate_fixed.*(probs_superior.>=thr_fixed .||probs_superior.<=1-thr_fixed))
    res_dict_plot_null[((p_1, p_1), ("RR","CX-S"))] =sum(prob_coefs_long_null.*(probs_superior.>=upper_thr_cond .||probs_superior .<= lower_thr_cond).*coefs_trialsize_cond.*indc_endstate_cond)
end

#store the results
using Pickle
res_eval_arrest= Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
Pickle.store("data/res_arrest_analysis.pkl", res_eval_arrest)


