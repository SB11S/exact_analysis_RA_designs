# generate the results for the M_PTW trial described in the supplement
# after running this script, run the python file generate_results_M_PTW.py
using  DSP, RCall, BitBasis, JLD, Distributions, StatsBase, TickTock, Pickle

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
 
function inv_idx_map(idx::Int64, N::Int64)
    # inverse of idx_map, idx is the index and N is the trial horizon
    S_2 = Int64(floor(idx/(N*(N+2)+1)))
    N_1 = Int64(floor((idx - (1+(N + 2)*N)*S_2)/(N+1)))
    S_1 = Int64(idx - 1 - (1+(N + 2)*N)*S_2 - N_1*(N+1))
      return S_1, N_1, S_2
end

function idx_map(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # an index mapping less efficient but a bit easier to work with than the one in ''Jacko, P. (2019). BinaryBandit: An efficient
    #  Julia package for optimization and evaluation of the finite-horizon bandit problem with binary responses. can be made more efficient.''
    # S_a are the successes for arm a, N_1 is the allocations to arm 1, N is the trial horizon
    return(1 + S_1 + N_1*(N+1) + (1+(N + 2)*N)*S_2)
  end

perc_N1 = function(S_1::Int64, N_1::Int64, S_2::Int64, N::Int64)
    # percentage of participants allocated to group 1
    return(N_1/N)
end 

function simulate_logrank_M_PTW_design(Nvals, M, p_1, p_2)
    #simulate outcomes of the log-rank test for the M-PTW trial
    p = [p_1, p_2]
    res = zeros(M)
    N1 = zeros(M)
    lengths = []
    groups = []
    cens = []
    for m in 1:M#
        lengths_1 = []
        lengths_2 = []
        cens_1 = []
        cens_2 = []
        for N in Nvals
            # println(m/M*100)
            B = (rand()<= 1/2) + 1 #initialize the winner
            curlength = 1 #
            Win  =false # overwritten in for-loop
            for t in 0:N-1 # sample outcome for next participant t+1 and do bookkeeping
                assignment = B                # new assignment is current winner
                Win = rand() <= p[assignment] # win is sampled bernoulli

                # store N1
                if assignment == 1
                    N1[m] += 1
                end

                if (!Win) || (curlength == 15)# change the winner
                    if B == 1
                        B = 2 #change winner from 1 to 2
                        lengths_1 = push!(lengths_1, curlength) # store length of alloc. seq.
                        if Win 
                            cens_1 = push!(cens_1, 0) # censored
                        else
                            cens_1 = push!(cens_1, 1)
                        end
                    else
                        B =1
                        lengths_2 = push!(lengths_2, curlength)
                        if Win
                            cens_2 = push!(cens_2, 0) # censored
                        else
                            cens_2 = push!(cens_2, 1)
                        end
                    end
                    curlength = 1
                else
                    curlength += 1 # when winner does not change, we increment current length of alloc. seq.
                end
            end
            # do bookkeeping for last alloc. seq. if this has not yet been done 
            if Win & (curlength < 15) # change the winner
                if B == 1
                    lengths_1 = push!(lengths_1,curlength)
                    if Win 
                        cens_1 = push!(cens_1, 0) # censored
                    else
                        cens_1 = push!(cens_1, 1)
                    end
                else
                    lengths_2 = push!(lengths_2,curlength)
                    if Win 
                        cens_2 = push!(cens_2, 0) # censored
                    else
                        cens_2 = push!(cens_2,1)
                    end
                end
            end
        end
            lengths = vcat(lengths_1, lengths_2)
            groups = vcat(ones(length(lengths_1)), 2*ones( length(lengths_2)))
            cens = vcat(cens_1, cens_2)

        if all(lengths.==1) 
            res[m] =0
        else
            x =R"library(survival)
            survdiff(Surv(as.numeric($lengths),  as.numeric($cens)) ~ as.numeric($groups))[[6]][1]
            "
            res[m] =  x[1]<=0.05
        end
    end
    return [res, N1]
end

function determine_policy_coefs_M_PTW(N::Int64, Nvals,  cap::Int64 = 15) 
    # determine policy-based coefficients that occur in the distribution over endstates for the M-PTW trial
    # state encoding is as in suffstat case but, letting K be the # states in the suffstat case we have that 
    # the first set of K states correspond to a treatment sequence length of 1 (initial) and winner = 1 
    # second K states correspond to a treatment sequence length of 2 and winner = 1
    #  ...
    # then, the cap+1-th set of K states correspond to a treatment sequence of length 1 and winner = 2
    # this continues until we reach the 2*cap-th set of K states.
    # M-PTW sequence reset is slightly different than in the paper, where we make a transition with exactly the same amount of successes/failures, L = 1 and W is either 1 or 2.

    nstatesold::Int64 =(1 + N + N*(N+1) + (1+(N + 2)*N)*N)
    num_states_N::Int64 =  2*cap*nstatesold   # number of states when augmented with the winner (B) and treatment sequence length (L)
    a:: Array{ Float64 , 1 }  = zeros( Float64 , num_states_N)
    a[1] = 1/2 # probabiltiy of s0 = 1 initially
    a[1 + cap*nstatesold]=1/2   # first cap*nstatesold states correspond to winner = control/trtm 1
    for t in 1:1:N
        # a_old = deepcopy(a)
        println("Progress determining coefs: "*string(round(t/N*100,digits = 2))*"%") # print progress
        for N_1::Int64 = t :-1: 0 , S_1::Int64 = N_1 :-1:  0  , S_2::Int64 = t-N_1:-1:  0, B::Int64 = 1:2, lengthseq = 1:cap
        @inbounds begin  
            # for each state for next decision epoch, sum the coefs for possible previous states 
            a_val = 0
            if ((S_2 <t-N_1)  & (N_1<t) & (B == 1) & (lengthseq == 1))# transitions from states where B is 2 and L<=15 to state where B is 1 and L is 1 by a loss on arm 1
                a_val += sum([a[idx_map(S_1, N_1, S_2, N) +  Int64((cap + k)*nstatesold)] for k in 0:cap-1])
            end

            if (S_1 >0) & (B == 1) & (lengthseq >1) # if B == 1 and L_t+1 >1 then we can only reach this from a state where B == 1 and L_t = L_t+t -1, with one success/alloc less to 1
                a_val += a[idx_map(S_1-1, N_1-1, S_2, N) + Int64((lengthseq-2)*nstatesold) ]
            end
            
            if (S_1 > 0) & (B == 2) & (lengthseq ==1) # success arm 1 and 15 consequtive successes
                a_val += a[idx_map(S_1-1, N_1-1, S_2, N) + Int64((cap-1)*nstatesold) ]
            end

            if ((N_1 > 0) & (S_1 < N_1) & (B == 2) & (lengthseq == 1)) # transitions from states where B is 1 and L<=15 to state where B is 2 and L is 1 by a loss on arm 2
                a_val +=  sum([a[idx_map(S_1, N_1-1, S_2, N) +  Int64(k*nstatesold)] for k in 0:cap-1])
            end
            if ((S_2 >0) & (B == 2)& (lengthseq >1)) # if B == 2 and L_t+1 >1 then we can only reach this from a state where B == 2 and L_t = L_t+t -1, with one success/alloc less to 2
                a_val += a[idx_map(S_1, N_1, S_2-1, N)+  Int64((cap + lengthseq-2)*nstatesold) ]
            end

            if (S_2 > 0) & (B == 1)& (lengthseq  == 1)# success arm 2 and 15 consequtive successes
                a_val += a[idx_map(S_1, N_1, S_2-1, N) + Int64((2*cap-1)*nstatesold) ]
            end
            a[idx_map(S_1, N_1, S_2, N) + (B-1)* Int64(cap*nstatesold) + (lengthseq-1)*nstatesold]   = a_val # assign the new value to element of a corresponding to the values of S_a, N_a, B and lengthseq
        end
        end

        # reset to a new trial sequence if t is equal to any stopping point
        if t in cumsum(Nvals)
            for N_1::Int64 = t :-1: 0 , S_1::Int64 = N_1 :-1:  0  , S_2::Int64 = t-N_1:-1:  0
            @inbounds begin  
                #sum all the a_values corresponding to B and lengthseqs for a state
                weight_state = sum([a[idx_map(S_1, N_1, S_2, N) + (B-1)* Int64(cap*nstatesold) + (lengthseq-1)*nstatesold] for B in 1:2 for lengthseq in 1:cap])

                # prob that new B is 1 or 2 is 1/2
                a[idx_map(S_1, N_1, S_2, N)] = 1/2*weight_state
                a[idx_map(S_1, N_1, S_2, N) +  Int64(cap*nstatesold)] = 1/2*weight_state

                # a-values corresponding to cap > 1 are now set to zero (we cannot have treatment lengths >2 directly after restart)
                for lengthseq in 2:cap, B in 1:2
                    a[idx_map(S_1, N_1, S_2, N) + (B-1)* Int64(cap*nstatesold) + (lengthseq-1)*nstatesold] = 0
                end
            end
            end
        end
    end
    #sum a values over lengthseqs and over B as the only thing that matters in the end is the S and N values
   return reduce(+,[a[Int64(1 +(k-1)*nstatesold) : Int64(k*nstatesold)] .+ a[Int64(1 +(k-1+cap)*nstatesold) : Int64((k+cap)*nstatesold)] for k in 1:cap])
end

WS_onesided = function(S_1, N_1, S_2, N)
    # determines the Agresti-Caffo adjusted Wald statistic for testing H_1: p_1, p_2 free vs. H_0: p_1 = p_2
    phat_1 = (S_1+1)/(N_1+2)
    phat_2 = (S_2+1)/(N-N_1+2)
    return (phat_1 - phat_2)/sqrt(phat_1*(1-phat_1)/(N_1+2) + phat_2*(1-phat_2)/(N-N_1+2) )   
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

# # uncomment to determine M-PTW coefficients (g^pi) using the function above 
N = 327
Nvals = [18, 15, 15,15,10,16,16,10,8,19,16,16,13,10,8,18,15,15,12,19,16,13,9,5]
input_path = "data/"
output_path = "data/"

if isfile(input_path* "coefs_PRW_reiertsen_N="*string(N)*".jld")
     #after first run, load M-PTW coefficients (faster alternative to above)
    coefs_M_PTW = JLD.load(input_path* "coefs_PRW_reiertsen_N="*string(N)*".jld")["coefs"] # calculated using determine_policy_coefs_M_PTW
else
    # if not available, read M-PTW coefs
    coefs_M_PTW = determine_policy_coefs_M_PTW(N, Nvals) 
    JLD.save(input_path* "coefs_PRW_reiertsen_N="*string(N)*".jld", "coefs", coefs_M_PTW)
end

# #determine CX-S thresholds
idx_S = []
num_states_N=  1 + N + N*(N+1) + (1+(N + 2)*N)*N   #1 + N + N^2 + (N + 1)*N^2 # number of entries of coef-vector 
for S in 0:N
    idx_S = push!(idx_S , [idx_map(S_1, N_1, S - S_1, N)  for N_1 = 0 : N for S_1 = max(S-N+N_1, 0) :  min(S, N_1) ])
end

outc_stat = apply_func_states(N, WS_onesided)
thr_1M_PTW_upper = zeros(num_states_N)
thr_1M_PTW_lower = zeros(num_states_N)
for S in 0:N
    thr_1M_PTW_upper[idx_S[S+1]] .=  right_tail_fn(outc_stat[idx_S[S+1]], ((coefs_M_PTW)[idx_S[S+1]]), 0.025)
    thr_1M_PTW_lower[idx_S[S+1]] .=  left_tail_fn(outc_stat[idx_S[S+1]], ((coefs_M_PTW)[idx_S[S+1]]), 0.025)
end
vec_EPASA = apply_func_states(N, perc_N1)


if isfile(input_path*"res_thr_PTW_exact.jld")
    res_thr_PTW_exact = JLD.load(input_path*"res_thr_PTW_exact.jld")["res_thr_PTW_exact"]# load unconditional threshold# #
else
    # use function above to determine UX critical value for the Wald statistic
    res_thr_PTW_exact = determine_unconditional_CV(N, coefs_M_PTW, outc_stat ,0.00005, 0.05/2)
    JLD.save(input_path*"res_thr_PTW_exact.jld","res_thr_PTW_exact",res_thr_PTW_exact)
end
# after first run, load UX crit. val for the Wald statistic (faster alternative to above)
thr_PTW_exact = res_thr_PTW_exact[1]

# determine the type I errors and powers
range_p1 =  sort( vcat(0:0.01:1.0, [0.830, 0.748]))
range_p2 = 0.748*ones(length(range_p1))

prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)]
prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)]

policies = ["PTW"]
measures = ["RR uncond", "RR cond"]
res_dict_plot_null = Dict([((( range_p1[i], range_p1[i]),  (policy, measure)), 0.0) for i in 1:length(range_p1)  for  policy in policies for measure in measures])
res_dict_plot_alt = Dict([(((range_p1[i], range_p2[i]),  (policy, measure)), 0.0)  for i in 1:length(range_p1)  for  policy in policies for measure in measures])

for idx_p in 1:length(prob_keys_alt)
    # evaluate power
    p_1 = prob_keys_alt[idx_p][1]
    p_2 = prob_keys_alt[idx_p][2]
    println("evaluation plots p_1 = "*string(p_1)*" p_2 = "*string(p_2))
    pval_func = function(S_1, N_1, S_2, N)
        return(p_1^S_1*(1-p_1)^(N_1-S_1)*p_2^S_2*(1-p_2)^(N - N_1 - S_2))
    end

    prob_coefs = apply_func_states(N, pval_func)
    
    prob_PTW = prob_coefs.*coefs_M_PTW
    res_dict_plot_alt[((p_1, p_2), ("PTW", "RR uncond"))] =sum(prob_PTW.*((outc_stat.>=thr_PTW_exact) .|| (outc_stat.<= -1 .*thr_PTW_exact)))
    res_dict_plot_alt[((p_1, p_2), ("PTW", "RR cond"))] =sum(prob_PTW.*((outc_stat.>=thr_1M_PTW_upper) .|| (outc_stat.<= thr_1M_PTW_lower)))
    res_dict_plot_alt[((p_1, p_2), ("PTW", "RR uncond"))] =sum(prob_PTW.*((outc_stat.>=thr_PTW_exact) .|| (outc_stat.<= -1 .*thr_PTW_exact)))
    res_dict_plot_alt[((p_1, p_2), ("PTW", "EPASA"))] =sum(prob_PTW.*vec_EPASA)

    # evaluate type I error
    pval_func = function(S_1, N_1, S_2, N)
        return(p_1^S_1*(1-p_1)^(N_1-S_1)*p_1^S_2*(1-p_1)^(N - N_1 - S_2))
    end
    prob_coefs = apply_func_states(N, pval_func)
    prob_PTW = prob_coefs.*coefs_M_PTW
    res_dict_plot_null[((p_1, p_1), ("PTW", "RR uncond"))] =sum(prob_PTW.*((outc_stat.>=thr_PTW_exact) .|| (outc_stat.<= -1 .*thr_PTW_exact)))
    res_dict_plot_null[((p_1, p_1), ("PTW", "RR cond"))] =sum(prob_PTW.*((outc_stat.>=thr_1M_PTW_upper) .|| (outc_stat.<= thr_1M_PTW_lower)))
    res_dict_plot_null[((p_1, p_1), ("PTW", "Fopt" ))] =NaN
end

#store the results
res_eval_reiertsen = Dict([("res_mats_alt", res_dict_plot_alt), ("res_mats_null", res_dict_plot_null)])
Pickle.store(output_path*"res_M_PTW_analysis.pkl", res_eval_reiertsen)


### uncomment below to get simulation results ###
# #estimate the operating characteristics of the logrank test from simulation
# M =100000
# tick()
# range_p1 =  sort(vcat(0.0:0.01:1.0, [ 0.830, 0.748]))
# range_p2 = 0.748*ones(length(range_p1))

# prob_keys_alt = [(range_p1[i], range_p2[i]) for i in 1:length(range_p1)]
# prob_keys_null = [(range_p2[i], range_p2[i]) for i in 1:length(range_p2)]

# res_dict_plot_null_sim = Dict([(( range_p1[i], range_p1[i]), [0.0,[]]) for i in 1:length(range_p1)  ])
# res_dict_plot_alt_sim = Dict([((range_p1[i], range_p2[i]), [0.0,[]])  for i in 1:length(range_p1)  ])

# for idx_p in 1:length(prob_keys_alt)
#     p_1 = prob_keys_alt[idx_p][1]
#     p_2 = prob_keys_alt[idx_p][2]

#     # evaluate power
#     println("evaluation plots p_1 = "*string(p_1)*" p_2 = "*string(p_2))
#     simres = simulate_logrank_M_PTW_design(Nvals, M, p_1, p_2)
#     res_dict_plot_alt_sim[(p_1, p_2 )] =[mean(simres[1]), simres]

#     #evaluate type I error
#     println("evaluation plots p_1 = "*string(p_1)*" p_2 = "*string(p_1))
#     simres = simulate_logrank_M_PTW_design(Nvals, M, p_1, p_1)
#     res_dict_plot_null_sim[(p_1, p_1 )] =[mean(simres[1]), simres]
# end
# ElapsedTime = tok()
# res_eval_reiertsen_sim = Dict([("res_mats_alt", res_dict_plot_alt_sim), ("res_mats_null", res_dict_plot_null_sim), ("time", ElapsedTime)])

# #store
# Pickle.store(output_path*"res_M_PTW_analysis_sim.pkl", res_eval_reiertsen_sim)
