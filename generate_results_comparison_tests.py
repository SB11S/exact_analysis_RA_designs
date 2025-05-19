# generate figure 1 in paper or figure 4 in supplement (note: you might have to change result_path and input_path to the correct path to the file)
# note: uncomment line 21-36 to generate one input file from the input files for i = 60,240,960 for the wald RR, before this you must run full_evaluation_tests_[policy].jl for these trial sizes
#       uncomment line 41-56 to generate one input file from the input files for i = 60,240,960 for the FET RR, before this you must run full_evaluation_tests_[policy].jl for these trial sizes
# note that this results in the file "res_comparison_tests_[policy].pkl", which is already attached.
import pickle as pkl  
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
addletters=False#adding letters to plot
result_path = "data/figures/" #change this to path where you want results
input_path = "data/"#change this to input path
tables_path = "data/"#change this to path for tables
remove_border = True
policy = "EA" # either EA or RDP
starting_points = [0.01, 0.3,0.9]   

include_BT = False # include Boschloo test in plots (not used in paper)
code_i_end = "$\\bar{i}$"
# file = open(input_path + "res_comparison_tests_"+policy+".pkl", 'rb')
# res_dict = pkl.load(file)

file = open("data/eval_tests_"+policy+"_N=60.pkl", 'rb')
res_dict = pkl.load(file)

file = open("data/eval_tests_"+policy+"_N=240.pkl", 'rb')
res_dict_240 = pkl.load(file)

res_dict['res_mats_alt'] = res_dict['res_mats_alt'] | res_dict_240['res_mats_alt'] 
res_dict['res_mats_null'] = res_dict['res_mats_null'] | res_dict_240['res_mats_null'] 

file = open("data/eval_tests_"+policy+"_N=960.pkl", 'rb')
res_dict_960 = pkl.load(file)

res_dict['res_mats_alt'] = res_dict['res_mats_alt'] | res_dict_960['res_mats_alt'] 
res_dict['res_mats_null'] = res_dict['res_mats_null'] | res_dict_960['res_mats_null'] 
with open('data/res_comparison_tests_'+policy+'.pkl', 'wb') as handle:
    pkl.dump(res_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

# file = open("data/res_FET_"+policy+".pkl", 'rb')
# res_dict_FET = pkl.load(file)

file = open("data/RR_FET_"+policy+"_N=60.pkl", 'rb')
res_dict_FET = pkl.load(file)

file = open("data/RR_FET_"+policy+"_N=240.pkl", 'rb')
res_dict_240 = pkl.load(file)

res_dict_FET['res_mats_alt'] = res_dict_FET['res_mats_alt'] | res_dict_240['res_mats_alt'] 
res_dict_FET['res_mats_null'] = res_dict_FET['res_mats_null'] | res_dict_240['res_mats_null'] 

file = open("data/RR_FET_"+policy+"_N=960.pkl", 'rb')
res_dict_960 = pkl.load(file)

res_dict_FET['res_mats_alt'] = res_dict_FET['res_mats_alt'] | res_dict_960['res_mats_alt'] 
res_dict_FET['res_mats_null'] = res_dict_FET['res_mats_null'] | res_dict_960['res_mats_null'] 
with open('data/res_FET_'+policy+'.pkl', 'wb') as handle:
    pkl.dump(res_dict_FET, handle, protocol=pkl.HIGHEST_PROTOCOL)

plt.close("all")


# Make global figure
fz = 25
plt.rcParams.update({'font.size': fz}) # must set in top

fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(40*0.6, 40*0.6))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# -------------------------------------------------------make plots for comparison under null---------------------------------------------------
res_dict_plot_null = res_dict["res_mats_null"]
df_plot_null = pd.DataFrame([[key[0][0], key[0][1],key[0][2],key[1], res_dict_plot_null[key]] for key in res_dict_plot_null.keys()])
df_plot_null.columns =[code_i_end, "$p$", "$p_2$","Test",  "val"]
df_plot_null.loc[:,"val"]= df_plot_null.loc[:,"val"]

if include_BT:
    Colors =["royalblue" ,'darkblue', 'green','darkviolet', 'orangered',"blue"]
    markers = ["o","s","*","^",'D', '','']
    types = ['-','-','-','-', "-", "-"]
    names = [ 'CX-S Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald']
else:
    Colors =["royalblue",'darkblue', 'green','darkviolet', 'orangered' ]
    markers = ['D',"o","s","*","^", '']
    types = ['-','-','-','-', "-"]
    names = [ 'CX-S Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald']


Nvals = [60,240, 960]
lwd = 1

res_dict_plot_null_FET =res_dict_FET["res_mats_null"]
df_plot_null_FET = pd.DataFrame([[key[0][0], key[0][1],key[0][2], key[1], res_dict_plot_null_FET[key]] for key in res_dict_plot_null_FET.keys()])
df_plot_null_FET.columns =[code_i_end, "$p$", "$p_2$", "test", "val"]

order = [1,2,3,0]
scale_msz = 1.25
if policy == "EA":
    if include_BT:
        lwds = [lwd*4, lwd*2, lwd*2,lwd,lwd, lwd, lwd]
        msz = [10, 7.5, 5,5,5, 5,5]
    else:
        lwds = [lwd*4, lwd*2, lwd,lwd,lwd, lwd]
        msz = [10*scale_msz, 7.5*scale_msz, 5*scale_msz,5*scale_msz,5*scale_msz, 5*scale_msz]
        
else:
    lwds = [lwd]*5 + [lwd*1.5]
    msz = [5*scale_msz]*6
    
    
markevery = [10, 5, 2]
letters = ['A','C','E']  
for i, N in enumerate(Nvals):
    ax = axs[i, 0]
    df_plot_null_N = df_plot_null.loc[df_plot_null.iloc[:,0]==N]        
    df_plot_null_N = df_plot_null_N.pivot_table(index = [ "$p$"], columns = [ "Test"], values = "val")
    df_plot_null_N = df_plot_null_N.loc[:, df_plot_null_N.columns != 'EPASA']
    df_plot_null_N = df_plot_null_N.iloc[:,order]
    df_plot_null_N.columns = names

    if include_BT:
        df_plot_null_FET_N = df_plot_null_FET.loc[(df_plot_null_FET.iloc[:,0]==N)]        
    else:
        df_plot_null_FET_N = df_plot_null_FET.loc[(df_plot_null_FET.iloc[:,0]==N) & (df_plot_null_FET.iloc[:,3]=="RR")]        
    if include_BT:
        df_plot_null_FET_N = df_plot_null_FET_N.pivot_table(index = [ "$p$"], columns = [ "test"],  values = "val")
        df_plot_null_N.loc[:,["FET", "Boschloo"]] = df_plot_null_FET_N.values
    else:
        df_plot_null_FET_N = df_plot_null_FET_N.pivot_table(index = [ "$p$"],   values = "val")
        df_plot_null_N.loc[:,["FET"]] = df_plot_null_FET_N.values


    # df_plot_null_N = df_plot_null_N.loc[:, df_plot_null_N.columns != 'Asymp.']
    df_plot_null_N = df_plot_null_N.iloc[:,[4,0,1,2,3]]
    df_plot_null_N.plot(ax = ax,grid = True,linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$", style = types ,
                        yticks = np.arange(0,0.2,0.01),xlim =[-0.012,1.012], xticks = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 
                        ylabel = "Type I error rate",   color =Colors, legend = None, zorder = 4)
    if policy == "EA":
        ax.set_ylim([-0.003,0.061])
    else:
        ax.set_ylim([-0.003,0.071])
    # plt.subplots_adjust(left=0.25, bottom  =0.2)
    pd.Series(0.05, index = df_plot_null_N.index).plot(ax = ax, grid = True, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$",
                                                       linewidth = lwd, color ='k', style = '--', label='_nolegend_', zorder = 3)
    ax.set_title("${\\bf "+letters[i]+"}$: Type I error rate profiles, "+code_i_end+" = "+str(N), fontsize = 27)
    ax.set_xticklabels(['','','0.1','','0.3','','0.5','','0.7','','0.9',''])
     
    for j, line in enumerate(ax.get_lines()):
        line.set_marker(markers[j])
        line.set_markevery(every = [0,1,10,30,50,70,90,95, 99, 100])
        line.set_linewidth(lwds[j])
        line.set_markersize(msz[j])
        
    ax.axvline(0.01, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder =2)
    ax.axvline(0.3, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder = 2)
    ax.axvline(0.9, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder = 2)
    ax.spines['top'].set_visible(not remove_border)
    ax.spines['right'].set_visible(not remove_border)
    ax.spines['bottom'].set_visible(not remove_border)
    ax.spines['left'].set_visible(not remove_border)
    # ax.axvline(1, color = 'darkblue', linestyle = '-', linewidth = lwd*1)
    if addletters:
        axs[i,0].text(0.02 + 0.03*(policy == "EA"), 0.9, letters[i], transform=axs[i,0].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left', zorder = 4,bbox=dict(
        facecolor='white',  # Background color of the box
        edgecolor='black',  # No edge around the box (optional)
        boxstyle='round,pad=0.15',  # Rounded corners with padding
    ))
   
types = ['-','-','-','-', "-"]

# -------------------------------------------------------------- make latex table for supplement-------------------------------------------------------
res_dict_plot_alt = res_dict["res_mats_alt"]
df_plot_alt = pd.DataFrame([[key[0][0], key[0][1],key[0][2],key[1], res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df_plot_alt.columns =[code_i_end, "$p$", "$p_2$","Test",  "val"]


df_plot_alt_FET =res_dict_FET['res_mats_alt']
df_plot_alt_FET = pd.DataFrame([[key[0][0], key[0][1],key[0][2], key[1], df_plot_alt_FET[key]] for key in df_plot_alt_FET.keys()])
df_plot_alt_FET.columns =[code_i_end, "$p$", "$p_2$", "Test", "val"]

if not include_BT:
    df_plot_alt_FET = df_plot_alt_FET.loc[ (df_plot_alt_FET.iloc[:,3] == 'RR').values,: ].iloc[:,[0,1,2,4]]
    df_plot_alt_FET.loc[:,"Test"] = "FET"
else:
    df_plot_alt_FET.loc[df_plot_alt_FET.loc[:,"Test"]=='RR', "Test"] = 'FET'
    df_plot_alt_FET.loc[df_plot_alt_FET.loc[:,"Test"]=='RR BT', "Test"] = 'Boschloo'


df = pd.concat([df_plot_alt, df_plot_alt_FET], axis =0)
df.loc[:,"val"]= df.loc[:,"val"]*100

p2vals = [0.0, 0.01, 0.05, 0.1, 0.3,0.5,0.7,0.9,0.95,0.99,1.0]  

# if policy == 'RDP':
p1vals = dict([(60, dict([
    (0.0,[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
               (0.01,[0.01, 0.11, 0.21, 0.31, 0.41, 0.51]),
               (0.05,[0.05, 0.15, 0.25, 0.35, 0.45, 0.55]),
               (0.1,[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
               (0.3,[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
               (0.5,[0.5, 0.6, 0.7, 0.8, 0.9, 1.]),
               (0.7, [0.7, 0.8, 0.9, 1.]),
               (0.9, [0.9, 1.]),
               (0.95, [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]),
               (0.99, [0.99, 1.0]),
               (1.0, [1.0])])), 
               (240, dict([(0.0,[0.  , 0.05, 0.1 , 0.15, 0.2 , 0.25]),
                              (0.01,[0.01, 0.06, 0.11, 0.16, 0.21, 0.26]),
                              (0.05,[0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 ]),
                              (0.1,[0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35 ]),
                              (0.3,[0.3 , 0.35, 0.4 , 0.45, 0.5 , 0.55]),
                              (0.5,[0.5 , 0.55, 0.6 , 0.65, 0.7 , 0.75]),
                              (0.7, [0.7 , 0.75, 0.8 , 0.85, 0.9 , 0.95 ]),
                              (0.9, [0.9 , 0.95, 1.0  ]),
                              (0.95, [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]),
                              (0.99, [0.99, 1.0]),
                              (1.0, [1.0])])),
               
               (960, dict([(0.0,[0.  , 0.02, 0.04 , 0.06, 0.08 , 0.1]),
                              (0.01,[0.01, 0.03, 0.05 , 0.07, 0.09 , 0.11]),
                              (0.05,[0.05, 0.07, 0.09 , 0.11, 0.13 , 0.15 ]),
                              (0.1,[0.1 , 0.12, 0.14 , 0.16, 0.18 , 0.2 ]),
                              (0.3,[0.3 , 0.32, 0.34 , 0.36,  0.38 ,  0.4]),
                              (0.5,[0.5 , 0.52, 0.54 , 0.56, 0.58 , 0.6]),
                              (0.7, [0.7 , 0.72,  0.74 , 0.76,  0.78 ,  0.8 ]),
                              (0.9, [0.9 , 0.92,  0.94 , 0.96,  0.98 ,  1.0 ]),
                              (0.95, [0.95, 0.96, 0.97, 0.98, 0.99, 1.0]),
                              (0.99, [0.99, 1.0]),
                              (1.0, [1.0])])),
               ])


df.columns = ["$\\Iend$" , "$\\theta_{\\text{D}}$", "$\\theta_\\text{C}$", "Test", "val"]

for Nval in Nvals:
    df_N = df.loc[df.loc[:, "$\\Iend$"]== Nval] 
    df_N = df_N.loc[[(df_N.iloc[i,2] in p2vals) and (df_N.iloc[i,1] in p1vals[Nval][df_N.iloc[i,2]]) and (df_N.iloc[i,1] >= df_N.iloc[i,2] )  for i in range(int(len(df_N.loc[:,"$\\theta_{\\text{D}}$"])))],:]
    df_N = df_N.pivot_table(index = [ "$\\Iend$","$\\theta_\\text{C}$", "$\\theta_{\\text{D}}$"], columns = [ "Test"], values = "val")

    if include_BT:
        order = [2,3,4,0,5,1]    
    else:
        if (policy == 'EA') :
            order = [1,2,3,0,4]
        else:
            order =  [2,3,4,0,5,1]
    df_N = df_N.iloc[:,order].round(2)

    if include_BT:
        df_N.columns = [ 'CX-S Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', "FET", "Boschloo"]    
    else:
        if (policy == 'EA') :
            df_N.columns = [ 'CX-S Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', "FET"]
        else:
            df_N.columns = [ 'CX-S Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', "FET (corr.)", "EPASA"]
  
    typeIerrorinfl = [key[1] == key[2] and df_N.loc[key,'Asymp. Wald']>5.  for key in df_N.index]
    indc_alt = [key[1] != key[2]   for key in df_N.index]
    if include_BT:
        highestpower = df_N.iloc[indc_alt, [0,1,2,4,5]].apply(lambda v: np.argwhere(v == np.amax(v)), 1)
    else:
        highestpower = df_N.iloc[indc_alt, [0,1,2,4]].apply(lambda v: np.argwhere(v == np.amax(v)), 1)

    df_N = df_N.astype("string").applymap(lambda x: f'\\phantom{{00}}{float(x):.2f}' if np.log(round(float(x),2))/np.log(10)<1 else f'\\phantom{{0}}{float(x):.2f}' if np.log(round(float(x),2))/np.log(10)<2 else f'{float(x):.2f}')
    
    df_N = df_N.rename(index=lambda val: str(round(val, 3)))
    df_N.iloc[typeIerrorinfl,3] = ["\\textcolor{red}{\\bf "+element+"}" for element in df_N.iloc[typeIerrorinfl,3]]
    k=0
    for i in range(df_N.shape[0]):
        if indc_alt[i]:
            indices = highestpower.iloc[k].T[0]
            if include_BT:
                indices[indices == 4]=5
            indices[indices == 3]=4
            df_N.iloc[i,indices] = "\\textcolor{darkgreen}{\\bf"+df_N.iloc[i,indices]+"}"  
            k+=1

    latex_table = df_N.to_latex(escape=False)
    if policy == 'EA':
        # Modify the LaTeX code to align the last four columns to the right
        latex_table = latex_table.replace(
            "\\begin{tabular}{llllllll}",
            "\\begin{tabular}{lllrrrrr}"  # Align last four columns to the right
        )
    else:
        # Modify the LaTeX code to align the last four columns to the right
        latex_table = latex_table.replace(
            "\\begin{tabular}{lllllllll}",
            "\\begin{tabular}{lllrrrrrr}"  # Align last four columns to the right
        )
    # store
    if include_BT:
        with open(tables_path + "res_table_comparison_tests_"+policy+"_also1perc_BT_"+str(Nval)+".tex", "w") as f:
            f.write(latex_table)
    else:
        with open(tables_path + "res_table_comparison_tests_"+policy+"_also1perc_"+str(Nval)+".tex", "w") as f:
                f.write(latex_table)
    

# ------------------------------------------ make plots for power comparison --------------------------------------------------------------------
starting_points2=np.append(starting_points, np.inf)
f = 1
msz_scale = 1.25
if policy == "EA":
    if include_BT:
        lwds = np.tile([3*lwd,lwd,3*lwd, lwd] ,len(starting_points)*3)
        msz = np.tile([7.5,5,5,5,7.5,5,5,5,7.5*f,5*f,5*f,5*f] ,len(starting_points))
        Colors = ['darkblue', 'green', 'darkviolet', 'blue']
    else:
        lwds = np.tile([3*lwd,lwd,lwd] ,len(starting_points)*3)
        msz = np.tile([7.5*scale_msz,5*scale_msz,5*scale_msz, 7.5*scale_msz,5*scale_msz,5*scale_msz, 7.5*f*scale_msz,5*f*scale_msz,5*f*scale_msz] ,len(starting_points))
else:
    lwds =np.tile([lwd,lwd,lwd] ,len(starting_points)*3)
    lwds[-9:-3] = np.tile([2*lwd,2*lwd,lwd] ,2)
    msz =np.tile([msz_scale*5,msz_scale*5,msz_scale*5, msz_scale*5,msz_scale*5,msz_scale*5, msz_scale*5*2/2,msz_scale*5*2/2,msz_scale*5*2/2] ,len(starting_points))
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Test"]!='EPASA',:]
tests = np.unique(df_plot_alt['Test'])
if include_BT:
    markers = np.tile(["D","s","*", ""] ,len(starting_points)*3)
else:
    markers = np.tile(["D","s","*"] ,len(starting_points)*3)
order = [1,2,3,0]
markersteps = [10, 5, 2]
letters = ['B','D','F'] 

for i_N,N in enumerate(Nvals):
    ax = axs[i_N, 1]
    for i in range(len(starting_points)):
        sp = starting_points[i]
        df_plot_alt_N = df_plot_alt.loc[df_plot_alt.iloc[:,0]==N].loc[df_plot_alt.iloc[:,2]==sp]
        df_plot_alt_N = df_plot_alt_N.pivot_table(index = [ "$p$"], columns = [ "Test"], values = "val")
        df_plot_alt_N = df_plot_alt_N.iloc[:,order]
        df_plot_alt_N_FET = df_plot_alt_FET.loc[df_plot_alt_FET.iloc[:,0]==N].loc[df_plot_alt_FET.iloc[:,2]==sp]
        
        if include_BT:
            df_plot_alt_N_FET = df_plot_alt_N_FET.pivot_table(index = [ "$p$"], columns = [ "Test"],  values = "val")
        else:
            df_plot_alt_N_FET = df_plot_alt_N_FET.pivot_table(index = [ "$p$"],   values = "val")
         
        df_plot_alt_N.loc[:, 'FET'] = df_plot_alt_N_FET.loc[:,'val']
         
        for test in np.append(tests, ['FET']):
            if include_BT:
                if not test == 'Exact Cond. (1M)':
                    df_plot_alt_N.loc[:,test] = df_plot_alt_N.loc[:,test] -  df_plot_alt_N.loc[:,'Exact Cond. (1M)']
            else:
                if not test == 'Exact Cond. (1M)':
                    df_plot_alt_N.loc[:,test] = df_plot_alt_N.loc[:,test] -  df_plot_alt_N.loc[:,'Exact Cond. (1M)']
          
        if include_BT:
            df_plot_alt_N.loc[:,"Boschloo"] = df_plot_alt_N_FET.loc[:,'Boschloo']-df_plot_alt_N_FET.loc[:,'FET']
            
        df_plot_alt_N = df_plot_alt_N.loc[:, df_plot_alt_N.columns != 'Asymp.']
        df_plot_alt_N = df_plot_alt_N.loc[:, df_plot_alt_N.columns != 'Exact Cond. (1M)']
        df_plot_alt_N2 = df_plot_alt_N.loc[df_plot_alt_N.index<=starting_points2[i+1],:]
        df_plot_alt_N = df_plot_alt_N.iloc[:,[2,0,1]]
        df_plot_alt_N2 = df_plot_alt_N2.iloc[:,[2,0,1]]
        ax.spines['top'].set_visible(not remove_border)
        ax.spines['right'].set_visible(not remove_border)
        ax.spines['bottom'].set_visible(not remove_border)
        ax.spines['left'].set_visible(not remove_border)
        if policy == "EA":
            df_plot_alt_N2.loc[df_plot_alt_N2.index >= sp].plot(ax = ax, legend = None,#marker = markers[i],markevery=5,
                                                           color  = ['royalblue', 'green', 'darkviolet'], grid = True, xticks = np.arange(0.1,1.1,0.2),
                                                          ylabel = "Power diff. versus FET (corr.)",zorder= 3,
                                                          linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types) 
            df_plot_alt_N.loc[df_plot_alt_N.index >= sp].plot(ax = ax, alpha = 0.6,legend = None,#marker = markers[i],markevery=5,
                                                           color  = [ 'royalblue','green', 'darkviolet'],grid = True, xticks = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],#xticks = np.arange(0.1,1.1,0.2),
                                                          ylabel = "Power diff. versus FET (corr.)",zorder= 3,
                                                          linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types) 
            
            df_plot_alt_N.loc[df_plot_alt_N.index == sp].plot(ax = ax, legend = None,marker = markers[i],markerfacecolor='white',
                                                             color  = ['royalblue', 'green', 'darkviolet'],grid = True, xticks = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],#xticks = np.arange(0.1,1.1,0.2),
                                                            ylabel = "Power difference \ncompared to CX-S Wald",xlim =[-0.012,1.012],zorder= 3,yticks = np.arange(0.0,0.21,0.03),
                                                            linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types) 

        else:
                
            df_plot_alt_N2.loc[df_plot_alt_N2.index >= sp].plot(ax = ax, legend = None, 
                                                           color = ['royalblue', 'green', 'darkviolet'],zorder= 3,grid = True, xticks = np.arange(0.1,1.1,0.2),
                                                          ylabel = "Power diff. versus  FET (corr.)",  
                                                          linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types)
            df_plot_alt_N.loc[df_plot_alt_N.index >= sp].plot(ax = ax,alpha = 0.6, legend = None, 
                                                           color = ['royalblue', 'green', 'darkviolet'],zorder= 3,grid = True, xticks = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],#xticks = np.arange(0.1,1.1,0.2),
                                                          ylabel = "Power diff. versus FET (corr.)",  yticks = np.arange(0.45,-0.6,-0.15),
                                                          linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types)
            df_plot_alt_N.loc[df_plot_alt_N.index == sp].plot(ax = ax, legend = None,marker = markers[i],markerfacecolor='white',
                                                             color = ['royalblue', 'green', 'darkviolet'],grid = True, xticks = [0,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],#xticks = np.arange(0.1,1.1,0.2),
                                                            ylabel = "Power difference \ncompared to CX-S Wald",xlim =[-0.012,1.012],zorder= 3,
                                                            linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types,
                                                            yticks = np.arange(-1.1,0.1,0.1)) 

         
            
    ax.set_xticklabels(['','','0.1','','0.3','','0.5','','0.7','','0.9',''])
    if include_BT:
        ntest = 4
    else:
        ntest = 3
        
    for j, line in enumerate(ax.get_lines()):
        line.set_marker(markers[j])
        if not np.mod(np.floor(j/ntest),3)==2.0:
            i_starting_point = np.int64(np.floor(j/(ntest*3)))
            marker_x = [int(np.round(((val  - starting_points[i_starting_point])/0.01))) for val in p1vals[N][starting_points[i_starting_point]]]
            # print(marker_x)
            length_line = len(line.get_xdata())
            marker_x = [val for val in marker_x if val < length_line]
            line.set_markevery(every = marker_x)
        else:
            line.set_markevery(1)
       
        line.set_linewidth(lwds[j])
        line.set_markersize(msz[j])

    plt.subplots_adjust(left=0.25, bottom  =0.2)
    ax.set_title("${\\bf "+letters[i_N]+"}$: Power differences, "+code_i_end+" = "+str(N), fontsize = 27)
    ax.axvline(0.01, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder = 2)
    ax.axvline(0.3, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder = 2)
    ax.axvline(0.9, color = 'k', linestyle = '-', linewidth = lwd*1.5, zorder = 2)
     
    if policy == "EA":
        ax.set_ylim([-0.01,0.18])
    else:
        ax.set_ylim([-0.65,0.02])
    if addletters:
        axs[i_N,1].text(0.02+ 0.03*(policy == "EA"), 0.9, letters[i_N], transform=axs[i_N,1].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
        facecolor='white',  # Background color of the box
        edgecolor='black',  # No edge around the box (optional)
        boxstyle='round,pad=0.15',  # Rounded corners with padding
    ))
        

# Add a 2-line legend
if policy == 'RDP':
    legend_labels = ['CX-S   Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', 'FET (corr.)']
    legend_colors = ['darkblue', 'green', 'darkviolet', 'orangered', 'royalblue']
    legend_markers = ["o","s","*","^",'D']
else:
    if include_BT:
        legend_labels = ['CX-S   Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', 'FET', 'Boschloo']
        legend_colors = ['darkblue', 'green', 'darkviolet', 'orangered', 'royalblue', 'blue']
        legend_markers = ["o","s","*","^",'D', '']
    else:
         legend_labels = ['CX-S   Wald', 'CX-SA Wald', 'UX Wald', 'Asymp. Wald', 'FET']
         legend_colors = ['darkblue', 'green', 'darkviolet', 'orangered', 'royalblue']
         legend_markers = ["o","s","*","^",'D']
    

lines = []
for i_color,_ in enumerate(legend_colors):
    lines.append(plt.Line2D([0], [0], color=legend_colors[i_color], marker =legend_markers[i_color],  linewidth=2, markersize = 10))  # Set linewidth as per your preference

wide_legend = fig.legend(lines, legend_labels, loc='lower center', ncol=3,   bbox_to_anchor=(0.55, 0.09), bbox_transform=plt.gcf().transFigure)


if include_BT:
    plt.savefig(result_path + "plot_comparison_tests_"+policy+"_BT.pdf", bbox_inches='tight') 
else:
    plt.savefig(result_path + "plot_comparison_tests_"+policy+".pdf", bbox_inches='tight')

