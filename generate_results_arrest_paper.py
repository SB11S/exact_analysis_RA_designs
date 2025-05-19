# generate Figure  in supplement for ARREST trial in supplement (note: you might have to change result_path and input_path to the correct path to the file)
import pickle as pkl   
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

result_path = "data/figures/" #change this to path where you want results
input_path = "data/"#change this to input path
file = open(input_path + "res_arrest_analysis.pkl", 'rb')
res_dict = pkl.load(file)

plt.close("all")
fz = 30
plt.rcParams.update({'font.size': fz}) # must set in top
# Make one figures for results
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(40*0.6, 40*0.4))
plt.subplots_adjust(wspace=0.32, hspace=0.4)
N=150

# make plots for comparison under null
res_dict_plot_null = res_dict["res_mats_null"]
df_plot_null = pd.DataFrame([[key[0][0], key[0][1],key[1][1] ,res_dict_plot_null[key]] for key in res_dict_plot_null.keys()])
df_plot_null.columns =[ "$p$", "$p_2$","Measure", "val"]

Colors =[ 'darkblue', "orangered","darkviolet"]
markers = ["o","^","*","",'', '']
types = ['-', '-', "-"]
lwd = 1

df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,2] != "Fopt"]
df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,2] != "benefit assumed"]
df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,2] != "trialsize assumed"]
df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,2] != "trialsize uncond"]
df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,2] != "benefit uncond"]
df_plot_null = df_plot_null.pivot_table(index = [ "$p$"], columns = [ "Measure"], values = "val")
df_plot_null = df_plot_null.iloc[:,[0,2,3]]

df_plot_null.plot(ax = axs[0,0],grid = True,linewidth = 1, xlabel = "    $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$", style = types ,ylim = [-0.003,0.091],
                    yticks = [0., 0.01,0.02,0.03,0.04, 0.05, 0.06, 0.07, 0.08,0.09],xticks = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 
                    ylabel = "Type I error rate", fontsize = fz, marker = '',  color =Colors, legend = None, zorder = 3,xlim =[-0.012,1.012])
axs[0,0].set_xticklabels(['','0.1','','0.3','','0.5','','0.7','','0.9',''])
pd.Series(0.05, index = df_plot_null.index).plot(ax = axs[0,0],grid = True, xlabel = "    $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$",linewidth = 1.5, color ='k', style = '--', label='_nolegend_', zorder = 2)
axs[0,0].set_title("${\\bf A}$: Type I error rate profiles", fontsize = 32)
axs[0,0].axvline( 0.12, color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)


Colors = [ 'orangered', 'darkviolet']
# make plots for comparison rejection rate under alternative/power
res_dict_plot_alt = res_dict["res_mats_alt"]
df_plot_alt = pd.DataFrame([[key[0][0], key[0][1],key[1][0],key[1][1] ,res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df_plot_alt.columns =[ "$p$", "$p_2$","Measure","OST", "val"]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "benefit",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "benefit_in",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "trial size",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc_in",:]

df_plot_alt = df_plot_alt.pivot_table(index = [ "$p$"], columns = ["OST"], values = "val")

df_plot_alt.loc[:,"UX" ] =  df_plot_alt.loc[:,"UX" ] - df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"SB" ] =  df_plot_alt.loc[:,"SB" ] - df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"CX-S (adj.)" ] -=  df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt = df_plot_alt.iloc[:,[1,2,3]]
df_plot_alt = df_plot_alt.iloc[:,[1,2]]
df_plot_alt.iloc[df_plot_alt.index < 0.12] = np.nan
df_plot_alt.plot(ax = axs[0,1], grid = True,linewidth = 1, xlabel = "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types ,ylim = [-0.023,0.1025], yticks = np.arange(-0.04,0.14,0.02),
                       xlim = [0.12 - 0.012,1.012], zorder = 3, legend = None,  xticks = [0.12,0.2, 0.3,0.37,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0], ylabel = "Power difference \ncompared to CX-S OST", marker = '',  color =Colors)
axs[0,1].set_xticklabels(['0.12','', '0.3','','','0.5','', '0.7', '','0.9',''])
axs[0,1].set_title("${\\bf B}$: Power differences", fontsize = 32)
axs[0,1].axvline( 0.37, color = 'k', linestyle = '-.', linewidth = 1.5, zorder = 2)
axs[0,1].axvline( 0.12, color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)

# make plots for comparison expected patient outcomes under alternative
res_dict_plot_alt = res_dict["res_mats_alt"]
df_plot_alt = pd.DataFrame([[key[0][0], key[0][1],key[1][0], key[1][1] ,res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df_plot_alt.columns =[ "$p$", "$p_2$","Measure", "OST","val"]

df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "RR",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "trial size",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "benefit_in",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc_in",:]
df_plot_alt = df_plot_alt.pivot_table(index = [ "$p$"], columns = ["OST"], values = "val")

df_plot_alt.loc[:,"UX" ] -=   df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"SB" ] -=   df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"CX-S (adj.)" ] -=   df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt = df_plot_alt.iloc[:,[1,2,3]]
df_plot_alt = df_plot_alt.iloc[:,[1,2]]
df_plot_alt = df_plot_alt 
df_plot_alt.iloc[df_plot_alt.index < 0.12] = np.nan
df_plot_alt.plot(ax = axs[1,0],grid = True,linewidth = 1, xlabel = "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types , yticks = np.arange(-0.0,0.05, 0.01),
                      ylim = [-0.003, 0.033], xlim = [0.12 - 0.012,1.012],  xticks = [0.12,0.2, 0.3,0.37,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0], ylabel = "Difference in EPASA \ncompared to CX-S OST",  marker = '',
                        color =Colors, legend = None, zorder = 3)
axs[1,0].set_xticklabels(['0.12','', '0.3','','','0.5','', '0.7','','0.9',''])
axs[1,0].set_title("${\\bf C}$: EPASA differences", fontsize = 32)
axs[1,0].axvline( 0.37, color = 'k', linestyle = '-.', linewidth = 1.5, zorder = 2)
axs[1,0].axvline( 0.12, color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)

# make plots for comparison trial size under alternative
res_dict_plot_alt = res_dict["res_mats_alt"]
df_plot_alt = pd.DataFrame([[key[0][0], key[0][1],key[1][0], key[1][1] ,res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df_plot_alt.columns =[ "$p$", "$p_2$","Measure", "OST","val"]

df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "RR",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "benefit",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "benefit_in",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc",:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.loc[:,"Measure"] != "Esucc_in",:]

df_plot_alt = df_plot_alt.pivot_table(index = [ "$p$"], columns = ["OST"], values = "val")

df_plot_alt.loc[:,"UX" ] -=  df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"SB" ] -=  df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt.loc[:,"CX-S (adj.)" ] -=  df_plot_alt.loc[:,"CX-S" ] 
df_plot_alt = df_plot_alt.iloc[:,[1,2,3]]
df_plot_alt = df_plot_alt.iloc[:,[1,2]]
df_plot_alt.iloc[df_plot_alt.index < 0.12] = np.nan
df_plot_alt = df_plot_alt/150
df_plot_alt.plot(ax = axs[1,1], grid = True,linewidth = 1, xlabel = "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types ,ylim = [-18.5/150-0.005,.5/150], yticks = np.arange(-0.14,0.003,0.02),
                       xlim = [0.12 - 0.012,1.012],xticks = [0.12,0.2, 0.3,0.37,0.4,0.5,0.6, 0.7, 0.8, 0.9, 1.0], ylabel = "Difference in ETS ratio \ncompared to CX-S OST", marker = '',
                        color =Colors, legend = None, zorder = 3)
axs[1,1].set_title("${\\bf D}$: ETS ratio differences", fontsize = 32)
axs[1,1].set_xticklabels(['0.12','', '0.3','','','0.5','', '0.7','','0.9',''])
axs[1,1].axvline( 0.37, color = 'k', linestyle = '-.', linewidth = 1.5, zorder = 2)
axs[1,1].axvline( 0.12, color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)
# axs[0,0].text(0.05, 0.9, 'A', transform=axs[0,0].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#     facecolor='white',  # Background color of the box
#     edgecolor='black',  # No edge around the box (optional)
#     boxstyle='round,pad=0.15',  # Rounded corners with padding
# ))
# axs[1,0].text(0.05, 0.9, 'C', transform=axs[1,0].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#         facecolor='white',  # Background color of the box
#         edgecolor='black',  # No edge around the box (optional)
#         boxstyle='round,pad=0.15',  # Rounded corners with padding
#     ))
# axs[0,1].text(0.05, 0.9,'B', transform=axs[0,1].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#         facecolor='white',  # Background color of the box
#         edgecolor='black',  # No edge around the box (optional)
#         boxstyle='round,pad=0.15',  # Rounded corners with padding
#     ))
# axs[1,1].text(0.05, 0.9,'D', transform=axs[1,1].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#         facecolor='white',  # Background color of the box
#         edgecolor='black',  # No edge around the box (optional)
#         boxstyle='round,pad=0.15',  # Rounded corners with padding
#     ))


ms_scale = 1.25

for i, line in enumerate(axs[0,0].get_lines()):
            line.set_marker(markers[i])
            line.set_markersize(ms_scale*5)
            line.set_markevery([0,10,12, 30,50,70,90,100])
    
for i, line in enumerate(axs[0,1].get_lines()):
            line.set_marker(markers[i+1])
            line.set_markersize(ms_scale*5)
            if len(line.get_xdata())>0:
               line.set_markevery([12,20,30,37,50, 70, 90])
for i, line in enumerate(axs[1,1].get_lines()):
            line.set_marker(markers[i+1])
            line.set_markersize(ms_scale*5)
            if len(line.get_xdata())>0:
               line.set_markevery([12,20,30,37,50, 70, 90])
for i, line in enumerate(axs[1,0].get_lines()):
            line.set_marker(markers[i+1])
            line.set_markersize(ms_scale*5)
            if len(line.get_xdata())>0:
               line.set_markevery([12,20,30,37,50, 70, 90])

# Add a 2-line legend
legend_labels = ['CX-S OST',  'UX OST', 'SB OST']
legend_colors = ['darkblue',  'darkviolet', 'orangered']
legend_markers = ["o","*","^"]
lines = []
for i_color,_ in enumerate(legend_colors):
    lines.append(plt.Line2D([0], [0], color=legend_colors[i_color], marker = legend_markers[i_color], markersize= 10, linewidth=2))  # Set linewidth as per your preference

wide_legend = fig.legend(lines, legend_labels, loc='lower center', ncol=3,   bbox_to_anchor=(0.5, -0.05), bbox_transform=plt.gcf().transFigure, fontsize=fz)
#store
remove_border = True
axs[0,0].spines['top'].set_visible(not remove_border)
axs[0,0].spines['right'].set_visible(not remove_border)
axs[0,0].spines['bottom'].set_visible(not remove_border)
axs[0,0].spines['left'].set_visible(not remove_border)
axs[0,1].spines['top'].set_visible(not remove_border)
axs[0,1].spines['right'].set_visible(not remove_border)
axs[0,1].spines['bottom'].set_visible(not remove_border)
axs[0,1].spines['left'].set_visible(not remove_border)
axs[1,0].spines['top'].set_visible(not remove_border)
axs[1,0].spines['right'].set_visible(not remove_border)
axs[1,0].spines['bottom'].set_visible(not remove_border)
axs[1,0].spines['left'].set_visible(not remove_border)
axs[1,1].spines['top'].set_visible(not remove_border)
axs[1,1].spines['right'].set_visible(not remove_border)
axs[1,1].spines['bottom'].set_visible(not remove_border)
axs[1,1].spines['left'].set_visible(not remove_border)
plt.savefig(result_path + "plot_arrest_analysis.pdf", bbox_inches='tight')






