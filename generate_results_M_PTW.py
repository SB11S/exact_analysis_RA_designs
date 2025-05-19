# Generate the type I error and Power difference plots for the Reiertsen trial as given in the supplement (note: you might have to change result_path and input_path to the correct path to the file)
import pickle as pkl   
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


result_path = "data/figures/" #change this to path where you want results
input_path = "data/"#change this to input path
tables_path = "data/"#change this to path for tables 
file = open(input_path + "res_M_PTW_analysis.pkl", 'rb')

res_dict = pkl.load(file)

file = open(input_path + "res_M_PTW_analysis_sim.pkl", 'rb')
res_dict_sim = pkl.load(file)
M = len(res_dict_sim["res_mats_null"][(0.,0.)][1][0])

plt.close("all")

fz =30
plt.rcParams.update({'font.size': fz}) # must set in top
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(40*0.6, 40*0.17), gridspec_kw={'width_ratios': [1,1]})
plt.subplots_adjust(wspace=0.37, hspace=0.4)


# make plots for comparison under null
res_dict_plot_null = res_dict["res_mats_null"]
df_plot_null = pd.DataFrame([[key[0][0], key[0][1],key[1][0],key[1][1], res_dict_plot_null[key]] for key in res_dict_plot_null.keys()])
df_plot_null.columns =[ "$p$", "$p_2$","Policy","Measure", "val"]
df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,3]!= 'RR 1M',:]


Colors =['darkblue', 'darkviolet']
types = ['-', '-', "-",'-']
lwd = 1

df_plot_null = df_plot_null.loc[df_plot_null.iloc[:,3] != "Fopt"]
df_plot_null = df_plot_null.pivot_table(index = [ "$p$"], columns = [ "Policy","Measure"], values = "val")
df_plot_null = df_plot_null["PTW"]
res_dict_plot_null_sim = res_dict_sim["res_mats_null"]
df_plot_null_sim = pd.DataFrame([[ key[0] ,res_dict_plot_null_sim[key][0]] for key in res_dict_plot_null_sim.keys()])
df_plot_null_sim = df_plot_null_sim.pivot_table(index = [ 0], values = 1)

   
df_plot_null.plot(ax = axs[0], grid = True,linewidth = lwd, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$", style = types ,
                  ylim = [-.001,0.061],
                    yticks = np.arange(0,0.1,0.01),xticks = [0.0, 0.1 ,0.2,0.3,0.4, 0.5,0.6, 0.7,0.8,0.9, 1.0], 
                    ylabel = "Type I error rate",  xlim = [-0.012, 1.012], marker = '',  color =Colors, legend = None, zorder = 3)
axs[0].set_xticklabels(['','0.1','','0.3','','0.5','','0.7','','0.9',''])
confr = 1.96*np.sqrt(df_plot_null_sim.values*(1-df_plot_null_sim.values)/M)
axs[0].plot(df_plot_null_sim.index,df_plot_null_sim.values.T[0],  linewidth = lwd, color ='orangered', zorder =3)
axs[0].set_title("${\\bf A}$: Type I error rate profiles", fontsize = 32)
markers = ["o","*","^",'', '']
ms_scale = 1.25
for i, line in enumerate(axs[0].get_lines()):
    line.set_marker(markers[i])
    line.set_markersize(ms_scale*5)
    line.set_markevery([0,10,30,50,70, 75, 91,101])
pd.Series(0.05, index = df_plot_null.index).plot(ax = axs[0],zorder = 2,grid = True, xlabel = "        $\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}=\\theta_{\\text{D}})$",linewidth = 1.5, color ='k', style = '--', label='_nolegend_')

axs[0].axvline( 0.748,color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)
axs[0].fill_between(df_plot_null_sim.index,df_plot_null_sim.values.T[0] - confr.T[0],df_plot_null_sim.values.T[0] +confr.T[0], facecolor='orangered', alpha=0.5, zorder =3)

np.mean(((df_plot_null_sim.values.T[0] - confr.T[0]) >0.05)) 


# make a LaTeX table
res_dict_plot_alt = res_dict["res_mats_alt"]
df = pd.DataFrame([[key[0][0], key[0][1],key[1][0], key[1][1], res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df.columns =[ "$p$", "$p_2$","Policy","Test",  "val"]
df = df.loc[df.loc[:,'Test']!= 'RR 1M',:]
df.loc[:,"val"]= df.loc[:,"val"]*100
df.columns = ["$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", "$\\theta_\\text{C}$","Policy", "Test", "val"]
df = df.loc[[(df.iloc[i,0] >= df.iloc[i,1] and df.iloc[i,0] in [0.748,0.8, 0.83,0.85,0.9,0.95,1.0]) for i in range(len(df.loc[:,"$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$"]))],:]
df = df.loc[df.loc[:,"Policy"] == 'PTW',:]
df = df.loc[df.loc[:,"Test"] != 'Fopt',:]

df = df.pivot_table(index = [ "$\\theta_\\text{C}$", "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$"], columns = [ "Test"], values = "val")


res_dict_plot_alt_sim = res_dict_sim["res_mats_alt"]
df_plot_alt_sim = pd.DataFrame([[ key[0] ,res_dict_plot_alt_sim[key][0]] for key in res_dict_plot_alt_sim.keys()])
df_plot_alt_sim = df_plot_alt_sim.pivot_table(index = [ 0], values = 1)
df_plot_alt_sim = df_plot_alt_sim.loc[[indexval>=0.748 and  indexval in  [0.748,0.8,0.83, 0.85,0.9,0.95,1.0]for indexval in  df_plot_alt_sim.index ],:]
confr = 1.96*np.sqrt(df_plot_alt_sim.values*(1-df_plot_alt_sim.values)/M)*100
df.loc[:,"log-rank"] = (df_plot_alt_sim.values*100)
df = df.astype("string").applymap(lambda x: f'\\phantom{{00}}{float(x):.2f}' if np.log(round(float(x),2))/np.log(10)<1 else f'\\phantom{{0}}{float(x):.2f}' if np.log(round(float(x),2))/np.log(10)<2 else f'{float(x):.2f}')

df.loc[:,"log-rank"] = df.loc[:,"log-rank"] + " +/- " + [f"{val[0]:.2f}" for val in confr]

df = df.iloc[:,[1,2,3,0]]
df.columns = ['CX-S Wald',  'UX Wald', "log-rank test",'EPASA']
df = df.rename(index=lambda val: f'{val:.3f}')


# make plots for comparison under alternative
res_dict_plot_alt = res_dict["res_mats_alt"]
df_plot_alt = pd.DataFrame([[key[0][0], key[0][1],key[1][0],key[1][1], res_dict_plot_alt[key]] for key in res_dict_plot_alt.keys()])
df_plot_alt.columns =[ "$p$", "$p_2$","Policy","Measure", "val"]
df_plot_alt = df_plot_alt.loc[df_plot_alt.iloc[:,3]!= 'RR 1M',:]
df_plot_alt = df_plot_alt.loc[df_plot_alt.iloc[:,3] != "EPASA"]
df_plot_alt = df_plot_alt.pivot_table(index = [ "$p$"], columns = [ "Policy","Measure"], values = "val")
df_plot_alt = df_plot_alt.loc[:,"PTW"]
res_dict_plot_alt_sim = res_dict_sim["res_mats_alt"]
df_plot_alt_sim = pd.DataFrame([[ key[0] ,res_dict_plot_alt_sim[key][0]] for key in res_dict_plot_alt_sim.keys()])
df_plot_alt_sim = df_plot_alt_sim.pivot_table(index = [ 0], values = 1)

  
plt.rcParams.update({'font.size': fz}) # must set in top
df_plot_alt.loc[df_plot_alt_sim.index, ("logrank")] = df_plot_alt_sim.values.T[0]
Colors =['darkviolet',  'orangered']
df_plot_alt = df_plot_alt.iloc[:, [1,0,2]]

df_plot_alt["RR uncond"] = df_plot_alt["RR uncond"] -df_plot_alt["RR cond"] 
df_plot_alt["logrank"] = df_plot_alt["logrank"] -df_plot_alt["RR cond"] 
df_plot_alt = df_plot_alt.iloc[:,[0,2]]


types = ['-', '-', "-"]
df_plot_alt.iloc[df_plot_alt.index < 0.748] = np.nan
for i in range(1):
    df_plot_alt.iloc[:,i].plot(ax = axs[1],grid = True,linewidth = lwd, xlabel = "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types[i] ,ylim = [-0.056,0.005], yticks = np.arange(-0.06,0.01,0.01),
                       xlim = [0.748,1.0], xticks = [0.748, 0.83, 0.9], ylabel = "Power difference",  marker = '',zorder =3,  color =Colors[i])

df_plot_alt.iloc[:,1].plot(ax = axs[1],grid = True,linewidth = lwd, xlabel = "$\\theta_{\\text{D}}$  ($\\theta_{\\text{C}}\\leq\\theta_{\\text{D}})$", style = types[2] ,ylim =[-0.052,0.011], yticks = np.arange(-0.06,0.02,0.01),
                     xlim = [0.748-0.003,1.003], xticks = [0.748, 0.8, 0.9, 1.0], legend = None, ylabel = "Power difference \ncompared to CX-S Wald",  marker = '',  color =Colors[1], zorder =3)

axs[1].set_xticklabels(['0.748','', '0.9','1.0'])
handles, labels = plt.gca().get_legend_handles_labels()
confr = 1.96*np.sqrt(df_plot_alt_sim.values*(1-df_plot_alt_sim.values)/M)
axs[1].fill_between(df_plot_alt_sim.index,df_plot_alt.iloc[:, 1] - confr.T[0], df_plot_alt.iloc[:, 1] +confr.T[0],zorder =3, facecolor=Colors[1], alpha=0.5)
axs[1].axvline( 0.830, color = 'k', linestyle = '-.', linewidth = 1.5, zorder = 2)
axs[1].axvline( 0.748, color = 'k', linestyle = '-', linewidth = 1.5, zorder = 2)
order = [0,1,2]
axs[1].set_title("${\\bf B}$: Power differences", fontsize = 32)
# Add a 2-line legend
ms_scale = 1.25
markers = ['*',"^","",'', '']
for i, line in enumerate(axs[1].get_lines()):
    line.set_marker(markers[i])
    line.set_markersize(ms_scale*5)
    line.set_markevery([75,81, 84, 86,91,96, 101])

legend_labels = ['CX-S Wald','UX Wald', 'log-rank test']
legend_colors = ['darkblue', 'purple', 'orangered' ]
legend_markers = ["o","*","^"]
lines = []
for i_color,_ in enumerate(legend_colors):
    lines.append(plt.Line2D([0], [0], color=legend_colors[i_color], marker = legend_markers[i_color], markersize= 10,linewidth=2))  # Set linewidth as per your preference

wide_legend = fig.legend(lines, legend_labels, loc='lower center', ncol=3,   bbox_to_anchor=(0.5, -0.25), bbox_transform=plt.gcf().transFigure)
# axs[0].text(0.05, 0.9, 'A', transform=axs[0].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#         facecolor='white',  # Background color of the box
#         edgecolor='black',  # No edge around the box (optional)
#         boxstyle='round,pad=0.15',  # Rounded corners with padding
#     ), zorder =4)
# axs[1].text(0.05, 0.9, 'B', transform=axs[1].transAxes, fontsize=fz, weight='bold', va='bottom', ha='left',bbox=dict(
#         facecolor='white',  # Background color of the box
#         edgecolor='black',  # No edge around the box (optional)
#         boxstyle='round,pad=0.15',  # Rounded corners with padding
#     ), zorder = 4)

remove_border = True
axs[0].spines['top'].set_visible(not remove_border)
axs[0].spines['right'].set_visible(not remove_border)
axs[0].spines['bottom'].set_visible(not remove_border)
axs[0].spines['left'].set_visible(not remove_border)
axs[1].spines['top'].set_visible(not remove_border)
axs[1].spines['right'].set_visible(not remove_border)
axs[1].spines['bottom'].set_visible(not remove_border)
axs[1].spines['left'].set_visible(not remove_border)
#store
plt.savefig(result_path + "plot_M_PTW_analysis.pdf", bbox_inches='tight')
df.to_latex(tables_path + "res_table_M_PTW.tex", escape=False)
