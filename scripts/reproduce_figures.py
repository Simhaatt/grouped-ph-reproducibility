"""Numerical reconstructions of the three supplied figures; originals stay unchanged."""
from frozen_results import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#333333','#882255']
def save(fig,name):
    for ext in ['pdf','png']:fig.savefig(ROOT/'results/figures'/f'{name}.{ext}',dpi=200)
    plt.close(fig)
def main():
    m=main_rows();fig,axs=plt.subplots(1,2,figsize=(12,4.5))
    for color,cohort in zip(COLORS,dict.fromkeys(r['cohort'] for r in m)):
        rows=sorted([r for r in m if r['cohort']==cohort],key=lambda r:r['modal_event'])
        for ax,key,title in zip(axs,['B','KP'],['(a) Efron + Breslow baseline','(b) Efron + profile baseline']):
            ax.errorbar([r['modal_event'] for r in rows],[r['D_'+key] for r in rows],yerr=[r['SE_'+key] for r in rows],fmt='o',ms=4,capsize=2,color=color,label=cohort)
            ax.set(xlabel='Modal event mass',ylabel='Cox minus grouped NLL',title=title);ax.axhline(0,color='#888888',lw=.7)
    handles,labels=axs[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,bbox_to_anchor=(.5,-.07),fontsize=8)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,'F1_baseline_decomposition')
    e=event_rows();fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for ax,key,title in zip(axs,['modal_all','modal_event'],['(a) All exits','(b) Events only']):
        for color,target in zip(COLORS,['low','mid','high']):
            rows=[r for r in e if r['target']==target]
            ax.errorbar([r[key] for r in rows],[r['D_B'] for r in rows],yerr=[r['SE_B'] for r in rows],fmt='o',ms=4,capsize=2,color=color,label=target)
        ax.set(xlabel='Modal mass',ylabel='Cox minus grouped NLL',title=title);ax.axhline(0,color='#888888',lw=.7)
    axs[1].legend(title='Baseline level');fig.tight_layout();save(fig,'F2_event_concentration')
    w=[r for r in weibull_rows() if r['shape']==.8 and r['censor_pct']==0 and r['n']==10000]
    assert len(w)==5
    fig,axs=plt.subplots(1,2,figsize=(11,4.2));x=[r['T'] for r in w]
    for key,label,color in [('bias_cox_pct','Cox/Efron',COLORS[0]),('bias_grouped_pct','Grouped joint MLE',COLORS[1])]:axs[0].plot(x,[r[key] for r in w],'o-',color=color,label=label)
    axs[0].set(xlabel='Number of intervals T',ylabel='Mean absolute relative coefficient bias (%)',title='(a) Coefficient bias');axs[0].legend()
    axs[1].errorbar(x,[r['D_B'] for r in w],yerr=[r['SE_B'] for r in w],fmt='o-',capsize=3,color=COLORS[0]);axs[1].set(xlabel='Number of intervals T',ylabel='Cox minus grouped NLL',title='(b) Predictive difference');axs[1].axhline(0,color='#888888',lw=.7)
    fig.suptitle('Continuous Weibull PH: shape 0.8, n = 10,000, no censoring');fig.tight_layout();save(fig,'F3_grouped_weibull')
    print('PASS: generated F1, F2, F3 in PDF and PNG from frozen numerical logs; error bars are one corrected SE.')
if __name__=='__main__':main()
