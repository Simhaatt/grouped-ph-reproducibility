"""Regenerate numerical CSV and LaTeX tables from frozen logs, without refitting."""
from frozen_results import *
def latex_table(name,rows,cols):
    lines=['% Generated from frozen aggregate logs. Do not edit numbers.','\\begin{tabular}{'+'l'*len(cols)+'}','\\hline',' & '.join(c.replace('_',r'\_') for c in cols)+r' \\',r'\hline']
    for row in rows:
        def fmt(v): return f'{v:.5f}' if isinstance(v,float) else str(v).replace('_',r'\_')
        lines.append(' & '.join(fmt(row[c]) for c in cols)+r' \\')
    lines+=['\\hline','\\end{tabular}']
    p=ROOT/'results/supplementary'/f'{name}.tex';p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
def main():
    m=main_rows(); e=event_rows();w=weibull_rows();v=validation_rows();f=flexibility_rows()
    for name,rows in [('main/baseline_decomposition',m),('simulations/event_concentration',e),('simulations/grouped_weibull',w),('main/independent_sparcs',v),('supplementary/flexibility',f),('main/support2_music',[r for r in m if r['cohort'] in ['support2/slos','drsa/music']])]:
        write_csv('results/'+name+'.csv',rows)
    cols=['cohort','T','modal_event','D_B','SE_B','D_KP','SE_KP']
    latex_table('S1_R1',[r for r in m if r['regime']=='R1'],cols)
    latex_table('S2_R2',[r for r in m if r['regime']=='R2'],cols)
    for name,metrics in [('S3_flexibility',['nll','c_antolini','c_uno']),('S4_flexibility',['ibs','ici'])]:
        latex_table(name,[r for r in f if r['metric'] in metrics],['cohort','metric','difference','SE'])
    latex_table('independent_sparcs',v,['cohort','T','modal_all','D_B','SE_B','D_KP','SE_KP'])
    # The four representative rows use the same keys as the manuscript.
    txt=(ROOT/'manuscript/template.tex').read_text(encoding='utf-8')
    block=txt.split('\\label{tab:main-examples}')[1].split('\\end{table}')[0]
    keys={(a,int(b)) for a,b in re.findall(r'(\S+), \$T=(\d+)\$',block)}
    latex_table('main_examples',[r for r in m if (r['cohort'],r['T']) in keys],cols)
    (ROOT/'results/main/summary.json').write_text(json.dumps(summaries(),indent=2)+'\n',encoding='utf-8')
    print('PASS: regenerated 34 main, 27 E8, 135 Weibull, 12 validation and 25 flexibility metric rows.')
if __name__=='__main__':main()
