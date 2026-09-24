"""Plot saved comparisons only. No solver runs are launched by this module."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

METHODS = ('IMEX_RK2', 'SDIRK2_mr_SAV')
LABELS = ('SDIRK2', 'SDIRK2-mr-SAV')
COLORS = ('#D55E00', '#0072B2')


def load_run(folder, run_id, case=None):
    path = Path(folder)/f'{run_id}.npz'
    if not path.exists():
        raise FileNotFoundError(f'Missing saved run: {path}; run computation explicitly.')
    with np.load(path, allow_pickle=False) as f:
        result = {k: f[k].copy() for k in f.files}
    meta = json.loads(str(result['metadata']))
    if meta['run_id'] != run_id or (case is not None and meta['config']['case'] != case):
        raise ValueError(f'Incompatible metadata: {path}')
    return result, meta


def plot_comparison(path, destination, selected_steps=None):
    path, destination = Path(path), Path(destination)
    summary = json.loads(path.read_text())
    destination.mkdir(parents=True, exist_ok=True)
    c = summary['case']
    ref, ref_meta = load_run(path.parent, summary['reference_id'], c)
    if ref_meta['status'] != 'completed':
        raise ValueError('Reference is incomplete or failed.')
    rows = sorted(summary['records'], key=lambda r: r['tau'])
    for row in rows:
        for method in METHODS:
            _, meta = load_run(path.parent, row['methods'][method]['run_id'], c)
            if meta['status'] != row['methods'][method]['status']:
                raise ValueError('Summary and run status disagree.')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.subplots_adjust(left=.065, right=.98, bottom=.085, top=.90, wspace=.32, hspace=.40)
    tau = np.array([r['tau'] for r in rows])
    def error(row, method, field, index=-1):
        result = row['methods'][method]
        if result['status'] != 'completed':
            return np.nan
        target = ref['times'][1:][index]
        matching = [item for item in result['errors'] if np.isclose(item['time'], target, rtol=0, atol=1e-12)]
        return matching[0][field] if len(matching) == 1 else np.nan
    for j, method in enumerate(METHODS):
        for ax, field in zip(axes[0,:2], ('omega_relative', 'velocity_relative')):
            values = [error(r, method, field) for r in rows]
            ax.loglog(tau, values, '-o', ms=4, color=COLORS[j], label=LABELS[j])
            failed = [r['tau'] for r in rows if r['methods'][method]['status'] != 'completed']
            ax.plot(failed, [.96-j*.07]*len(failed), 'x', color=COLORS[j],
                    transform=ax.get_xaxis_transform(), clip_on=False,
                    label=f'{LABELS[j]} failed' if failed else None)
    for ax, title in zip(axes[0,:2], ('Terminal vorticity error', 'Terminal velocity error')):
        ax.set(xlabel=r'Fixed step $\tau$', ylabel=r'Relative $L^2$ error', title=title)
        ax.legend(fontsize=8)
    for k, t in enumerate(ref['times'][1:]):
        ratios = [error(r, METHODS[1], 'omega_relative', k)/error(r, METHODS[0], 'omega_relative', k)
                  for r in rows]
        axes[0,2].semilogx(tau, ratios, '-o', ms=3, label=f't={t:g}')
    axes[0,2].axhline(1, color='0.5', ls='--')
    axes[0,2].axhline(.8, color='0.7', ls=':')
    axes[0,2].set(xlabel=r'Fixed step $\tau$', ylabel='mr-SAV / SDIRK2 error', title='Vorticity error ratio')
    axes[0,2].legend(fontsize=8)
    if selected_steps is None:
        candidates = [r for r in rows if all(r['methods'][m]['status'] == 'completed' for m in METHODS)]
        row = min(candidates, key=lambda r: error(r,METHODS[1],'omega_relative')/error(r,METHODS[0],'omega_relative'))
    else:
        row = next(r for r in rows if r['steps'] == selected_steps)
    for j, method in enumerate(METHODS):
        value, meta = load_run(path.parent, row['methods'][method]['run_id'], c)
        d = value['diagnostics']
        axes[1,0].semilogy(d[:,0], d[:,2], color=COLORS[j], label=LABELS[j]+' ('+meta['status']+')')
        axes[1,1].plot(d[:,0], np.abs(d[:,1]-1), color=COLORS[j], label=LABELS[j])
    axes[1,0].plot(d[:,0], d[:,-1], 'k--', lw=1, label='Forced enstrophy bound')
    axes[1,0].plot(ref['times'],np.sqrt(np.mean(ref['omega']**2,axis=(1,2))), 'ks',ms=3,label='ETDRK4 reference')
    axes[1,0].set(xlabel='Time', ylabel='Vorticity RMS', title=f"Selected step: {row['tau']:.6g}")
    axes[1,0].legend(fontsize=7)
    axes[1,1].set(xlabel='Time',ylabel=r'$|q-1|$',title='Auxiliary variable deviation')
    axes[1,1].legend(fontsize=8)
    status_rows = []
    for r in rows:
        status_rows.append([f"{r['tau']:.5g}", *['OK' if r['methods'][m]['status']=='completed'
                            else r['methods'][m]['status'].replace('solution_blowup','blowup') for m in METHODS]])
    axes[1,2].axis('off')
    table = axes[1,2].table(cellText=status_rows, colLabels=['step','SDIRK2','mr-SAV'],
                          bbox=[0,0,1,1],cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(8)
    axes[1,2].set_title('Completion status')
    for ax in axes[0,:]:
        ax.xaxis.set_major_locator(FixedLocator([.01,.02,.05,.1,.2,.5,1]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(tau.min()*.9,tau.max()*1.12)
    for ax in axes.flat[:-1]:
        ax.grid(alpha=.2)
    fig.suptitle(f"Forced periodic NS: N={c['grid']}, nu={c['nu']:g}, gamma={c['gamma']:g}, "
                 f"A={c['amplitude']:g}, F={c['force']:g}, {c['initial']}")
    fig.text(.98,.02,'OK denotes completion, not accuracy. Crosses mark stopped runs.',ha='right',fontsize=9)
    fig.savefig(destination/'comparison.png',dpi=220)
    fig.savefig(destination/'comparison.svg')
    plt.close(fig)
    # Matched physical time and common color scale; error maps expose damping.
    fields = [ref['omega'][-1]]
    titles = ['ETDRK4 reference']
    for method,label in zip(METHODS,LABELS):
        value,meta = load_run(path.parent,row['methods'][method]['run_id'],c)
        if meta['status'] != 'completed':
            continue
        fields.append(value['omega'][-1]);titles.append(label)
    fig, axes = plt.subplots(2,len(fields),figsize=(4*len(fields),7),squeeze=False,layout='constrained')
    vmax = max(float(np.max(np.abs(w))) for w in fields)
    emax = max([float(np.max(np.abs(w-fields[0]))) for w in fields[1:]]+[1e-15])
    for j,(w,title) in enumerate(zip(fields,titles)):
        im=axes[0,j].imshow(w,origin='lower',extent=(0,2*np.pi,0,2*np.pi),cmap='RdBu_r',vmin=-vmax,vmax=vmax)
        axes[0,j].set_title(title)
        er=axes[1,j].imshow(w-fields[0],origin='lower',extent=(0,2*np.pi,0,2*np.pi),cmap='RdBu_r',vmin=-emax,vmax=emax)
        axes[1,j].set_title('Vorticity error')
    fig.colorbar(im,ax=axes[0,:],shrink=.8)
    fig.colorbar(er,ax=axes[1,:],shrink=.8)
    fig.suptitle(f"t={c['final_time']:g}, step={row['tau']:.7g}, N={c['grid']}")
    fig.savefig(destination/'fields.png',dpi=200);fig.savefig(destination/'fields.svg')
    plt.close(fig)
    return destination/'comparison.png'


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('comparison',type=Path)
    p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--selected-steps',type=int)
    a=p.parse_args()
    print(plot_comparison(a.comparison,a.destination,a.selected_steps))
