#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Draw the current README figures exclusively from the headline-r1 public records.

uv run bench/plot.py
Missing files, missing cells and numeric inconsistencies are fatal. Historical
figures are retained as dated artifacts and are not inputs to these figures.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from headline import load

OUT = Path(__file__).resolve().parent.parent / 'docs/img'
COLORS = {1: '#0969da', 4: '#8250df', 8: '#cf222e'}
plt.rcParams.update({
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelcolor': '#57606a', 'axes.edgecolor': '#d8dee4',
    'svg.fonttype': 'none', 'svg.hashsalt': 'headline-r1-20261116',
})


def save(fig, name, title, note):
    fig.suptitle(title, fontweight='bold', fontsize=12)
    fig.text(.5, .015, note, ha='center', fontsize=8, color='#57606a')
    fig.tight_layout(rect=(0, .055, 1, .94))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, format='svg', bbox_inches='tight', metadata={'Title': title, 'Date': None})
    plt.close(fig)
    path = OUT / name
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')


def style(ax, ylabel, xs=(1, 4, 8)):
    ax.set_xticks(xs)
    ax.set_xlabel('Concurrent streams')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=.2)
    ax.set_ylim(bottom=0)


def figure_std_bench(real):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8))
    for ax, field, label in zip(axes.flat,
            ('output_tok_s', 'per_stream_tok_s', 'ttft_p50_ms', 'ttft_p99_ms'),
            ('Output / benchmark wall (tok/s)', 'Per stream: 1000 / TPOT p50 (tok/s)', 'TTFT p50 (ms)', 'TTFT p99 (ms)')):
        for size, ls, color in (('subset', '-', '#0969da'), ('full', '--', '#8250df')):
            ys = [real[size, c][field] for c in (1, 4, 8)]
            ax.plot([1, 4, 8], ys, ls, marker='o', color=color, label='120 subset' if size == 'subset' else '480 full check')
            if size == 'subset':
                for x, y in zip((1, 4, 8), ys):
                    ax.annotate(f'{y:.1f}', (x, y), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=8)
        style(ax, label)
        ax.margins(y=.16)
        ax.legend(fontsize=8, loc='best')
    save(fig, 'std-bench.svg', 'Real work · Spec-Bench · 2026-11-16',
         '2.68 bpw + HermiHg MTP · names-fast r1, default name · T=0 · 256 forced tokens · template-default thinking')


def figure_decode_scaling(code):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    for ax, field, label in zip(axes,
            ('aggregate_tok_s', 'mean_per_stream_tok_s'),
            ('Sum of common-window rates (tok/s)', 'Mean per-stream common-window rate (tok/s)')):
        for append, ls, color in ((2048, '-', '#0969da'), (8192, '--', '#8250df')):
            ys = [code[1024, append, c]['window'][field] for c in (1, 4, 8)]
            ax.plot([1, 4, 8], ys, ls, marker='o', color=color, label=f'{append // 1024}K append')
            for x, y in zip((1, 4, 8), ys):
                ax.annotate(f'{y:.1f}', (x, y), xytext=(0, 8 if append == 2048 else -13), textcoords='offset points', ha='center', fontsize=8)
        style(ax, label)
        ax.margins(y=.2)
        ax.legend(fontsize=8, loc='best')
    save(fig, 'decode-scaling.svg', 'Peak realistic code · cached 1K prefix · 2026-11-16',
         'Primary peak = 2K append · 400 forced tokens · thinking off · T=0 · shared context · canonical token counts')


def figure_prefill(code):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    prefixes = [1024, 32768, 65536]
    # Discrete measured targets only; lines join those targets without new observations.
    xs = [0, 1, 2]
    for c in (1, 4, 8):
        for append, ls in ((2048, '-'), (8192, '--')):
            cells = [code[p, append, c] for p in prefixes]
            label = f'c{c}, {append // 1024}K append'
            axes[0].plot(xs, [r['append_ttft_p50_ms'] for r in cells], ls, marker='o', color=COLORS[c], label=label)
            axes[1].plot(xs, [r['window']['mean_per_stream_tok_s'] for r in cells], ls, marker='o', color=COLORS[c], label=label)
    for ax, ylabel in zip(axes, ('Cached-append TTFT p50 (ms)', 'Mean per-stream code decode (tok/s)')):
        ax.set_xticks(xs, ['1K', '32K', '64K'])
        ax.set_xlabel('Cached-prefix target (actual decode depth includes append)')
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
        ax.legend(fontsize=8, loc='center', bbox_to_anchor=(.5, .66), ncols=2) if ax is axes[0] else ax.legend(fontsize=8, loc='lower left', ncols=2)
    save(fig, 'prefill.svg', 'Cached append and code at depth · 2026-11-16',
         '2K solid / 8K dashed append · c1/c4/c8 · shared context · T=0 · 400 forced tokens · thinking off · p99 in numeric matrix')


if __name__ == '__main__':
    real, code, rr, cc = load()
    figure_std_bench(rr)
    figure_decode_scaling(cc)
    figure_prefill(cc)
    for r in real['table']:
        print(r['tag'], {k: r[k] for k in ('output_tok_s', 'per_stream_tok_s', 'ttft_p50_ms', 'ttft_p99_ms', 'draft_acceptance')})
    for r in code['table']:
        print(r['tag'], r['window']['aggregate_tok_s'], r['window']['mean_per_stream_tok_s'], r['append_ttft_p50_ms'], r['append_ttft_p99_ms'])
    print('wrote std-bench.svg, decode-scaling.svg, prefill.svg from validated headline-r1 records')
