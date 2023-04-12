import sys

import click
import numpy as np
import pandas as pd
import re

SOL_NEW_BEST = 'new best energy'
SOL_NEW = 'new energy'
SOL_DUPLICATE_BEST = 'best energy (dup)'
SOL_DUPLICATE = 'duplicate energy'
SOL_NOTHING = ''

sol_types = dict(
    NOTHING=0,  # nothing new, do nothing
    NEW_HIGH_ENERGY_SOL=1,  # ADDED by LL, used when generating annotations (v<=3)
    NEW_HIGH_ENERGY_UNIQUE_SOL=1,  # solution is unique, highest new energy
    DUPLICATE_HIGHEST_ENERGY=2,  # two cases, solution is unique, duplicate energy
    DUPLICATE_ENERGY=3,  # two cases, solution is duplicate highest energy
    DUPLICATE_ENERGY_UNIQUE_SOL=4,  # two cases, solution is unique, highest energy
    NEW_ENERGY_UNIQUE_SOL=5  # solution is unique, new highest energy
)
# colors = ['black', 'red', 'darkred', 'darkcyan', 'cornflowerblue', 'coral']
colors = ['black', 'crimson', 'darkred', 'darkcyan', 'cornflowerblue', 'coral']


def _annotations_from_answer_list(answers):
    current_best = 1E4
    annotations = []
    for ans in answers:
        if ans < current_best:
            current_best = ans
            annotations.append('NEW_HIGH_ENERGY_SOL')
        else:
            annotations.append('NOTHING')
    return annotations


class Iterator:

    def __init__(self, items):
        self.len = len(items)
        self.cursor = 0
        self.iter = iter(items)

    def next(self):
        if self.has_next():
            self.cursor += 1
            return next(self.iter)
        return None

    def has_next(self):
        return self.cursor < self.len

    def __len__(self):
        return self.len


_extract_time = lambda line: float(re.search('^\d+\.\d*', line)[0])
_extract_energy = lambda line: float(re.match('.*((loop =)|(answer  ))(-?\d+\.\d*)', line).groups()[-1])


def parse(lines):
    started = False
    verbose_3plus = True
    lines = Iterator(lines)

    best_energy = None
    answers = []
    times = []
    annotations = []

    try:
        while lines.has_next():
            line = lines.next()
            if not started:
                if 'Energy of solution' in line:
                    started = True
                    while 'Starting outer loop' not in line:
                        line = lines.next()  # skip the state dump
                    times.append(_extract_time(line))
                    answers.append(_extract_energy(line))
                    annotations.append('NEW_HIGH_ENERGY_UNIQUE_SOL')

            elif 'after partition pass' in line:
                while not 'Latest answer' in line:
                    line = lines.next()

                times.append(_extract_time(line))
                answers.append(_extract_energy(line))
                if verbose_3plus:
                    sol_type = lines.next().strip().split(' ')[0]
                    if sol_type not in sol_types:
                        verbose_3plus = False
                    else:
                        annotations.append(sol_type)
                while not 'V Best outer loop' in line:
                    line = lines.next()
                best_energy = _extract_energy(line)
    except:
        pass

    return times, answers, best_energy, (annotations if verbose_3plus else None)


def plot_energies(times, answers, annotations=None, filename=None):
    from plotly.offline import plot
    import plotly.graph_objs as go

    if annotations is None or len(annotations) != len(answers):
        annotations = _annotations_from_answer_list(answers)
    if filename is None:
        filename = 'temp-qbsolv.html'

    best_idx = answers.index(min(answers))
    title = '<b>qbsolv - solution evolution over time</b>'
    title += ('<br>' * 2) + '<br>'.join([f'{colors[sol_types[k]]} = <i>{k}</i>' for k in set(annotations)])

    traces = [go.Scatter(
        x=times,
        y=answers,
        hoverinfo='text+x+y',
        mode='lines+markers',
        name='solutions over time',
        line=dict(color='#D5D2D2'),
        text=[f'sol {i+1}/{len(answers)}' for i in range(len(answers))],
        marker=dict(size=4, color=[colors[sol_types[a]] for a in annotations]),
    )]

    layout = go.Layout(
        title=title,
        xaxis=dict(title='time (seconds)'),
        yaxis=dict(title='energy of solution'),
        # hovermode='closest',
        showlegend=False,
        annotations=[
            dict(
                x=times[best_idx],
                y=answers[best_idx],
                xref='x',
                yref='y',
                text='Best solution',
                showarrow=True,
                arrowhead=1,
                ax=3,
                ay=25
            )
        ]
    )
    plot(go.Figure(data=traces, layout=layout), filename=filename)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option('-i', '--input', type=str, help="Path to a log file.")
@click.option('-o', '--output', type=str, help="File path for the output csv file.")
@click.option('--plot/--no-plot', is_flag=True, default=True, help="Whether or not to generate plots.")
def cli(input, output, plot):
    if input is None:
        print('input missing. Try --help for details.')
        return
    if output is not None and not output.endswith('.csv'):
        print(f'error. output "{output}" should be a path to a csv file, i.e. ending with ".csv".')

    with open(input) as f:
        times, answers, best_energy, annotations = parse(f.readlines())
        if annotations is not None:
            df = pd.DataFrame(np.column_stack([times, answers, annotations]),
                              columns=['timestamp', 'energy', 'soltype'])
        else:
            df = pd.DataFrame(np.column_stack([times, answers]), columns=['timestamp', 'energy'])

        results_as_csv = df.to_csv(index=False)
        if output is not None:
            with open(output, 'w') as f:
                f.write(results_as_csv)
        else:
            print(results_as_csv)

        if plot:
            filename = output.replace('.csv', '.html') if output is not None else None
            plot_energies(times, answers, annotations, filename=filename)

        df.head()


if __name__ == "__main__":
    cli()
