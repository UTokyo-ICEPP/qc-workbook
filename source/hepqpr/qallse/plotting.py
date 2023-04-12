import plotly.graph_objs as go
from plotly.offline import plot

from .utils import track_to_xplets, merge_dicts, XpletType

pplot = plot
default_dims = ['x', 'y']


def set_notebook_mode():
    from plotly.offline import init_notebook_mode, iplot
    global pplot
    pplot = iplot
    init_notebook_mode(connected=True)


# ----------------- buttons and layers on the plots

## Topology (specific to the TML detector simulation used in the TrackML dataset)
barrel_layers = [
    dict(name='8-2', volume=8, layer=2, radius=32, z=455),
    dict(name='8-4', volume=8, layer=4, radius=72, z=455),
    dict(name='8-6', volume=8, layer=6, radius=116, z=455),
    dict(name='8-8', volume=8, layer=8, radius=172, z=455),

    dict(name='13-2', volume=13, layer=2, radius=260, z=1030),
    dict(name='13-4', volume=13, layer=4, radius=360, z=1030),
    dict(name='13-6', volume=13, layer=6, radius=500, z=1030),
    dict(name='13-8', volume=13, layer=8, radius=660, z=1030),

    dict(name='17-2', volume=17, layer=2, radius=820, z=1030),
    dict(name='17-4', volume=17, layer=4, radius=1020, z=1030),
]

# barrel_layer_colors = dict([(8, '#a6cee3'), (13, '#b3df8a'), (17, '#fb9a99')])
barrel_layer_colors = dict([(8, '#ccc'), (13, '#cccc8d'), (17, '#ccc')])

xy_layer_shapes = [dict(
    type='circle',
    x0=-layer['radius'], x1=layer['radius'], y0=-layer['radius'], y1=layer['radius'], opacity=.5, layer='below',
    line={'color': barrel_layer_colors[layer['volume']]}
) for layer in barrel_layers]

rz_layer_shapes = [dict(
    type='line',
    x0=-layer['z'], x1=layer['z'], y0=layer['radius'], y1=layer['radius'], opacity=.5, layer='below',
    line={'color': barrel_layer_colors[layer['volume']]}
) for layer in barrel_layers]


def _get_shapes(dims):
    if dims == ['x', 'y']: return xy_layer_shapes
    if dims == ['z', 'r']: return rz_layer_shapes
    return []


def _get_layers_button(dims, xpad=.1):
    shapes = _get_shapes(dims)
    return dict(type='buttons', buttons=[
        dict(label='hide layers', method='relayout', args=['shapes', []]),
        dict(label='show layers', method='relayout', args=['shapes', shapes]),
    ], direction='left', pad={'l':xpad}, xanchor='left', y=1.1, yanchor='top')

def _get_toggle_line_button(xpad=.1):
    return dict(type='buttons', buttons=[
        dict(label='hits+tracks', method='update', args=[{'mode': 'markers+lines'}]),
        dict(label='hits only', method='update', args=[{'mode': 'markers'}]),
    ], direction='left', pad={'l':xpad}, xanchor='left', y=1.1, yanchor='top')

def _get_ratio_button(xpad=.1):
    return dict(type='buttons', buttons=[
        dict(label='free', method='relayout', args=['yaxis.scaleanchor', None]),
        dict(label='fixed', method='relayout', args=['yaxis.scaleanchor', 'x']),
    ], direction='left', pad={'l':xpad}, xanchor='left', y=1.1, yanchor='top')

def _add_buttons(layout, dims):
    layout.updatemenus = [
        _get_layers_button(dims),
        _get_toggle_line_button(),
        _get_ratio_button()
    ]


# ----------------- plotting

def colorcycle():
    # cf https://github.com/plotly/plotly.py/blob/master/plotly/colors.py
    from plotly.colors import DEFAULT_PLOTLY_COLORS
    from itertools import cycle
    # return cycle(DEFAULT_PLOTLY_COLORS[:3] + DEFAULT_PLOTLY_COLORS[4:]) # drop red
    return cycle(DEFAULT_PLOTLY_COLORS)


_default_doublet_colors = ['red', 'darkseagreen', 'green', 'blue']


# ----------------- plotting

def create_trace(hits, t, dims=None, **trace_params):
    if dims is None:
        dims = default_dims

    coords = dict((ax, v) for _, ax, v in zip(dims, list('xyz'), hits.loc[t][dims].values.T))
    if len(dims) == 3:
        params = merge_dicts(dict(marker=dict(size=3), hoverinfo='name+text', **coords), trace_params)
        return go.Scatter3d(**params)
    else:
        return go.Scatter(**coords, **trace_params)


def show_plot(traces, dims, show_buttons, return_fig=False, filename=None, **layout_params):
    if dims is None: dims = default_dims
    ax_titles = dict((k + 'axis', dict(title=v)) for k, v in zip(list('xyz'), dims))
    if len(dims) == 3:  # 3D
        params = dict(showlegend=True, width=900, height=900, hovermode='closest',
                      scene=dict(
                          **ax_titles,
                          camera=dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0),
                                      eye=dict(x=0.1, y=2.5, z=0.1))))
        if len(layout_params) > 0: params = merge_dicts(params, layout_params)
        layout = go.Layout(**params)
        if show_buttons:
            layout.updatemenus = [
                _get_toggle_line_button(),
                _get_ratio_button(xpad=200)
            ]
    else:  # 2D
        params = dict(showlegend=True, width=800, height=800, hovermode='closest', **ax_titles)
        if len(layout_params) > 0: params = merge_dicts(params, layout_params)
        layout = go.Layout(**params)
        if show_buttons:
            layout.updatemenus = [
                _get_layers_button(dims),
                _get_toggle_line_button(xpad=200),
                _get_ratio_button(xpad=380)
            ]
    fig = go.Figure(traces, layout)
    if filename is not None: pplot(fig, filename=filename)
    else: pplot(fig)

    return fig if return_fig else None


def iplot_results(dw, tracks, missing=None, dims=None, show_buttons=True, **kwargs):
    from collections import OrderedDict
    xplet_types = OrderedDict((i.name, []) for i in XpletType)
    _ = [xplet_types[dw.is_real_xplet(t).name].append(t) for t in tracks]
    if missing:
        xplet_types['MISSING'] = missing

    data = []
    for (name, subset), col in zip(xplet_types.items(), _default_doublet_colors):
        show_legend = True
        name = name.lower()
        for t in subset:
            data.append(
                create_trace(dw.hits, t, dims,
                             text=t, hoverinfo='name+text', legendgroup=name, name=name,
                             showlegend=show_legend, opacity=1, line=dict(color=col, width=1))
            )
            show_legend = False

    return show_plot(data, dims, show_buttons, **kwargs)


def iplot_results_tracks(dw, tracks, dims=None, show_buttons=True, **kwargs):
    traces = []
    first_real = True

    for idx, t in enumerate(tracks):
        doublets = track_to_xplets(t, x=2)
        status = [dw.is_real_doublet(s) for s in doublets]
        is_real = all(status)

        name = f'T{idx}: real' if is_real \
            else f'T{idx}: {sum(map(bool, status))}/{len(doublets)} ok'

        for i, (dblet, ok) in enumerate(zip(doublets, status)):
            traces.append(
                create_trace(dw.hits, dblet, dims,
                             text=t, hoverinfo='name+text',
                             legendgroup=is_real or idx, name=name,
                             showlegend=i == 0 and (not is_real or first_real),
                             opacity=1,
                             line=dict(color=_default_doublet_colors[status[i]], width=1))
            )
            if is_real: first_real = False
    return show_plot(traces, dims, show_buttons, **kwargs)


def iplot_any(hits, tracks, dims=None, show_buttons=True, line_color=None, **kwargs):
    traces = []

    for idx, t in enumerate(tracks):
        traces.append(
            create_trace(hits, t, dims,
                         text=t, hoverinfo='text',
                         line=dict(color=line_color, width=1))
        )

    return show_plot(traces, dims, show_buttons, **kwargs)
