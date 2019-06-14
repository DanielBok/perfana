import os
from typing import Dict, List, Union

import plotly.graph_objs as go
from plotly.offline import offline as o

from ._type import GRAPH_DESC

__all__ = ["plot"]

Graph = Union[
    go.Layout,
    go.Waterfall,
    go.Volume,
    go.Violin,
    go.Table,
    go.Surface,
    go.Sunburst,
    go.Streamtube,
    go.Splom,
    go.Scatterternary,
    go.Scatterpolargl,
    go.Scatterpolar,
    go.Scattermapbox,
    go.Scattergl,
    go.Scattergeo,
    go.Scattercarpet,
    go.Scatter3d,
    go.Scatter,
    go.Sankey,
    go.Pointcloud,
    go.Pie,
    go.Parcoords,
    go.Parcats,
    go.Ohlc,
    go.Mesh3d,
    go.Isosurface,
    go.Histogram2dContour,
    go.Histogram2d,
    go.Histogram,
    go.Heatmapgl,
    go.Heatmap,
    go.Contourcarpet,
    go.Contour,
    go.Cone,
    go.Choropleth,
    go.Carpet,
    go.Candlestick,
    go.Box,
    go.Barpolar,
    go.Bar,
    go.Area,
    go.Frame,
]


def plot(figure_or_data: Union[go.Figure, go.Data, Dict[str, Graph], List[Graph]],
         validate=True,
         output_type='file',
         include_plotlyjs="cdn",
         auto_open=True,
         filename: str = None,
         image: str = None,
         image_filename='plot_image',
         image_width=800,
         image_height=600,
         config=None,
         include_mathjax=False,
         auto_play=True,
         animation_opts=None,
         is_app_mode=False) -> GRAPH_DESC:
    """
    Plots the chart specified.

    If running in app, set the environment variable 'RUNNING_APP' to '1' or 'TRUE'.
    This will generate a dictionary that can be converted to a JSON payload which
    can then be fed into the Plotly.js object to generate the graph.

    Parameters
    ----------
    figure_or_data
        A plotly.graph_objs.Figure or plotly.graph_objs.Data or dict or
        list that describes a Plotly graph. See https://plot.ly/python/
        for examples of graph descriptions.

    validate: bool
        If True, validates that all of the keys in the figure are valid. Omit if
        the version of plotly.js has become outdated with the version of
        graph_reference.json or if there is a need to to include extra, unnecessary
        keys in the figure.

    output_type
        If 'file', then the graph is saved as a standalone HTML file and `plot`
        returns None. If 'div', then `plot` returns a string that just contains the
        HTML <div> that contains the graph and the script to generate the graph.

        Use 'file' if you want to save and view a single graph at a time
        in a standalone HTML file.

        Use 'div' if you are embedding these graphs in an HTML file with
        other graphs or HTML markup, like a HTML report or an website.

    include_plotlyjs
        Specifies how the plotly.js library is included in the output html
        file or div string.

        If True, a script tag containing the plotly.js source code (~3MB)
        is included in the output.  HTML files generated with this option are
        fully self-contained and can be used offline.

        If 'cdn', a script tag that references the plotly.js CDN is included
        in the output. HTML files generated with this option are about 3MB
        smaller than those generated with include_plotlyjs=True, but they
        require an active internet connection in order to load the plotly.js
        library.

        If 'directory', a script tag is included that references an external
        plotly.min.js bundle that is assumed to reside in the same
        directory as the HTML file.  If output_type='file' then the
        plotly.min.js bundle is copied into the directory of the resulting
        HTML file. If a file named plotly.min.js already exists in the output
        directory then this file is left unmodified and no copy is performed.
        HTML files generated with this option can be used offline, but they
        require a copy of the plotly.min.js bundle in the same directory.
        This option is useful when many figures will be saved as HTML files in
        the same directory because the plotly.js source code will be included
        only once per output directory, rather than once per output file.

        If a string that ends in '.js', a script tag is included that
        references the specified path. This approach can be used to point
        the resulting HTML file to an alternative CDN.

    auto_open
        If True, open the saved file in a web browser after saving.
        This argument only applies if `output_type` is 'file'.

    filename
        The local filename to save the outputted chart to. If the filename already
        exists, it will be overwritten. This argument only applies if `output_type`
        is 'file'.

    image
        This parameter sets the format of the image to be downloaded. This
        parameter has a default value of None indicating that no image should
        be downloaded.

    image_filename
        Sets the name of the file your image will be saved to. The extension
        should not be included.

    image_width
        Specifies the width of the image in `px`.

    image_height
        Specifies the height of the image in `px`.

    config
        Plot view options dictionary. Keyword arguments `show_link` and
        `link_text` set the associated options in this dictionary if it
        doesn't contain them already.

    include_mathjax
        Specifies how the MathJax.js library is included in the output html
        file or div string.  MathJax is required in order to display labels
        with LaTeX typesetting.

        If False, no script tag referencing MathJax.js will be included in the
        output. HTML files generated with this option will not be able to
        display LaTeX typesetting.

        If 'cdn', a script tag that references a MathJax CDN location will be
        included in the output.  HTML files generated with this option will be
        able to display LaTeX typesetting as long as they have internet access.

        If a string that ends in '.js', a script tag is included that
        references the specified path. This approach can be used to point the
        resulting HTML file to an alternative CDN.

    auto_play
        Whether to automatically start the animation sequence on page load if
        the figure contains frames. Has no effect if the figure does not contain
        frames.

    animation_opts
        Dict of custom animation parameters that are used for the automatically
        started animation on page load. This dict is passed to the function
        Plotly.animate in Plotly.js. See
        https://github.com/plotly/plotly.js/blob/master/src/plots/animation_attributes.js
        for available options. Has no effect if the figure does not contain
        frames, or auto_play is False.

    is_app_mode
        If app mode is True, instead of plotting graphs, this function will return the
        dictionary describing the figure. For app mode, the figure_or_data argument must
        be a plotly.graph_obj.Figure instance.

    Returns
    -------
    Any
        The allocation chart object if on a python console or Jupyter notebook.
        If used in app, returns the dictionary to form the graphs on the frontend
    """

    is_app_env = os.environ.get('RUNNING_APP', '0').upper() in ('1', 'TRUE')
    if is_app_mode or is_app_env:
        assert isinstance(figure_or_data, go.Figure)
        return figure_or_data.to_dict()

    if _on_jupyter():
        if filename is None:
            filename = 'plot_image'
        return o.iplot(figure_or_data, validate=validate, image=image, filename=filename,
                       image_width=image_width, image_height=image_height, config=config,
                       auto_play=auto_play, animation_opts=animation_opts)

    if filename is None:
        filename = 'temp-plot.html'

    return o.plot(figure_or_data, validate=validate, output_type=output_type,
                  include_plotlyjs=include_plotlyjs, filename=filename, auto_open=auto_open,
                  image=image, image_filename=image_filename, image_width=image_width,
                  image_height=image_height, config=config, include_mathjax=include_mathjax,
                  auto_play=auto_play, animation_opts=animation_opts)


def _on_jupyter():
    """Returns True if on Jupyter Notebook"""
    try:
        # noinspection PyUnresolvedReferences
        return get_ipython().has_trait('kernel')
    except NameError:
        return False
