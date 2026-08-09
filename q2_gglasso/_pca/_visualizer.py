import os
import zarr
import jinja2
import warnings
import itertools
import numpy as np
import pandas as pd
import qiime2

from biom.table import Table
from q2_gglasso.utils import PCA, get_seq_depth

from bokeh.plotting import figure
# bokeh 3 renamed the tab child container Panel -> TabPanel. `Panel` still
# imports successfully under bokeh 3 but resolves to an unrelated Panel(Pane)
# class, failing only later at construction -- so import TabPanel explicitly.
from bokeh.models import (
    ColumnDataSource,
    LinearColorMapper,
    ColorBar,
    Select,
    CustomJS,
    TabPanel,
    Tabs,
)
from bokeh.layouts import gridplot, row
from bokeh.palettes import Blues8
from bokeh.embed import components
from bokeh.resources import INLINE


def project_covariates(
    counts=pd.DataFrame(), metadata=pd.DataFrame(), L=np.ndarray, color_by=str
):
    """Project data onto principal component space with covariates.

    Parameters
    ----------
    counts : pd.DataFrame
        The count data to project.
    metadata : pd.DataFrame
        The metadata containing covariates.
    L : np.ndarray
        Low rank component from the GGLasso solution.
    color_by : str
        The metadata column to use for coloring the projection.

    Returns
    -------
    bokeh.layouts.row
        A Bokeh layout containing the PCA projection plot.

    """
    proj, loadings, eigv = PCA(counts.dropna(), L, inverse=True)
    r = np.linalg.matrix_rank(L)
    eigv_sum = np.sum(eigv)
    var_exp = [(value / eigv_sum) for value in sorted(eigv, reverse=True)]

    pc_columns = list(
        "PC{0} ({1}%)".format(i + 1, str(100 * var_exp[i])[:4]) for i in range(0, r)
    )
    df_proj = pd.DataFrame(proj, columns=pc_columns, index=counts.index)
    df = df_proj.join(metadata)

    varName1 = "PC1 ({0}%)".format(str(100 * var_exp[0])[:4])
    varName2 = "PC2 ({0}%)".format(str(100 * var_exp[1])[:4])
    df["x"] = df[varName1]
    df["y"] = df[varName2]

    source = ColumnDataSource(df)

    p0 = figure(
        tools="save, zoom_in, zoom_out, wheel_zoom, box_zoom, reset",
        # bokeh 3 removed plot_width/plot_height in favour of width/height.
        width=800,
        height=800,
        active_scroll="wheel_zoom",
        x_axis_label=varName1,
        y_axis_label=varName2,
        tooltips=[(varName1, "@" + varName1), (varName2, "@" + varName2)],
        title=varName1 + " vs " + varName2,
    )

    exp_cmap = LinearColorMapper(
        palette=Blues8[::-1],
        low=min(df[color_by].values),
        high=max(df[color_by].values),
    )
    # figure.circle(size=...) is deprecated since bokeh 3.4 in favour of scatter().
    p0.scatter(
        "x",
        "y",
        source=source,
        size=15,
        line_color=None,
        fill_color={"field": color_by, "transform": exp_cmap},
        fill_alpha=0.3,
    )

    options = list(df.columns)
    options.remove("x")
    options.remove("y")

    select1 = Select(title="X-Axis:", value=varName1, width=100, options=options)
    select2 = Select(title="Y-Axis:", value=varName2, width=100, options=options)

    changeVariables = CustomJS(
        args=dict(
            plot=p0,
            source=source,
            select1=select1,
            select2=select2,
            ax1=p0.xaxis,
            ax2=p0.yaxis,
        ),
        code="""
        var varName1 = select1.value;
        var varName2 = select2.value;
        plot.title.text = varName1 + " vs " + varName2;
        ax1[0].axis_label = varName1;
        ax2[0].axis_label = varName2;
        source.data['x'] = source.data[varName1];
        source.data['y'] = source.data[varName2];
        source.change.emit();
    """,
    )

    select1.js_on_change("value", changeVariables)
    select2.js_on_change("value", changeVariables)

    color_bar_plot = figure(
        title=color_by,
        title_location="right",
        height=500,
        width=150,
        toolbar_location=None,
        min_border=0,
        outline_line_color=None,
    )

    bar = ColorBar(color_mapper=exp_cmap, location=(1, 1))

    color_bar_plot.add_layout(bar, "right")
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = "12pt"

    layout_1 = row(p0, color_bar_plot, select1, select2)

    return layout_1


def add_color_bar(color_map: LinearColorMapper, title: str = None):
    """Create a color bar for the plot.

    Parameters
    ----------
    color_map : LinearColorMapper
        The color mapper to use for the color bar.
    title : str, optional
        The title of the color bar.

    Returns
    -------
    bokeh.plotting.figure
        A figure containing the color bar.

    """
    color_bar_plot = figure(
        title=title,
        title_location="right",
        height=50,
        width=50,
        toolbar_location=None,
        min_border=0,
        outline_line_color=None,
    )

    bar = ColorBar(color_mapper=color_map, location=(1, 1))

    color_bar_plot.add_layout(bar, "right")
    color_bar_plot.title.align = "center"
    color_bar_plot.title.text_font_size = "12pt"

    return color_bar_plot


def make_plots(
    df: pd.DataFrame,
    col_name: str = None,
    n_components=int,
    proj=np.ndarray,
    eigv=np.ndarray,
):
    """Create a grid of PCA plots.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data.
    col_name : str, optional
        The column name to use for coloring.
    n_components : in
        The number of principal components to plot.
    proj : np.ndarray
        The projected data points.
    eigv : np.ndarray
        The eigenvalues for the principal components.

    Returns
    -------
    bokeh.layouts.gridplo
        A grid of PCA projection plots.

    """
    t = np.arange(n_components)
    comb = list(itertools.combinations(t, 2))

    bokeh_tools = ["save, zoom_in, zoom_out, wheel_zoom, box_zoom, reset, hover"]

    grid = np.empty([n_components, n_components], dtype=object)
    eigv_sum = np.sum(eigv)
    var_exp = [(value / eigv_sum) for value in sorted(eigv, reverse=True)]

    for i, j in comb:
        p = figure(tools=bokeh_tools, toolbar_location="above")

        source = ColumnDataSource(
            {"x": proj[:, i], "y": proj[:, j], "col": df[col_name].values}
        )

        exp_cmap = LinearColorMapper(
            palette="Viridis256",
            low=min(df[col_name].values),
            high=max(df[col_name].values),
        )

        # circle(size=...) deprecated since bokeh 3.4 in favour of scatter().
        p.scatter(
            "x",
            "y",
            size=5,
            source=source,
            line_color=None,
            fill_color={"field": "col", "transform": exp_cmap},
        )

        p.xaxis.axis_label = "PC{0} ({1}%)".format(i + 1, str(100 * var_exp[i])[:4])
        p.yaxis.axis_label = "PC{0} ({1}%)".format(j + 1, str(100 * var_exp[j])[:4])

        grid[i][j] = p

        grid[-1][-1] = add_color_bar(color_map=exp_cmap, title=col_name)

    x = np.transpose(grid)
    x = x.flatten()
    layout_2 = gridplot(
        children=list(x), ncols=int(n_components), width=250, height=250
    )

    return layout_2


def pair_plot(
    counts: pd.DataFrame(),
    metadata=pd.DataFrame(),
    L=np.ndarray,
    n_components=int,
    color_by=str,
):
    """Create a pair plot of PCA projections.

    Parameters
    ----------
    counts : pd.DataFrame
        The count data to project.
    metadata : pd.DataFrame
        The metadata containing covariates.
    L : np.ndarray
        Low rank component from the GGLasso solution.
    n_components : in
        The number of principal components to include.
    color_by : str
        The metadata column to use for coloring the projection.

    Returns
    -------
    bokeh.layouts.row
        A row layout containing the pair plot.

    """
    # TO DO: Add widget for columns selection
    plot_dict = dict()
    for col in metadata.columns:
        plot_df = counts.join(metadata[col])
        plot_df = plot_df.dropna()

        proj, loadings, eigv = PCA(plot_df.iloc[:, :-1], L, inverse=True)
        r = np.linalg.matrix_rank(L)

        # `<=`, not `<`: requesting exactly as many components as the achieved
        # rank is legitimate. pca() validates this up front too, so by here it
        # should already hold -- this stays as a defensive check for direct
        # callers of pair_plot().
        assert n_components <= r, (
            f"n_components ({n_components}) is greater than the rank of the "
            f"low-rank component ({r})"
        )

        plot_dict[col] = make_plots(
            df=plot_df,
            col_name=col,
            n_components=n_components,
            proj=proj,
            eigv=eigv,
        )

    if color_by is None:
        warnings.warn(
            "Coloring covariate has not been selected, "
            "the first entry from the following list will be used:{0}".format(
                metadata.columns
            )
        )
        color_by = metadata.columns[0]
        p2 = row(plot_dict[color_by])
    else:
        p2 = row(plot_dict[color_by])

    return p2


def pca(
    output_dir: str,
    table: Table,
    solution: zarr.hierarchy.Group,
    n_components: int = 3,
    color_by: str = None,
    sample_metadata: qiime2.Metadata = None,
):
    """Generate PCA visualization from GGLasso solution.

    Parameters
    ----------
    output_dir : str
        The directory where visualization files will be written.
    table : Table
        The feature table containing sample data.
    solution : zarr.hierarchy.Group
        The GGLasso solution containing the low-rank component.
    n_components : int, optional
        The number of principal components to display, by default 3.
    color_by : str, optional
        The metadata column to use for coloring points, by default None.
    sample_metadata : qiime2.Metadata, optional
        The sample metadata, by default None.

    """
    J_ENV = jinja2.Environment(loader=jinja2.PackageLoader("q2_gglasso._pca", "assets"))
    template = J_ENV.get_template("index.html")
    df = table.to_dataframe()
    # Materialise as an ndarray: this is handed to np.linalg.eigh / matrix_rank,
    # and numpy 2 calls __array__(dtype, copy=...) which older zarr versions
    # reject. Being explicit here removes the dependency on zarr's __array__.
    L = np.asarray(solution["solution/lowrank_"])

    # Validate the low-rank component BEFORE any plotting. Without this, a
    # rank-0 L reaches np.linalg inside PCA() and surfaces as
    #   "matmul: Input operand 1 has a mismatch in its core dimension 0
    #    (size 0 is different from 1)"
    # which says nothing about the actual problem. The only pre-existing guard
    # lived inside pair_plot(), i.e. after project_covariates() had already
    # crashed, so it could never fire.
    rank = int(np.linalg.matrix_rank(L))
    if rank == 0:
        raise ValueError(
            "The low-rank component of this solution has rank 0, so there is "
            "no latent subspace to project onto. Model selection chose a mu1 "
            "large enough to shrink the low-rank block away entirely, which "
            "means no latent structure was detected. Re-solve with a smaller "
            "mu1 (smaller mu1 permits a larger rank) and check the achieved "
            "rank before running pca."
        )
    # Rank 1 is not just "one component fewer": utils.PCA squeezes a single
    # eigenvalue to a 0-d array, so sorted(eigv) in project_covariates() raises
    # "'numpy.float64' object is not iterable", and both tabs plot PC1 against
    # PC2, which a one-dimensional subspace cannot supply. Lowering
    # n_components does not help, so say so instead of advising it below.
    if rank == 1:
        raise ValueError(
            "The low-rank component of this solution has rank 1, so the latent "
            "subspace is a single direction. Both tabs of this visualizer plot "
            "PC1 against PC2 and a rank-1 block spans no such plane, so "
            "lowering --p-n-components will not make this run: rank 2 is the "
            "smallest rank that can be projected. Re-solve with a smaller mu1 "
            "if you need the projection."
        )
    if n_components > rank:
        raise ValueError(
            f"n_components ({n_components}) exceeds the rank of the low-rank "
            f"component ({rank}). Pass --p-n-components {rank} or lower."
        )

    # Orient the table against L rather than assuming a convention. The two
    # tables a user might reasonably pass here disagree:
    #   * a raw FeatureTable          -> to_dataframe() is features x samples
    #   * the output of transform-features -> samples x features
    # (transform_features returns a (p, N) frame, which QIIME 2's DataFrame
    # transformer then stores as if it were samples x features, swapping the
    # axes -- see the note in its docstring.)
    # PCA projects samples onto the latent subspace, so we need (N, p) to line
    # up with L's (p, p). Match on shape and fail loudly if neither axis fits.
    # (A square table -- as many samples as features -- is genuinely ambiguous;
    # it is taken as already being samples x features.)
    p = L.shape[0]
    if df.shape[1] != p:
        if df.shape[0] == p:
            df = df.T
        else:
            raise ValueError(
                f"The feature table {df.shape} does not line up with the "
                f"low-rank component ({p} x {p}). Neither axis has {p} "
                "entries, so this table is not the one the solution was fitted "
                "from. Pass the same table you used for calculate-covariance."
            )

    numeric_md_cols = sample_metadata.filter_columns(column_type="numeric")
    md = numeric_md_cols.to_dataframe()
    md = md.reindex(df.index)

    if color_by is None:
        depth = get_seq_depth(df)
        depth = pd.DataFrame(depth, index=depth.index, columns=["seq-depth"])
        md = md.join(depth)
        color_by = "seq-depth"

    p1 = project_covariates(counts=df, metadata=md, L=L, color_by=color_by)
    p2 = pair_plot(
        counts=df,
        metadata=md,
        L=L,
        n_components=n_components,
        color_by=color_by,
    )

    tab1 = TabPanel(child=p1, title="Single plot")
    tab2 = TabPanel(child=p2, title="Pair-plot")

    tabs = [tab1, tab2]
    p = Tabs(tabs=tabs)

    # components() takes (models, wrap_script, wrap_plot_info, theme) -- the old
    # `components(p, INLINE)` passed INLINE as wrap_script and never inlined
    # anything. The bokeh JS is injected into the template instead, at a version
    # that matches the installed library. INLINE rather than CDN keeps the .qzv
    # self-contained for offline viewing.
    script, div = components(p)
    output_from_parsed_template = template.render(
        bokeh_resources=INLINE.render(), plot_script=script, plot_div=div
    )

    with open(os.path.join(output_dir, "index.html"), "w") as fh:
        fh.write(output_from_parsed_template)
