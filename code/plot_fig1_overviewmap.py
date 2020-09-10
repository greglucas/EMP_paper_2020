import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bezpy

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

# taken from Greg's workbook
# https://github.com/greglucas/GeoelectricHazardPaper2019/blob/master/code/GeoelectricHazardPaper.ipynb
lon_bounds = (-93.5, -87.5)
lat_bounds = (34.8, 39.5)
inset_lon_bounds = (-180., -40.)
inset_lat_bounds = (0., 73.)

proj_data = ccrs.PlateCarree()
# @GREG: CAN YOU CHECK TO MAKE SURE WE'RE USING ALL THE SAME PROJECTION PARAMS
# projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)
projection = \
    ccrs.Mercator(central_longitude=(lon_bounds[0] + lon_bounds[1]) / 2.,
                  latitude_true_scale=(lat_bounds[0] + lat_bounds[1]) / 2.)
inset_projection = ccrs.Mercator()

plt.style.use(['seaborn-paper', './tex.mplstyle'])

# angles for ellipse plotting
a = np.linspace(0., 2. * np.pi, num=100)

scale_size = True
size_scaling = 1.

mt_data_folder = '../data/'
list_of_files = sorted(glob.glob(mt_data_folder + '*.xml'))
MT_sites = \
    {site.name: site for site in [bezpy.mt.read_xml(f) for f in list_of_files]}
list_of_sites = sorted(MT_sites.keys())
MT_xys = [[site.latitude, site.longitude] for site in MT_sites.values()]
site_1d = 'SFM06'
site_1d_coords = [MT_sites[site_1d].latitude, MT_sites[site_1d].longitude]
site_3d = 'RF111'
site_3d_coords = [MT_sites[site_3d].latitude, MT_sites[site_3d].longitude]


def add_features_to_ax(ax):
    land_alpha = 0.7
    scale = '10m'
    # 10m oceans are super slow...
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor='slategrey', alpha=0.65, zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale(scale),
                   facecolor='k', alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.STATES.with_scale(scale),
                   edgecolor='w', linewidth=0.4, alpha=land_alpha, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale(scale),
                   facecolor='slategrey', alpha=0.25, zorder=0)


def overview(ax, coords):

    ax.set_extent(lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    shape_feature = \
        ShapelyFeature(Reader('../data/mississippi_embayment.shp').geometries(),
                       ccrs.PlateCarree(), fc='tab:orange', ec='none',
                       alpha=0.47, zorder=0)
    ax.add_feature(shape_feature)

    ax.text(-93.4, 36.3, 'AR', fontsize=8, color='w', va='bottom', ha='left',
            zorder=3, transform=proj_data)
    ax.text(-93.4, 36.6, 'MO', fontsize=8, color='w', va='bottom', ha='left',
            zorder=3, transform=proj_data)
    ax.text(-87.9, 39.35, 'IL', fontsize=8, color='w', va='bottom', ha='left',
            zorder=3, transform=proj_data)
    ax.text(-87.8, 36.75, 'KY', fontsize=8, color='w', va='bottom', ha='left',
            zorder=3, transform=proj_data)
    ax.text(-87.8, 36.425, 'TN', fontsize=8, color='w', va='bottom', ha='left',
            zorder=3, transform=proj_data)

    ax.scatter(coords[:, 1], coords[:, 0], s=20., marker='v',
               facecolor='w', edgecolor='k', linewidth=0.8, zorder=2,
               transform=proj_data)
    ax.annotate(site_1d, xy=(site_1d_coords[1], site_1d_coords[0]),
                xytext=(site_1d_coords[1] + 0.2, site_1d_coords[0] + 0.2),
                arrowprops=dict(fc='w', ec='w', headlength=3,
                                headwidth=3, width=0.8, shrink=0.2),
                fontsize=7, color='w', va='top', ha='left',
                bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                xycoords=proj_data._as_mpl_transform(ax), zorder=3)
    ax.annotate(site_3d, xy=(site_3d_coords[1], site_3d_coords[0]),
                xytext=(site_3d_coords[1] - 0.08, site_3d_coords[0] - 0.15),
                arrowprops=dict(fc='w', ec='w', headlength=3,
                                headwidth=3, width=0.8, shrink=0.2),
                fontsize=7, color='w', va='top', ha='right',
                bbox=dict(pad=0, facecolor="none", edgecolor="none"),
                xycoords=proj_data._as_mpl_transform(ax), zorder=3)

    ax.scatter([-90.0490, -90.1994], [35.1495, 38.6270],
               s=200., marker='*', fc='tab:blue', edgecolor='k', linewidth=0.8,
               transform=proj_data, zorder=2)
    ax.annotate('Memphis', (-90.0490, 35.1495),
                xycoords=proj_data._as_mpl_transform(ax),
                textcoords='offset points', xytext=(5, 3),
                fontsize=10, color='w', va='bottom', ha='left', zorder=3)
    ax.annotate('St. Louis', (-90.1994, 38.6270),
                xycoords=proj_data._as_mpl_transform(ax),
                textcoords='offset points', xytext=(5, 3),
                fontsize=10, color='w', va='bottom', ha='left', zorder=3)

    ax.text(-91.6, 37.45, r'$\textbf{Ozark Dome}$', fontsize=8, color='w',
            weight='bold', style='oblique',
            va='center', ha='center', zorder=3, transform=proj_data)
    ax.text(-90.25, 35.7, r'$\textbf{Mississippi}$', fontsize=8, color='w',
            weight='bold', style='oblique', va='center', ha='center', zorder=3,
            rotation=65., transform=proj_data)
    ax.text(-89.6, 36.65, r'$\textbf{Embayment}$', fontsize=8, color='w',
            weight='bold', style='oblique', va='center', ha='center', zorder=3,
            rotation=65., transform=proj_data)
    ax.text(-89.15, 37.9, r'$\textbf{Illinois Basin}$', fontsize=8, color='w',
            weight='bold', style='oblique', va='center', ha='center', zorder=3,
            transform=proj_data)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=2, color='gray', alpha=0.5, linestyle='--')
    gl.xlabels_top = False
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xlocator = mticker.FixedLocator([-94., -92., -90., -88.])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.ylocator = mticker.FixedLocator([34., 36., 38., 40.])
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 8, 'color': 'black'}
    gl.ylabel_style = {'size': 8, 'color': 'black'}


def inset(ax):

    ax.set_extent(inset_lon_bounds + inset_lat_bounds, proj_data)
    land_alpha = 0.7
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor='slategrey', alpha=0.65, zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale('50m'),
                   facecolor='k', alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   edgecolor='w', linewidth=0.4, alpha=land_alpha, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'),
                   facecolor='slategrey', alpha=0.47, zorder=0)

    rect = Rectangle((lon_bounds[0], lat_bounds[0]),
                     lon_bounds[1] - lon_bounds[0],
                     lat_bounds[1] - lat_bounds[0],
                     linewidth=1.5, edgecolor='w',
                     facecolor='none', transform=proj_data)
    ax.add_patch(rect)


def main():

    coords = np.array(MT_xys)

    # @GREG, @JOSH -- WE SHOULD PROBABLY CHECK AND STANDARDIZE FIGURE SIZES
    #    ACROSS ALL OUR PLOTTING SCRIPTS
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, aspect='equal', projection=projection)
    overview(ax, coords)

    # Could probably do something more sophisticated with inset location here...
    # pos = ax.get_position()
    # x0 = pos.x0
    # y0 = pos.y0
    ax_inset = fig.add_axes([0.09, 0.05, 0.29, 0.29], aspect='equal',
                            projection=inset_projection)
    inset(ax_inset)

    plt.subplots_adjust(left=0.075, right=0.975, top=0.99, bottom=0.022,
                        hspace=0.05, wspace=0.1)
    plt.savefig('../figs/fig1_overview.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
