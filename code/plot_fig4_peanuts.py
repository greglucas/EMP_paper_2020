import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj_data = ccrs.PlateCarree()
projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)

# stuff for the colormap
cmap = get_cmap('RdYlBu_r')
# cmap = get_cmap('magma')
norm = LogNorm(0.1, 100.)

plt.style.use(['seaborn-paper', './tex.mplstyle'])

# angles for ellipse plotting
a = np.linspace(0., 2. * np.pi, num=100)

# 'geom', 'area', 'max'
size_scaling = 'max'
color_scaling = 'max'

# taken from Greg's workbook
# https://github.com/greglucas/GeoelectricHazardPaper2019/blob/master/code/GeoelectricHazardPaper.ipynb
US_lon_bounds = (-125, -66)
US_lat_bounds = (24, 50)
lon_bounds = US_lon_bounds
plot_lon_bounds = (lon_bounds[0] + 5, lon_bounds[1] - 5)
lat_bounds = US_lat_bounds


def add_features_to_ax(ax):
    land_alpha = 0.7
    scale = '10m'
    # 10m oceans are super slow...
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor='slategrey', alpha=0.65, zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale(scale),
                   facecolor='k', alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.STATES.with_scale(scale),
                   edgecolor='w', linewidth=0.4, alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale(scale),
                   facecolor='slategrey', alpha=0.25, zorder=0)


def e_polarization(ax, cbar_ax, fig):

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-96., 51., 'Impedance E-Polarization States',
    #         fontsize=18, color='k', va='center', ha='center', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-76.25, 23., 'E-Polarization',
            fontsize=18, color='k', va='center', ha='center', zorder=3,
            transform=proj_data, fontweight='bold')

    for period, zorder in zip([10., 100., 1000.], [1, 2, 3]):

        npz = np.load('Impedance_{:.0f}s.npz'.format(period))
        impedance_tensor = npz['impedance_tensor']
        coords = npz['coordinates']

        # E polarization state from Berdichevsky & Dmitriev book
        l1 = (np.absolute(impedance_tensor[:, 0, 0]) ** 2. +
              np.absolute(impedance_tensor[:, 0, 1]) ** 2.) / \
             np.absolute(impedance_tensor[:, 0, 0] *
                         impedance_tensor[:, 1, 1] -
                         impedance_tensor[:, 0, 1] *
                         impedance_tensor[:, 1, 0]) ** 2.
        l2 = 2. * np.real(impedance_tensor[:, 0, 0] *
                          impedance_tensor[:, 1, 0].conj() +
                          impedance_tensor[:, 1, 1] *
                          impedance_tensor[:, 0, 1].conj()) / \
             np.absolute(impedance_tensor[:, 0, 0] *
                         impedance_tensor[:, 1, 1] -
                         impedance_tensor[:, 0, 1] *
                         impedance_tensor[:, 1, 0]) ** 2.
        l3 = (np.absolute(impedance_tensor[:, 1, 1]) ** 2. +
              np.absolute(impedance_tensor[:, 1, 0]) ** 2.) / \
             np.absolute(impedance_tensor[:, 0, 0] *
                         impedance_tensor[:, 1, 1] -
                         impedance_tensor[:, 0, 1] *
                         impedance_tensor[:, 1, 0]) ** 2.

        for i in range(coords.shape[0]):
            z_e = np.sqrt(1. / (l1[i] * np.sin(a) ** 2. -
                                l2[i] * np.sin(a) * np.cos(a) +
                                l3[i] * np.cos(a) ** 2.))
            # REMEMBER Y is E/W and X is N/S
            vertices = np.column_stack((z_e * np.sin(a), z_e * np.cos(a)))

            if size_scaling == 'geom':
                size_scale = np.sqrt(np.amax(z_e) * np.amin(z_e)) * 5.
            elif size_scaling == 'area':
                size_scale = np.pi * np.amax(z_e) * np.amin(z_e) / 5.
            elif size_scaling == 'max':
                size_scale = np.amax(z_e) * 5.
            else:
                size_scale = 1.

            if color_scaling == 'geom':
                color_scale = np.sqrt(np.amax(z_e) * np.amin(z_e))
            elif color_scaling == 'area':
                color_scale = np.pi * np.amax(z_e) * np.amin(z_e)
            elif color_scaling == 'max':
                color_scale = np.amax(z_e)
            else:
                color_scale = 1.

            ax.scatter(np.atleast_2d(coords[i, 1]), np.atleast_2d(coords[i, 0]),
                       s=size_scale, c=np.atleast_2d(cmap(norm(color_scale))),
                       marker=vertices, zorder=zorder, edgecolor='k',
                       linewidth=0.1, transform=proj_data)

    cax = ax.scatter([], [], s=1., c=[], cmap=cmap, norm=norm,
                     marker='o')
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal',
                        use_gridspec=True, fraction=1., aspect=35.)
    cbar.set_label('Max Scaling [mv/km]/[nT]', fontsize=12,
                   labelpad=4, rotation=0.)
    # cbar.set_label('Size: {}, Color: {}'.format(size_scaling, color_scaling),
    #                fontsize=10, labelpad=4, rotation=0.)
    cbar.ax.tick_params(labelsize=10)


def h_polarization(ax):

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-96., 51., 'Impedance H-Polarization States',
    #         fontsize=18, color='k', va='center', ha='center', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-76.25, 23., 'H-Polarization',
            fontsize=18, color='k', va='center', ha='center', zorder=3,
            transform=proj_data, fontweight='bold')

    for period, zorder in zip([10., 100., 1000.], [1, 2, 3]):

        npz = np.load('Impedance_{:.0f}s.npz'.format(period))
        impedance_tensor = npz['impedance_tensor']
        coords = npz['coordinates']

        # H polarization state from Berdichevsky & Dmitriev book
        k1 = np.absolute(impedance_tensor[:, 0, 1])**2. + \
             np.absolute(impedance_tensor[:, 1, 1])**2.
        k2 = 2. * np.real(impedance_tensor[:, 0, 0] * impedance_tensor[:, 0, 1].conj() +
                          impedance_tensor[:, 1, 0] * impedance_tensor[:, 1, 1].conj())
        k3 = np.absolute(impedance_tensor[:, 0, 0])**2. + \
             np.absolute(impedance_tensor[:, 1, 0])**2.

        for i in range(coords.shape[0]):
            z_h = np.sqrt(k1[i] * np.sin(a) ** 2. +
                          k2[i] * np.sin(a) * np.cos(a) +
                          k3[i] * np.cos(a) ** 2.)
            # REMEMBER Y is E/W and X is N/S
            vertices = np.column_stack((z_h * np.sin(a), z_h * np.cos(a)))

            if size_scaling == 'geom':
                size_scale = np.sqrt(np.amax(z_h) * np.amin(z_h)) * 5.
            elif size_scaling == 'area':
                size_scale = np.pi * np.amax(z_h) * np.amin(z_h) / 5.
            elif size_scaling == 'max':
                size_scale = np.amax(z_h) * 5.
            else:
                size_scale = 1.

            if color_scaling == 'geom':
                color_scale = np.sqrt(np.amax(z_h) * np.amin(z_h))
            elif color_scaling == 'area':
                color_scale = np.pi * np.amax(z_h) * np.amin(z_h)
            elif color_scaling == 'max':
                color_scale = np.amax(z_h)
            else:
                color_scale = 1.

            ax.scatter(np.atleast_2d(coords[i, 1]), np.atleast_2d(coords[i, 0]),
                       s=size_scale, c=np.atleast_2d(cmap(norm(color_scale))),
                       marker=vertices, zorder=zorder, edgecolor='k',
                       linewidth=0.1, transform=proj_data)


def main():

    fig = plt.figure(figsize=(8.5, 11))
    ax1 = fig.add_subplot(211, aspect='equal', projection=projection)
    ax2 = fig.add_subplot(212, aspect='equal', projection=projection)
    ax3 = inset_axes(ax1, width='50%', height='1%', loc='lower center',
                     bbox_to_anchor=(0., 0.5175, 1., 1.),
                     bbox_transform=fig.transFigure, borderpad=0)
                     # bbox_to_anchor=(0., 0.515, 1., 1.),
                     # bbox_transform=fig.transFigure, borderpad=0)
    # ax3 = fig.add_axes([0.5, 0.5, 0.2, 0.1], frameon=False)
    # ax3.axes.get_xaxis().set_visible(False)
    # ax3.axes.get_yaxis().set_visible(False)

    e_polarization(ax1, ax3, fig)
    h_polarization(ax2)

    plt.subplots_adjust(left=0.001, right=0.98, top=0.99, bottom=0.022,
                        hspace=0.15, wspace=0.1)
    plt.savefig('fig4_peanuts.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
