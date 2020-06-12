import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bezpy
from scipy.interpolate import interp1d

import cartopy.crs as ccrs
import cartopy.feature as cfeature

proj_data = ccrs.PlateCarree()
projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)

# stuff for the colormap
cmap = get_cmap('RdYlBu_r')
# cmap = get_cmap('magma')
norm = LogNorm(1., 1000.)

plt.style.use(['seaborn-paper', './tex.mplstyle'])

# angles for ellipse plotting
a = np.linspace(0., 2. * np.pi, num=100)

# 'geom', 'area', 'max'
size_scaling = 'max'
color_scaling = 'max'

# taken from Greg's workbook
# https://github.com/greglucas/GeoelectricHazardPaper2019/blob/master/code/GeoelectricHazardPaper.ipynb
lon_bounds = (-93.5, -87.5)
plot_lon_bounds = lon_bounds
lat_bounds = (35., 39.25)

mt_data_folder = '../data/'
list_of_files = sorted(glob.glob(mt_data_folder + '*.xml'))
MT_sites = \
    {site.name: site for site in [bezpy.mt.read_xml(f) for f in list_of_files]}
list_of_sites = sorted(MT_sites.keys())
MT_xys = [[site.latitude, site.longitude] for site in MT_sites.values()]

periods = [1., 10., 100.]


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


def create_impedance_tensor_array():

    impedance_tensors = np.zeros((len(periods), len(MT_sites.keys()), 2, 2),
                                 dtype=np.complex128)
    for j in range(len(list_of_sites)):
        site = MT_sites[list_of_sites[j]]
        z = site.Z.reshape((2, 2, -1))
        zxx_re_interpolator = \
            interp1d(site.periods[~np.isnan(np.real(z[0, 0, :]))],
                     np.real(z[0, 0, :])[~np.isnan(np.real(z[0, 0, :]))],
                     fill_value='extrapolate')
        zxx_im_interpolator = \
            interp1d(site.periods[~np.isnan(np.imag(z[0, 0, :]))],
                     np.imag(z[0, 0, :])[~np.isnan(np.imag(z[0, 0, :]))],
                     fill_value='extrapolate')
        zxy_re_interpolator = \
            interp1d(site.periods[~np.isnan(np.real(z[0, 1, :]))],
                     np.real(z[0, 1, :])[~np.isnan(np.real(z[0, 1, :]))],
                     fill_value='extrapolate')
        zxy_im_interpolator = \
            interp1d(site.periods[~np.isnan(np.imag(z[0, 1, :]))],
                     np.imag(z[0, 1, :])[~np.isnan(np.imag(z[0, 1, :]))],
                     fill_value='extrapolate')
        zyx_re_interpolator = \
            interp1d(site.periods[~np.isnan(np.real(z[1, 0, :]))],
                     np.real(z[1, 0, :])[~np.isnan(np.real(z[1, 0, :]))],
                     fill_value='extrapolate')
        zyx_im_interpolator = \
            interp1d(site.periods[~np.isnan(np.imag(z[1, 0, :]))],
                     np.imag(z[1, 0, :])[~np.isnan(np.imag(z[1, 0, :]))],
                     fill_value='extrapolate')
        zyy_re_interpolator = \
            interp1d(site.periods[~np.isnan(np.real(z[1, 1, :]))],
                     np.real(z[1, 1, :])[~np.isnan(np.real(z[1, 1, :]))],
                     fill_value='extrapolate')
        zyy_im_interpolator = \
            interp1d(site.periods[~np.isnan(np.imag(z[1, 1, :]))],
                     np.imag(z[1, 1, :])[~np.isnan(np.imag(z[1, 1, :]))],
                     fill_value='extrapolate')
        for i in range(len(periods)):
            period = periods[i]
            impedance_tensors[i, j, 0, 0] = \
                zxx_re_interpolator(period) + 1.j * zxx_im_interpolator(period)
            impedance_tensors[i, j, 0, 1] = \
                zxy_re_interpolator(period) + 1.j * zxy_im_interpolator(period)
            impedance_tensors[i, j, 1, 0] = \
                zyx_re_interpolator(period) + 1.j * zyx_im_interpolator(period)
            impedance_tensors[i, j, 1, 1] = \
                zyy_re_interpolator(period) + 1.j * zyy_im_interpolator(period)

    return impedance_tensors


def e_polarization(coords, impedance_tensors, ax, cbar_ax, fig):

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-93.15, 35.15, 'Impedance E-Polarization States',
    #         fontsize=12, color='k', va='bottom', ha='left', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-93.4, 35.1, 'Impedance E-Polarization States',
            fontsize=12, color='w', va='bottom', ha='left', zorder=3,
            transform=proj_data)
    # ax.text(-76.25, 23., 'E-Polarization',
    #         fontsize=18, color='k', va='center', ha='center', zorder=3,
    #         transform=proj_data, fontweight='bold')

    for period, zorder in zip(periods, [1, 2, 3]):

        i = periods.index(period)
        impedance_tensor = impedance_tensors[i, :, :, :]

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


def h_polarization(coords, impedance_tensors, ax):

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-93.15, 35.15, 'Impedance B-Polarization States',
    #         fontsize=12, color='k', va='bottom', ha='left', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-93.4, 35.1, 'Impedance B-Polarization States',
            fontsize=12, color='w', va='bottom', ha='left', zorder=3,
            transform=proj_data)
    # ax.text(-76.25, 23., 'B-Polarization',
    #         fontsize=18, color='k', va='center', ha='center', zorder=3,
    #         transform=proj_data, fontweight='bold')

    for period, zorder in zip([1., 10., 100.], [1, 2, 3]):

        i = periods.index(period)
        impedance_tensor = impedance_tensors[i, :, :, :]

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

    coords = np.array(MT_xys)
    impedance_tensors = create_impedance_tensor_array()

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

    e_polarization(coords, impedance_tensors, ax1, ax3, fig)
    h_polarization(coords, impedance_tensors, ax2)

    plt.subplots_adjust(left=0.001, right=0.98, top=0.99, bottom=0.022,
                        hspace=0.15, wspace=0.1)
    plt.savefig('../figs/fig4_peanuts.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
