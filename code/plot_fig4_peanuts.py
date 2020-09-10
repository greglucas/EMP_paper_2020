import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.ticker as mticker
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bezpy
from scipy.interpolate import interp1d

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

proj_data = ccrs.PlateCarree()
lon_bounds = (-93.5, -87.5)
lat_bounds = (34.8, 39.5)
projection = \
    ccrs.Mercator(central_longitude=(lon_bounds[0] + lon_bounds[1]) / 2.,
                  latitude_true_scale=(lat_bounds[0] + lat_bounds[1]) / 2.)
# projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)

# stuff for the colormap
cmap = get_cmap('RdYlBu_r')
# cmap = get_cmap('magma')
norm = LogNorm(1., 1000.)

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

# periods = [1., 10., 100.]
periods = [1.]


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

    ax.set_extent(lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-93.15, 35.15, 'Impedance E-Polarization States',
    #         fontsize=12, color='k', va='bottom', ha='left', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-93.35, 35., 'E-Polarization',
            fontsize=10, color='w', va='bottom', ha='left', zorder=3,
            transform=proj_data)

    # E polarization state from Berdichevsky & Dmitriev book
    l1 = (np.absolute(impedance_tensors[:, :, 0, 0]) ** 2. +
          np.absolute(impedance_tensors[:, :, 0, 1]) ** 2.) / \
         np.absolute(impedance_tensors[:, :, 0, 0] *
                     impedance_tensors[:, :, 1, 1] -
                     impedance_tensors[:, :, 0, 1] *
                     impedance_tensors[:, :, 1, 0]) ** 2.
    l2 = 2. * np.real(impedance_tensors[:, :, 0, 0] *
                      impedance_tensors[:, :, 1, 0].conj() +
                      impedance_tensors[:, :, 1, 1] *
                      impedance_tensors[:, :, 0, 1].conj()) / \
         np.absolute(impedance_tensors[:, :, 0, 0] *
                     impedance_tensors[:, :, 1, 1] -
                     impedance_tensors[:, :, 0, 1] *
                     impedance_tensors[:, :, 1, 0]) ** 2.
    l3 = (np.absolute(impedance_tensors[:, :, 1, 1]) ** 2. +
          np.absolute(impedance_tensors[:, :, 1, 0]) ** 2.) / \
         np.absolute(impedance_tensors[:, :, 0, 0] *
                     impedance_tensors[:, :, 1, 1] -
                     impedance_tensors[:, :, 0, 1] *
                     impedance_tensors[:, :, 1, 0]) ** 2.
    # PERIOD ON AXIS 0, SITE ON AXIS 1

    for j in range(coords.shape[0]):

        for period, zorder in zip(periods, [1, 1, 1]):

            i = periods.index(period)

            z_e = np.sqrt(1. / (l1[i, j] * np.sin(a) ** 2. -
                                l2[i, j] * np.sin(a) * np.cos(a) +
                                l3[i, j] * np.cos(a) ** 2.))
            # REMEMBER Y is E/W and X is N/S
            vertices = np.column_stack((z_e * np.sin(a), z_e * np.cos(a)))

            if scale_size:
                size_scale = np.amax(z_e) * size_scaling
            else:
                size_scale = 200. / (2. * (np.log10(period) + 1))

            color_scale = np.amax(z_e)

            ax.scatter(np.atleast_2d(coords[j, 1]), np.atleast_2d(coords[j, 0]),
                       s=size_scale, c=np.atleast_2d(cmap(norm(color_scale))),
                       marker=vertices, zorder=zorder, edgecolor='k',
                       linewidth=0.1, transform=proj_data)

    cax = ax.scatter([], [], s=1., c=[], cmap=cmap, norm=norm,
                     marker='o')
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation='horizontal',
                        use_gridspec=True, fraction=1., aspect=35.)
    cbar.set_label('Max Scaling [mv/km]/[nT]', fontsize=8,
                   labelpad=2, rotation=0.)
    # cbar.set_label('Size: {}, Color: {}'.format(size_scaling, color_scaling),
    #                fontsize=10, labelpad=4, rotation=0.)
    cbar.ax.tick_params(labelsize=8)


def h_polarization(coords, impedance_tensors, ax):

    ax.set_extent(lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)

    # ax.text(-93.15, 35.15, 'Impedance B-Polarization States',
    #         fontsize=12, color='k', va='bottom', ha='left', zorder=3,
    #         path_effects=[pe.withStroke(linewidth=2, foreground='w')],
    #         transform=proj_data)
    ax.text(-93.35, 35., 'B-Polarization',
            fontsize=10, color='w', va='bottom', ha='left', zorder=3,
            transform=proj_data)

    # H polarization state from Berdichevsky & Dmitriev book
    k1 = np.absolute(impedance_tensors[:, :, 0, 1]) ** 2. + \
         np.absolute(impedance_tensors[:, :, 1, 1]) ** 2.
    k2 = 2. * np.real(
        impedance_tensors[:, :, 0, 0] * impedance_tensors[:, :, 0, 1].conj() +
        impedance_tensors[:, :, 1, 0] * impedance_tensors[:, :, 1, 1].conj())
    k3 = np.absolute(impedance_tensors[:, :, 0, 0]) ** 2. + \
         np.absolute(impedance_tensors[:, :, 1, 0]) ** 2.
    # PERIOD ON AXIS 0, SITE ON AXIS 1

    for j in range(coords.shape[0]):

        for period, zorder in zip(periods, [1, 1, 1]):

            i = periods.index(period)

            z_h = np.sqrt(k1[i, j] * np.sin(a) ** 2. +
                          k2[i, j] * np.sin(a) * np.cos(a) +
                          k3[i, j] * np.cos(a) ** 2.)
            # REMEMBER Y is E/W and X is N/S
            vertices = np.column_stack((z_h * np.sin(a), z_h * np.cos(a)))

            if scale_size:
                size_scale = np.amax(z_h) * size_scaling
            else:
                size_scale = 200. / (2. * (np.log10(period) + 1))

            color_scale = np.amax(z_h)

            ax.scatter(np.atleast_2d(coords[j, 1]), np.atleast_2d(coords[j, 0]),
                       s=size_scale, c=np.atleast_2d(cmap(norm(color_scale))),
                       marker=vertices, zorder=zorder, edgecolor='k',
                       linewidth=0.1, transform=proj_data)


def main():

    coords = np.array(MT_xys)
    impedance_tensors = create_impedance_tensor_array()

    fig = plt.figure(figsize=(10./2.54, 20./2.54))
    gs = GridSpec(3, 1, height_ratios=[1., 0.1, 1.])
    ax_a = fig.add_subplot(gs[0, 0], aspect='equal', projection=projection)
    ax_b = fig.add_subplot(gs[2, 0], aspect='equal', projection=projection)
    ax_cbar = inset_axes(ax_a, width='50%', height='1%', loc='lower center',
                         # bbox_to_anchor=(0., 0.3575, 1., 1.),
                         bbox_to_anchor=(0., 0.5225, 1., 1.),
                         bbox_transform=fig.transFigure, borderpad=0)

    # NEED TO COORDINATE LABEL PLACEMENT WITH OTHER MAPS
    ax_a.text(0.03, 0.97, r"$\bf{(a)}$", fontsize=10, color='w', va='top',
              ha='left', zorder=3, transform=ax_a.transAxes)
    ax_b.text(0.03, 0.97, r"$\bf{(b)}$", fontsize=10, color='w', va='top',
              ha='left', zorder=3, transform=ax_b.transAxes)

    e_polarization(coords, impedance_tensors, ax_a, ax_cbar, fig)
    h_polarization(coords, impedance_tensors, ax_b)

    plt.subplots_adjust(left=0.001, right=0.98, top=0.99, bottom=0.022,
                        hspace=0.05, wspace=0.1)
    plt.savefig('../figs/fig4_peanuts.png', dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
