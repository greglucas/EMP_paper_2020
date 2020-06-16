import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bezpy

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_figXX_Bfields import Br_E3A, Br_E3B, Bt_E3A, Bt_E3B, lat_epi, lon_epi

proj_data = ccrs.PlateCarree()
projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)

# stuff for the colormap
cmap = get_cmap('RdYlBu_r')
# cmap = get_cmap('magma')
norm = LogNorm(1., 1000.)

plt.style.use(['seaborn-paper', './tex.mplstyle'])

# 'geom', 'area', 'max'
size_scaling = 'max'
color_scaling = 'max'


def symlog(x, y):
    # Arcsinh performs logarithm and preserves sign
    angle = np.arctan2(y, x)
    mag = np.sqrt(x**2 + y**2)
    newmag = np.arcsinh(mag/2)
    newx = newmag*np.cos(angle)
    newy = newmag*np.sin(angle)
    return newx, newy


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
MT_xys = np.array([[site.latitude, site.longitude] for site in MT_sites.values()])


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


def plot_E3_Bfield_map_sites(ax1, ax2):
    """
    Plot maps of E3A and E3B normalized B-field at sites
    """
    # create grid for map
    pred_lons = MT_xys[:, 1]
    pred_lats = MT_xys[:, 0]

    # generate B-field maps at each of the tensor locations
    Bx_E3A, By_E3A, Bz_E3A = Br_E3A(pred_lats, pred_lons)
    Bx_E3B, By_E3B, Bz_E3B = Br_E3B(pred_lats, pred_lons)

    # calculate horizontal B-field intensities (H)
    Bh_E3A = np.sqrt(Bx_E3A**2 + By_E3A**2)
    Bh_E3B = np.sqrt(Bx_E3B**2 + By_E3B**2)

    # Red x marks the spot
    ax1.scatter(lon_epi, lat_epi, color='r', marker='x',
                s=50, transform=proj_data)
    # Arrows for the data
    ax1.quiver(pred_lons, pred_lats,
               By_E3A, Bx_E3A,
               transform=proj_data,
               pivot='middle',
               color='w')

    ax1.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax1)
    ax1.set_title('E3A B-field', fontsize=12)

    # Red x marks the spot
    ax2.scatter(lon_epi, lat_epi, color='r', marker='x',
                s=50, transform=proj_data)
    # Arrows for the data
    ax2.quiver(pred_lons, pred_lats,
               By_E3B, Bx_E3B,
               transform=proj_data,
               pivot='middle',
               color='w')

    ax2.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax2)
    ax2.set_title('E3B B-field', fontsize=12)


def calc_B_sites(sites, ts):
    nsites = len(sites)
    fs = 10  # sampling frequency
    ntimes = len(ts)
    ts = np.arange(ntimes) / fs  # time steps
    # Generate time series of B at origin that is then
    # scaled based on the spatial location defined later
    B_sites_E3A = np.ones((ntimes, nsites, 3)) * Bt_E3A(ts)[:, np.newaxis, np.newaxis]
    B_sites_E3B = np.ones((ntimes, nsites, 3)) * Bt_E3B(ts)[:, np.newaxis, np.newaxis]

    pred_lons = MT_xys[:, 1]
    pred_lats = MT_xys[:, 0]

    # generate B-field maps at each of the tensor locations
    Bx_E3A, By_E3A, Bz_E3A = Br_E3A(pred_lats, pred_lons)
    Bx_E3B, By_E3B, Bz_E3B = Br_E3B(pred_lats, pred_lons)

    B_sites_E3A[:, :, 0] *= Bx_E3A[np.newaxis, :]
    B_sites_E3A[:, :, 1] *= By_E3A[np.newaxis, :]
    B_sites_E3A[:, :, 2] *= Bz_E3A[np.newaxis, :]

    B_sites_E3B[:, :, 0] *= Bx_E3B[np.newaxis, :]
    B_sites_E3B[:, :, 1] *= By_E3B[np.newaxis, :]
    B_sites_E3B[:, :, 2] *= Bz_E3B[np.newaxis, :]

    return B_sites_E3A, B_sites_E3B


def calc_E_sites(sites, B_sites, fs):
    E_sites = np.zeros(B_sites.shape)
    for i, site in enumerate(MT_sites.values()):
        Ex, Ey = site.convolve_fft(B_sites[:, i, 0], B_sites[:, i, 1], dt=1/fs)
        E_sites[:, i, 0] = Ex
        E_sites[:, i, 1] = Ey

    return E_sites


def plot_E3_Efield_map_sites(ax1, ax2, B_sites, E_sites):
    """
    Plot maps of E3A and E3B normalized E-field at sites
    """
    # create x/y coords for the map
    pred_lons = MT_xys[:, 1]
    pred_lats = MT_xys[:, 0]

    # Fill nan's with zeros to ignore runtime warnings
    B_sites[np.isnan(B_sites)] = 0.
    E_sites[np.isnan(E_sites)] = 0.

    Bx = B_sites[:, :, 0]
    By = B_sites[:, :, 1]
    Ex = E_sites[:, :, 0]
    Ey = E_sites[:, :, 1]

    # Now find the maximum E-field for each site and the time it occurs
    Eh = np.sqrt(Ex**2 + Ey**2)
    max_ts = np.expand_dims(np.argmax(Eh, axis=0), axis=0)
    Ex_max = np.take_along_axis(Ex, max_ts, axis=0).squeeze()
    Ey_max = np.take_along_axis(Ey, max_ts, axis=0).squeeze()

    Bx_max = np.take_along_axis(Bx, max_ts, axis=0).squeeze()
    By_max = np.take_along_axis(By, max_ts, axis=0).squeeze()

    ax1.quiver(pred_lons, pred_lats,
               By_max, Bx_max,
               transform=proj_data,
               color='w')

    ax1.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax1)
    ax1.set_title('E3 B-field', fontsize=12)

    ax2.quiver(pred_lons, pred_lats,
               Ey_max, Ex_max,
               transform=proj_data,
               color='w')

    ax2.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax2)
    ax2.set_title('E3 E-field', fontsize=12)


def animate_fields(B_sites, E_sites, fs):
    from matplotlib import animation
    # Set up the figure and axes
    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0], projection=projection)
    ax2 = fig.add_subplot(gs[1], projection=projection)
    plt.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.05)

    # Set up the initial quiver plots
    pred_lons = MT_xys[:, 1]
    pred_lats = MT_xys[:, 0]

    Bx = B_sites[:, :, 0]
    By = B_sites[:, :, 1]
    Ex = E_sites[:, :, 0]
    Ey = E_sites[:, :, 1]

    # Log-scale vectors immediately?
    # Bx, By = symlog(Bx, By)
    # Ex, Ey = symlog(Ex, Ey)

    # Red x marks the spot
    ax1.scatter(lon_epi, lat_epi, color='r', marker='x',
                s=50, transform=proj_data)
    # Arrows for the data
    quiv_B = ax1.quiver(pred_lons, pred_lats,
                        By[1, :], Bx[1, :],
                        transform=proj_data,
                        color='w',
                        units='inches',
                        scale=2500)

    ax1.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax1)
    ax1.set_title('E3 B-field', fontsize=12)

    # Red x marks the spot
    ax2.scatter(lon_epi, lat_epi, color='r', marker='x',
                s=50, transform=proj_data)
    # Arrows for the data
    quiv_E = ax2.quiver(pred_lons, pred_lats,
                        Ey[0, :], Ex[0, :],
                        transform=proj_data,
                        color='w',
                        units='inches',
                        scale=10000)

    ax2.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax2)
    ax2.set_title('E3 E-field', fontsize=12)

    title = fig.suptitle('Time: 0 s')

    def animate(t):
        quiv_B.set_UVC(By[t, :], Bx[t, :])
        quiv_E.set_UVC(Ey[t, :], Ex[t, :])
        title.set_text(f'Time: {t/fs:.2f} s')

    anim = animation.FuncAnimation(fig, animate, frames=range(1, 2500), interval=10)
    anim.save('../figs/test_animation.mp4')


def main():
    fs = 100  # sampling frequency
    ntimes = 1500000
    ntimes = 150000
    ts = np.arange(ntimes) / fs  # time steps

    B_sites_E3A, B_sites_E3B = calc_B_sites(MT_sites, ts)
    # Add together for total B field
    B_sites = B_sites_E3A + B_sites_E3B
    E_sites = calc_E_sites(MT_sites, B_sites, fs)

    animate_fields(B_sites, E_sites, fs)

    fig1 = plt.figure(figsize=(6.5, 6.5))
    gs1 = fig1.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 1])
    ax1 = fig1.add_subplot(gs1[0], projection=projection)
    ax2 = fig1.add_subplot(gs1[1], projection=projection)

    plot_E3_Bfield_map_sites(ax1, ax2)

    plt.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.05)
    plt.savefig('../figs/figXX_Bfields_sites.png')

    fig2 = plt.figure(figsize=(6.5, 6.5))
    gs2 = fig2.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 1])
    ax3 = fig2.add_subplot(gs2[0], projection=projection)
    ax4 = fig2.add_subplot(gs2[1], projection=projection)
    plot_E3_Efield_map_sites(ax3, ax4, B_sites, E_sites)

    plt.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.05)
    plt.savefig('../figs/figXX_Efields_sites.png')

    plt.show()

    plt.close(fig1)
    plt.close(fig2)

if __name__ == "__main__":
    main()
