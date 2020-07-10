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


def calc_B(lats, lons, ts):
    nsites = len(lats)
    ntimes = len(ts)
    # Generate time series of B at origin that is then
    # scaled based on the spatial location defined later
    B_E3A = (np.ones((ntimes, nsites, 3)) *
             Bt_E3A(ts)[:, np.newaxis, np.newaxis])
    B_E3B = (np.ones((ntimes, nsites, 3)) *
             Bt_E3B(ts)[:, np.newaxis, np.newaxis])

    # generate B-field maps at each of the tensor locations
    Bx_E3A, By_E3A, Bz_E3A = Br_E3A(lats, lons)
    Bx_E3B, By_E3B, Bz_E3B = Br_E3B(lats, lons)

    # Spatial multiplication by the time series
    B_E3A[:, :, 0] *= Bx_E3A[np.newaxis, :]
    B_E3A[:, :, 1] *= By_E3A[np.newaxis, :]
    B_E3A[:, :, 2] *= Bz_E3A[np.newaxis, :]

    B_E3B[:, :, 0] *= Bx_E3B[np.newaxis, :]
    B_E3B[:, :, 1] *= By_E3B[np.newaxis, :]
    B_E3B[:, :, 2] *= Bz_E3B[np.newaxis, :]

    return B_E3A, B_E3B


def calc_E_sites(sites, B_sites, fs):
    E_sites = np.zeros(B_sites.shape)
    for i, site in enumerate(sites.values()):
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


def animate_fields(E_sites, ts, fs):
    from matplotlib import animation
    # Set up the figure and axes
    fig = plt.figure(figsize=(10, 5))
    height_ratios = [4, 10, 1]
    gs = fig.add_gridspec(ncols=3, nrows=3, height_ratios=height_ratios,
                          hspace=0.5)
    ax_time = fig.add_subplot(gs[0, :])
    ax_bfield = fig.add_subplot(gs[1, 0], projection=projection)
    ax_bfield_cbar = fig.add_subplot(gs[2, 0])
    ax_efield = fig.add_subplot(gs[1, 1], projection=projection)
    ax_efield_cbar = fig.add_subplot(gs[2, 1])
    ax_voltage = fig.add_subplot(gs[1, 2], projection=projection)
    ax_voltage_cbar = fig.add_subplot(gs[2, 2])

    # Time series
    # -----------
    ax = ax_time
    B_E3A = Bt_E3A(ts)
    B_E3B = Bt_E3B(ts)
    ax_time.plot(ts, B_E3A)
    ax_time.plot(ts, B_E3B)
    ax_time.plot(ts, B_E3A + B_E3B)
    time_line = ax_time.axvline(0, c='k')
    # zero line
    ax_time.axhline(0, c='gray', zorder=-5)
    ax_time.set_xlim(1e-1, 1e3)
    ax_time.set_xscale('log')
    ax_time.set_ylim(-2500, 2500)
    ax_time.xaxis.tick_top()
    ax_time.tick_params(which='both', direction='in', pad=0)

    # create grid for map
    enhance = 20
    nlon = int(np.diff(lon_bounds)) + 1
    nlat = int(np.diff(lat_bounds)) + 1
    grid_lons = np.linspace(lon_bounds[0]-2, lon_bounds[1]+2,
                            nlon*enhance)
    grid_lats = np.linspace(lat_bounds[0]-2, lat_bounds[1]+2,
                            nlat*enhance)
    # Specify the grid points and edges separately for plotting
    lon_mesh, lat_mesh = np.meshgrid(grid_lons, grid_lats)
    lon_edges, lat_edges = _mesh_grid(grid_lons, grid_lats)

    # generate gridded B-field maps
    B_E3A, B_E3B = calc_B(lat_mesh.ravel(), lon_mesh.ravel(), ts)
    B = B_E3A + B_E3B
    Bx = B[:, :, 0]
    By = B[:, :, 1]
    Bh = np.sqrt(Bx**2 + By**2)

    norm = LogNorm(1, 2500)
    ax = ax_bfield
    ax_cbar = ax_bfield_cbar
    pcol = ax.pcolormesh(lon_edges, lat_edges,
                         Bh[0, :].reshape(lat_mesh.shape),
                         transform=proj_data,
                         norm=norm,
                         alpha=.5,
                         linewidth=0)
    pcol.set_edgecolor('face')
    cb = plt.colorbar(pcol, cax=ax_cbar, orientation='horizontal')
    cb.set_label(label='$B_h$ (nT)', fontsize=12)
    cb.ax.tick_params(labelsize=12)

    # Separately calculate the quiver points to make it a smaller grid
    grid_lons = np.linspace(lon_bounds[0]-2, lon_bounds[1]+2,
                            nlon*2)
    grid_lats = np.linspace(lat_bounds[0]-2, lat_bounds[1]+2,
                            nlat*2)
    lonq, latq = _mesh_grid(grid_lons, grid_lats)
    B_E3A, B_E3B = calc_B(latq.ravel(), lonq.ravel(), ts)
    Bq = B_E3A + B_E3B
    Bqx = Bq[:, :, 0]
    Bqy = Bq[:, :, 1]

    quiv_B = ax.quiver(lonq.ravel(), latq.ravel(),
                       Bqy[0, :], Bqx[0, :],
                       transform=proj_data,
                       color='w',
                       units='inches',
                       scale=2500)

    # Red x marks the spot
    ax.scatter(lon_epi, lat_epi, color='r', marker='x',
               s=50, transform=proj_data)

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)
    ax.set_title('E3 B-field', fontsize=12)

    # E-field
    # -------
    pred_lons = MT_xys[:, 1]
    pred_lats = MT_xys[:, 0]

    Ex = E_sites[:, :, 0]
    Ey = E_sites[:, :, 1]

    # Force all E's to 0 at t=0 to avoid plotting the Gibb's ringing
    Ex[0, :] = 0
    Ey[0, :] = 0
    Eh = np.sqrt(Ex**2 + Ey**2)

    ax = ax_efield
    ax_cbar = ax_efield_cbar
    # Red x marks the spot
    ax.scatter(lon_epi, lat_epi, color='r', marker='x',
               s=50, transform=proj_data)
    # Arrows for the data
    quiv_E = ax.quiver(pred_lons, pred_lats,
                       Ey[0, :], Ex[0, :],
                       transform=proj_data,
                       color='w',
                       units='inches',
                       scale=50000)

    ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    add_features_to_ax(ax)
    ax.set_title('E3 E-field', fontsize=12)

    title = fig.suptitle('Time: 0 s')

    def animate(t):
        # t *= 10
        time_line.set_xdata(ts[t])
        pcol.set_array(Bh[t, :])
        quiv_B.set_UVC(Bqy[t, :], Bqx[t, :])
        quiv_E.set_UVC(Ey[t, :], Ex[t, :])
        title.set_text(f'Time: {ts[t]:.2f} s')

    anim = animation.FuncAnimation(fig, animate,
                                   frames=[x for x in range(25)],
                                   interval=10)
    anim.save('../figs/test_animation.mp4')


def _mesh_grid(x, y):
    """A helper function to extrapolate/center the meshgrid coordiantes.

    Matplotlib's pcolormesh currently needs data specified at edges
    and drops the last column of the data, unfortunately. This function
    borrows from matplotlib PR #16258, which will automatically extend
    the grids in the future (Likely MPL 3.3+).
    """
    def _interp_grid(X):
        # helper for below
        if np.shape(X)[1] > 1:
            dX = np.diff(X, axis=1)/2.
            X = np.hstack((X[:, [0]] - dX[:, [0]],
                           X[:, :-1] + dX,
                           X[:, [-1]] + dX[:, [-1]]))
        else:
            # This is just degenerate, but we can't reliably guess
            # a dX if there is just one value.
            X = np.hstack((X, X))
        return X

    X, Y = np.meshgrid(x, y)
    # extend rows
    X = _interp_grid(X)
    Y = _interp_grid(Y)
    # extend columns
    X = _interp_grid(X.T).T
    Y = _interp_grid(Y.T).T
    return X, Y


def main():
    fs = 10  # sampling frequency
    ntimes = 1500000
    ntimes = 15000
    ts = np.arange(ntimes) / fs  # time steps

    B_sites_E3A, B_sites_E3B = calc_B(MT_xys[:, 0], MT_xys[:, 1], ts)
    # Add together for total B field
    B_sites = B_sites_E3A + B_sites_E3B
    E_sites = calc_E_sites(MT_sites, B_sites, fs)

    animate_fields(E_sites, ts, fs)
    return

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