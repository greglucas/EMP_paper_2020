import glob
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import bezpy
import geopandas as gpd
import shapely

import cartopy.crs as ccrs
import cartopy.feature as cfeature

from plot_figXX_Bfields import Br_E3A, Br_E3B, Bt_E3A, Bt_E3B, lat_epi, lon_epi

proj_data = ccrs.PlateCarree()
projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)
projection = ccrs.PlateCarree()

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

# MT Impedances
mt_data_folder = '../data/'
list_of_files = sorted(glob.glob(mt_data_folder + '*.xml'))
MT_sites = \
    {site.name: site for site in [bezpy.mt.read_xml(f) for f in list_of_files]}
list_of_sites = sorted(MT_sites.keys())
MT_xys = np.array([[site.latitude, site.longitude] for site in MT_sites.values()])


def get_intersections(df, lon_bounds, lat_bounds):
    spatial_index = df.sindex

    x0, x1 = lon_bounds
    y0, y1 = lat_bounds
    polygon = shapely.geometry.Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    possible_matches_index = list(spatial_index.intersection(polygon.bounds))
    possible_matches = df.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(polygon)]
    return precise_matches


# Transmission Lines
transmission_lines_file = '../data/Electric_Power_Transmission_Lines/Electric_Power_Transmission_Lines.shp'

df_tl = gpd.read_file(transmission_lines_file)
# Change all MultiLineString into LineString objects by grabbing the first line
# Will miss a few coordinates, but should be OK as an approximation
df_tl.loc[df_tl["geometry"].apply(lambda x: x.geometryType()) == "MultiLineString","geometry"] = \
    df_tl.loc[df_tl["geometry"].apply(lambda x: x.geometryType()) == "MultiLineString","geometry"].apply(lambda x: x[0])

# Get rid of erroneous 1MV and low power line voltages
# df = df[(df["VOLTAGE"] >= 100)]

# Limit it to where EarthScope data is found
df_tl = get_intersections(df_tl, lon_bounds, lat_bounds)

# Print out the size of the dataframe so far
print("Number of transmission lines within lat/lon: {0}".format(len(df_tl)))

df_tl["obj"] = df_tl.apply(bezpy.tl.TransmissionLine, axis=1)
df_tl["length"] = df_tl.obj.apply(lambda x: x.length)

# Apply interpolation weights
t1 = time.time()
df_tl.obj.apply(lambda x: x.set_delaunay_weights(MT_xys))
print("Done filling interpolation weights: {0} s".format(time.time()-t1))

# Remove bad integration paths
E_test = np.ones((1, len(MT_xys), 2))

arr_delaunay = np.zeros(shape=(1, len(df_tl)))
for i, tLine in enumerate(df_tl.obj):
    arr_delaunay[:, i] = tLine.calc_voltages(E_test, how='delaunay')

df_tl = df_tl[~np.isnan(arr_delaunay[0, :])]
print("Number of transmission lines within MT site boundary: {0}".format(len(df_tl)))


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
    """Calculate E at all site locations requested.

    B_sites should have the second dimension being the number of sites.
    """
    E_sites = np.zeros(B_sites.shape)
    for i, site in enumerate(sites.values()):
        Ex, Ey = site.convolve_fft(B_sites[:, i, 0], B_sites[:, i, 1], dt=1/fs)
        E_sites[:, i, 0] = Ex
        E_sites[:, i, 1] = Ey

    # Force E to 0 at t=0
    E_sites[0, :, :] = 0
    # Change to V/km
    return E_sites/1000.


def calc_E_halfspace(B_sites, fs):
    """Calculate E based on half-space impedance."""
    # Depth, then resistivity (Ohm-m)
    cond = 1e-3
    site = bezpy.mt.Site1d('halfspace', [1000], [1/cond, 1/cond])

    E_sites = np.zeros(B_sites.shape)
    # iterate over the second dimensions of B (site locations)
    for i in range(B_sites.shape[1]):
        Ex, Ey = site.convolve_fft(B_sites[:, i, 0], B_sites[:, i, 1], dt=1/fs)
        E_sites[:, i, 0] = Ex
        E_sites[:, i, 1] = Ey

    # Force E to 0 at t=0
    E_sites[0, :, :] = 0
    # Change to V/km
    return E_sites/1000.


def calc_V_lines(df_tl, E_sites):
    """Calculate V over all transmission lines."""
    ntimes = len(E_sites)
    n_trans_lines = len(df_tl)

    arr_delaunay = np.zeros(shape=(ntimes, n_trans_lines))
    # Iterate over all transmission lines
    t0 = time.time()
    for i, tLine in enumerate(df_tl.obj):
        if i % 100 == 0:
            print(i, "transmission lines done: ", time.time()-t0, "s")
        arr_delaunay[:, i] = tLine.calc_voltages(E_sites[..., :2], how='delaunay')

    return arr_delaunay


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


def animate_fields(E_sites, E_half_sites, ts, fs):
    from matplotlib import animation
    # Set up the figure and axes
    fig = plt.figure(figsize=(11, 6), constrained_layout=True)
    height_ratios = [10, 1, 10]
    gs = fig.add_gridspec(ncols=3, nrows=3, height_ratios=height_ratios,
                          hspace=0.4, wspace=0.05)
    # B-field first column
    ax_time = fig.add_subplot(gs[0, 0])
    ax_bfield_cbar = fig.add_subplot(gs[1, 0])
    ax_bfield = fig.add_subplot(gs[2, 0], projection=projection)

    # E-field second column
    ax_efield = fig.add_subplot(gs[0, 1], projection=projection)
    ax_efield_cbar = fig.add_subplot(gs[1, 1])
    ax_efield2 = fig.add_subplot(gs[2, 1], projection=projection)

    # Voltage third column
    ax_voltage = fig.add_subplot(gs[0, 2], projection=projection)
    ax_voltage_cbar = fig.add_subplot(gs[1, 2])
    ax_voltage2 = fig.add_subplot(gs[2, 2], projection=projection)

    for ax in [ax_bfield, ax_efield, ax_efield2, ax_voltage, ax_voltage2]:
        add_features_to_ax(ax)
        ax.set_extent(plot_lon_bounds + lat_bounds, proj_data)

    for ax, label in zip([ax_time, ax_bfield, ax_efield,
                          ax_efield2, ax_voltage, ax_voltage2],
                         ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']):
        if label == '(a)':
            color = 'k'
        else:
            color = 'w'
        ax.text(.02, .96, label,
                fontsize=12, color=color, va='top', ha='left', zorder=3,
                transform=ax.transAxes)

    # Time series
    # -----------
    ax = ax_time
    B_E3A = Bt_E3A(ts)
    B_E3B = Bt_E3B(ts)
    ax_time.plot(ts + 1, B_E3A, c='r')
    ax_time.plot(ts + 1, B_E3B, c='b')
    ax_time.plot(ts + 1, B_E3A + B_E3B, c='k')
    time_line = ax_time.axvline(0, c='k')
    # zero line
    ax_time.axhline(0, c='gray', zorder=-5)
    ax_time.set_xlim(1, 1e3)
    ax_time.set_xscale('log')
    # ax_time.set_xticks()
    ax_time.xaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
    minors = np.arange(10)
    minors = ([x for x in minors] +
              [x*10 for x in minors] +
              [x*100 for x in minors])
    ax_time.xaxis.set_minor_locator(FixedLocator(minors))
    ax_time.set_ylim(-1600, 2100)
    ax_time.set_yticks([-1500, -1000, -500, 0, 500, 1000, 1500, 2000])
    ax_time.set_ylabel("$B_h$ (nT)")

    # create grid for map
    enhance = 3
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

    normB = LogNorm(1, 2500)
    ax = ax_bfield
    ax_cbar = ax_bfield_cbar
    pcol = ax_bfield.pcolormesh(lon_edges, lat_edges,
                                Bh[0, :].reshape(lat_mesh.shape),
                                transform=proj_data,
                                norm=normB,
                                alpha=.5,
                                linewidth=0)
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
                       scale=5000)

    # Red x marks the spot
    ax.scatter(lon_epi, lat_epi, color='r', marker='x',
               s=50, transform=proj_data)

    # E-field
    # -------
    scaleE = 40

    # E-field half-space
    Egrid = calc_E_halfspace(B, fs)
    Ex = Egrid[:, :, 0]
    Ey = Egrid[:, :, 1]
    Ehgrid = np.sqrt(Ex**2 + Ey**2)

    normE = LogNorm(0.1, 50)
    ax = ax_efield
    ax_cbar = ax_efield_cbar
    pcolE = ax.pcolormesh(lon_edges, lat_edges,
                          Ehgrid[0, :].reshape(lat_mesh.shape),
                          transform=proj_data,
                          norm=normE,
                          alpha=.5,
                          linewidth=0)

    cb = plt.colorbar(pcolE, cax=ax_cbar, orientation='horizontal')
    cb.set_label(label='$E_h$ (V/km)', fontsize=12)
    cb.ax.tick_params(labelsize=12)

    # Quiver E-field
    Eq = calc_E_halfspace(Bq, fs)
    Eqx = Eq[:, :, 0]
    Eqy = Eq[:, :, 1]

    quiv_Egrid = ax.quiver(lonq.ravel(), latq.ravel(),
                           Eqy[0, :], Eqx[0, :],
                           transform=proj_data,
                           color='w',
                           units='inches',
                           scale=scaleE)

    # Red x marks the spot
    ax.scatter(lon_epi, lat_epi, color='r', marker='x',
               s=50, transform=proj_data)
    qk = ax.quiverkey(quiv_Egrid, 0.45, -0.1, 10, r'$10 \frac{V}{km}$',
                      color='k', labelpos='E', fontproperties={'size': 12})

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

    ax = ax_efield2
    # Red x marks the spot
    ax.scatter(lon_epi, lat_epi, color='r', marker='x',
               s=50, transform=proj_data)
    # Arrows for the actual site data
    quiv_E = ax.quiver(pred_lons, pred_lats,
                       Ey[0, :], Ex[0, :],
                       Eh[0, :], norm=normE,
                       transform=proj_data,
                       units='inches',
                       scale=scaleE)

    # Voltages
    # --------
    # We need to multiply by 1000 to undo our previous E division, turning from mV to V.
    voltages_orig = calc_V_lines(df_tl, E_sites)*1000
    voltages_half_orig = calc_V_lines(df_tl, E_half_sites)*1000
    voltages = np.abs(voltages_orig)
    voltages_half = np.abs(voltages_half_orig)
    print("Min/max voltages:", np.min(voltages), np.max(voltages))
    print("Min/max half-space voltages:", np.min(voltages_half), np.max(voltages_half))
    line_loc = np.argmax(np.max(voltages, axis=0))
    print("Line number with max voltage:", line_loc)
    newfig = plt.figure()
    newax = newfig.add_subplot()
    newax.plot(ts + 1, voltages_orig[:, line_loc], c='r')
    newax.plot(ts + 1, voltages_half_orig[:, line_loc], c='b')
    newax.axhline(0, c='gray', zorder=-5)
    newax.set_xscale('log')
    newax.set_xlim(1, 1e3)
    newfig.savefig('../figs/voltage_comparison.png')
    # Save the V and time data
    np.savetxt(f"../data/voltage_time.csv", ts + 1, delimiter=',', header="time")
    np.savetxt(f"../data/voltage_3D.csv", voltages_orig[:, line_loc], delimiter=',', header="Voltage")
    np.savetxt(f"../data/voltage_halfspace.csv", voltages_half_orig[:, line_loc], delimiter=',', header="Voltage")
    line_geom = [np.array(linestring)[:, :2] for linestring in df_tl['geometry']][line_loc]
    np.savetxt(f"../data/voltage_line_coordinates.csv", line_geom, delimiter=',', header="lon,lat")

    # Set the first time to the minimum norm value to make sure they appear
    # in the very first frame
    voltages[0, :] = 10
    voltages_half[0, :] = 10
    coll = mpl.collections.LineCollection([np.array(linestring)[:, :2] for linestring in df_tl['geometry']])
    coll_half = mpl.collections.LineCollection([np.array(linestring)[:, :2] for linestring in df_tl['geometry']])
    vmin, vmax = 10, 2000
    cmapV = plt.get_cmap('magma')
    normV = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)

    coll.set_cmap(cmapV)
    coll.set_norm(normV)
    coll.set_transform(proj_data) 
    coll.set_linewidths(1)

    coll_half.set_cmap(cmapV)
    coll_half.set_norm(normV)
    coll_half.set_transform(proj_data)   
    coll_half.set_linewidths(1)
    coll.set_array(voltages[0, :])
    coll_half.set_array(voltages_half[0, :])

    # The top panel is the half-space
    ax_voltage.add_collection(coll_half)
    ax_voltage2.add_collection(coll)

    sm = mpl.cm.ScalarMappable(cmap=cmapV, norm=normV)
    # Set scalar mappable array
    sm._A = []
    cbar = mpl.colorbar.Colorbar(ax=ax_voltage_cbar, mappable=sm, orientation='horizontal')
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Voltage (V)', size=12)
    # cbar.ax.xaxis.set_label_position('top')
    cbar.set_ticks([10, 100, 1000])
    # cbar.set_ticklabels(['10', '100', '1000'])

    title = fig.suptitle('Time: 0 s')

    # Need to save first to get tight layout to work.
    fig.savefig('../figs/test.png')

    def animate(t):
        # t *= 10
        time_line.set_xdata(ts[t] + 1)
        # B-fields
        pcol.set_array(Bh[t, :])
        quiv_B.set_UVC(Bqy[t, :], Bqx[t, :])
        # Half-space E-fields
        pcolE.set_array(Ehgrid[t, :])
        quiv_Egrid.set_UVC(Eqy[t, :], Eqx[t, :])
        # Site E-fields
        quiv_E.set_UVC(Ey[t, :], Ex[t, :], Eh[t, :])
        # Voltages
        coll.set_array(voltages[t, :])
        coll_half.set_array(voltages_half[t, :])
        title.set_text(f'Time: {(ts[t]+1):.2f} s')

    fig_vdiff, ax_vdiff = plt.subplots(subplot_kw={'projection': projection})
    add_features_to_ax(ax_vdiff)
    ax_vdiff.set_extent(plot_lon_bounds + lat_bounds, proj_data)
    coll_vdiff = mpl.collections.LineCollection([np.array(linestring)[:, :2] for linestring in df_tl['geometry']])
    cmapVdiff = plt.get_cmap('RdBu_r')
    normVdiff = mpl.colors.Normalize(-1000, 1000)
    normVdiff = mpl.colors.SymLogNorm(vmin=-1000, vmax=1000, linthresh=10)

    coll_vdiff.set_cmap(cmapVdiff)
    coll_vdiff.set_norm(normVdiff)
    coll_vdiff.set_transform(proj_data)
    coll_vdiff.set_linewidths(1)
    coll_vdiff.set_array(voltages[0, :] - voltages_half[0, :])
    ax_vdiff.add_collection(coll_vdiff)

    sm = mpl.cm.ScalarMappable(cmap=cmapVdiff, norm=normVdiff)
    # Set scalar mappable array
    sm._A = []
    cbar = fig_vdiff.colorbar(sm, orientation='horizontal')
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # cbar.ax.tick_params(labelsize=12)
    cbar.set_label('Voltage difference [3D - halfspace] (V)', size=12)
    # cbar.ax.xaxis.set_label_position('top')
    # cbar.set_ticks([10, 100, 1000])

    # Make 3 snapshots
    # t = 0.5
    # t = 2
    # t = 19
    t = int(0.5*fs)
    animate(t)
    fig.savefig('../figs/fig10.png')
    coll_vdiff.set_array(voltages[t, :] - voltages_half[t, :])
    fig_vdiff.savefig('../figs/fig10_vdiff.png')
    t = int(2*fs)
    animate(t)
    fig.savefig('../figs/fig11.png')
    coll_vdiff.set_array(voltages[t, :] - voltages_half[t, :])
    fig_vdiff.savefig('../figs/fig11_vdiff.png')
    t = int(19*fs)
    animate(t)
    fig.savefig('../figs/fig12.png')
    coll_vdiff.set_array(voltages[t, :] - voltages_half[t, :])
    fig_vdiff.savefig('../figs/fig12_vdiff.png')
    animate(0)

    anim = animation.FuncAnimation(fig, animate,
                                   frames=[x for x in range(100)],
                                   interval=10)
    anim.save('../figs/animation.mp4')


def calc_specific_site(name, ts, fs):
    if name not in MT_sites:
        raise ValueError(f"{name} is not in the site dictionary.")
    site = MT_sites[name]

    lats = np.array([site.latitude, 0])
    lons = np.array([site.longitude, 0])


    # Pass in the lat/lon of the requested site
    B_E3A, B_E3B = calc_B(lats, lons, ts)
    # Add together for total B field
    B = B_E3A + B_E3B
    E = calc_E_sites({name: MT_sites[name]}, B, fs)[:, 0, :2]
    E_half = calc_E_halfspace(B, fs)[:, 0, :2]
    B = B[:, 0, :2]

    np.savetxt(f"../data/{name}_time.csv", ts + 1, delimiter=',', header="time")
    np.savetxt(f"../data/{name}_B.csv", B, delimiter=',', header="Bx,By")
    np.savetxt(f"../data/{name}_E.csv", E, delimiter=',', header="Ex,Ey")
    np.savetxt(f"../data/{name}_E_half.csv", E_half, delimiter=',', header="Ex,Ey")

    fig, ax = plt.subplots()
    ax.plot(ts + 1, E[:, 0], 'b', label='$E_x$')
    ax.plot(ts + 1, E[:, 1], 'r', label='$E_y$')
    ax.plot(ts + 1, E_half[:, 0], 'b--', label='$Ehalf_x$')
    ax.plot(ts + 1, E_half[:, 1], 'r--', label='$Ehalf_y$')
    ax.set_ylabel('E (V/km)')
    ax.set_xlabel('Time (s)')
    ax.set_xscale('log')
    ax.set_title(f"Site {name}")
    ax.set_xlim(1, 1e3)
    ax.legend()
    plt.show()
    fig.savefig(f"../figs/{name}_E.png", bbox_inches='tight')


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

    # fs = 100  # sampling frequency
    ts = np.arange(ntimes) / fs  # time steps

    # TODO: Remove to save data at specific sites
    # calc_specific_site("SFM06", ts, fs)
    # calc_specific_site("RF111", ts, fs)


    B_sites_E3A, B_sites_E3B = calc_B(MT_xys[:, 0], MT_xys[:, 1], ts)
    # Add together for total B field
    B_sites = B_sites_E3A + B_sites_E3B
    E_sites = calc_E_sites(MT_sites, B_sites, fs)
    E_half_sites = calc_E_halfspace(B_sites, fs)
    # return
    animate_fields(E_sites, E_half_sites, ts, fs)
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
