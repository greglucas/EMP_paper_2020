import numpy as np
import scipy.signal as sps

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import cartopy.crs as ccrs
import cartopy.feature as cfeature


#
# Global plotting parameters
#

# style
plt.style.use(['seaborn-paper', './tex.mplstyle'])

# colormap
cmap = get_cmap('RdYlBu_r')
cmap = get_cmap('magma')

# maps
proj_data = ccrs.PlateCarree()
projection = ccrs.LambertConformal(central_latitude=30, central_longitude=-96)

# taken from Greg's workbook
# https://github.com/greglucas/GeoelectricHazardPaper2019/blob/master/code/GeoelectricHazardPaper.ipynb
US_lon_bounds = (-130, -60)
US_lat_bounds = (20, 55)
lon_bounds = US_lon_bounds
plot_lon_bounds = (lon_bounds[0] + 10, lon_bounds[1] - 10)
lat_bounds = US_lat_bounds
plot_lat_bounds = (lat_bounds[0] + 5, lat_bounds[1] - 5)

# angles for ellipse plotting
a = np.linspace(0., 2. * np.pi, num=100)

# 'geom', 'area', 'max'
size_scaling = 'max'
color_scaling = 'max'


#
# Global processing parameters
#

# for Dawson's integral time series
from scipy.special import dawsn as dawson

mu0 = 4*np.pi*1e-7 # permittivity of free space
sigma = 1e-4 # half-space conductivity

EA = 0.04
aA = 0.02
bA = 2
kA = 1.058

EB = 0.01326
aB = 0.015
bB = 0.02
kB = 9.481

fs = 10 # sampling frequency
ts = np.arange(1500000) / fs # time steps


# for double SECS
lat_epi = 37.25 # epicenter latitude (degrees north)
lon_epi = -90.5 # epicenter longitude (degrees east)
rad_earth = 6378e3 # earth radius (meters)
rad_iono = rad_earth + 110e3 # ionosphere radius (meters)
delta = 800e3 # N-S separation distance of SECs (meters)


#
# Plotting methods
# 

# add map features
def add_features_to_ax(ax):
    land_alpha = 0.7
    scale = '50m'
    # 10m oceans are super slow...
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor='slategrey', alpha=0.65, zorder=-1)
    ax.add_feature(cfeature.LAND.with_scale(scale),
                   facecolor='k', alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.STATES.with_scale(scale),
                   edgecolor='w', linewidth=0.4, alpha=land_alpha, zorder=0)
    ax.add_feature(cfeature.LAKES.with_scale(scale),
                   facecolor='slategrey', alpha=0.25, zorder=0)

# plot B1 and B2 time series, along with their superposition
def plot_E3_timeseries(ax):
    """
    Plot time series of B-field at EMP epicenter
    """

    # generate time series
    E3As = Bt_E3A(ts)
    E3Bs = Bt_E3B(ts)

    # generate plots
    ax.plot(ts, E3As)
    ax.plot(ts, E3Bs)
    ax.plot(ts, E3As + E3Bs)
    ax.set_ylim(-1500, 2200)
    ax.set_xlim(0.075, 1.5e5)
    ax.set_xscale('log')
    ax.set_xticks([1e-1, 1e1, 1e3, 1e5])
    ax.tick_params(labelsize=12)
    ax.grid()
    ax.set_ylabel('nT', fontsize='12')
    ax.set_xlabel('seconds', fontsize=12)
    ax.legend(('$B(t)_{E3A}$', '$B(t)_{E3B}$', '$B(t)_{total}$', ),
              bbox_to_anchor=(0.0, 1.15, 1, 0.1), loc='center', 
              fontsize=12, ncol=3)
    
    return ts, E3As, E3Bs

def plot_E3_spectra(ax):
    """
    Plot spectrum of B-field at EMP epicenter
    FIXME: the digital periodogram does not match the analytic spectrum
           returned by Bf() by a very large amount (several orders of mag); 
           not sure where Bf() algorithm comes from, so it's difficult to
           determine which, if either, is correct.
    """
    # frequency spectrum
    freqs, dspectrum = sps.welch(
        Bt_E3A(ts) + Bt_E3B(ts), 
        scaling='spectrum', fs=fs, nperseg=2**12,
    )
    aspectrum = Bf(freqs)

    ax.plot(freqs[1:], np.sqrt(dspectrum[1:]), '.-')
    ax.plot(freqs[1:], aspectrum[1:], '.-')
    # ax2.plot(1./(ts), Bf(1./(ts)), '.-')

    ax.set_xlim(1e-4, 20)
    ax.set_ylim(1, 4e5)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_yticks([.1, 1, 1e1, 1e2, 1e3, 1e4, 1e5])
    ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1, 10])
    ax.grid()
    ax.set_aspect(1.)

    # compare with empirical spectrum
    ax.plot()


# plot E3 B-field maps
def plot_E3_map(ax1, ax2):
    """
    Plot maps of E3A and E3B normalized B-field
    """

    # draw map
    ax1.set_extent(plot_lon_bounds + plot_lat_bounds, proj_data)
    add_features_to_ax(ax1)

    # create grid for map
    pred_lons = np.linspace(lon_bounds[0], lon_bounds[1], 
                            int(np.diff(lon_bounds) / 1.5) + 1)
    pred_lats = np.linspace(lat_bounds[0], lat_bounds[1], 
                            int(np.diff(lat_bounds) / 1.5) + 1)
    lon_mesh, lat_mesh = np.meshgrid(pred_lons, pred_lats)

    # generate gridded B-field
    Bx_E3A, By_E3A, Bz_E3A = Br_E3A(lat_mesh.ravel(), lon_mesh.ravel())
    Bx_E3B, By_E3B, Bz_E3B = Br_E3B(lat_mesh.ravel(), lon_mesh.ravel())

    # calculate gridded horizontal B-field intensities (H)
    Bh_E3A = np.sqrt(Bx_E3A**2 + By_E3A**2)
    Bh_E3B = np.sqrt(Bx_E3B**2 + By_E3B**2)
    
    norm = LogNorm(0.01, 1.1)
    cax1 = ax1.pcolormesh(lon_mesh, lat_mesh, 
                          Bh_E3A.reshape(lat_mesh.shape), 
                          transform=proj_data,
                          norm=norm,
                          alpha=.5,
                          linewidth=0)
    cax1.set_edgecolor('face')

    ax1.quiver(lon_mesh, lat_mesh, 
               By_E3A.reshape(lat_mesh.shape),
               Bx_E3A.reshape(lat_mesh.shape),
               transform=proj_data,
               scale=40,
               pivot='middle',
               color='w')
    # cb1 = plt.colorbar(cax1, ax=ax1, orientation='vertical')
    # cb1.set_label(label='$B_h$ (nT)', fontsize=12)
    # cb1.ax.tick_params(labelsize=12)
    t1 = ax1.text(.02, .88, '(a) E3A basis $\mathbf{b}^A(x,y)$',
             fontsize=12, color='k', va='bottom', ha='left', zorder=3,
             transform=ax1.transAxes)
    t1.set_bbox(dict(facecolor='w', edgecolor='none',
                alpha=0.75, boxstyle='round'))
    # draw a box of the zoomed-in area
    # (box limits taken from EMP paper)
    from shapely.geometry.polygon import LinearRing
    lons = [-88, -88, -93, -93]
    lats = [35, 39.5, 39.5, 35]
    ring = LinearRing(list(zip(lons, lats)))
    ax1.add_geometries([ring], ccrs.PlateCarree(), 
                       linewidth=1, edgecolor='red', facecolor='none')


    
    # draw map
    ax2.set_extent(plot_lon_bounds + plot_lat_bounds, proj_data)
    ax2.set_extent
    add_features_to_ax(ax2)

    norm = LogNorm(0.01, 1.1)
    cax2 = ax2.pcolormesh(lon_mesh, lat_mesh, 
                          Bh_E3B.reshape(lat_mesh.shape), 
                          transform=proj_data,
                          norm=norm,
                          alpha=.5,
                          linewidth=0)
    cax2.set_edgecolor('face')

    ax2.quiver(lon_mesh, lat_mesh, 
               By_E3B.reshape(lat_mesh.shape),
               Bx_E3B.reshape(lat_mesh.shape),
               transform=proj_data,
               scale=40,
               pivot='middle',
               color='w')
    # cb2 = plt.colorbar(cax2, ax=ax2, orientation='vertical')
    # cb2.set_label(label='$B_h$ (nT)', fontsize=12)
    # cb2.ax.tick_params(labelsize=12)
    t2 = ax2.text(.02, .88, '(b) E3B basis $\mathbf{b}^B(x,y)$',
             fontsize=12, color='k', va='bottom', ha='left', zorder=3,
             transform=ax2.transAxes)
    t2.set_bbox(dict(facecolor='w', edgecolor='none',
                alpha=0.75, boxstyle='round'))
    
    # draw a box of the zoomed-in area
    # (box limits taken from EMP paper)
    from shapely.geometry.polygon import LinearRing
    lons = [-88, -88, -93, -93]
    lats = [35, 39.5, 39.5, 35]
    ring = LinearRing(list(zip(lons, lats)))
    ax2.add_geometries([ring], ccrs.PlateCarree(), 
                       linewidth=1, edgecolor='red', facecolor='none')

    # force colorbar between panels (adapted/stolen from Ben Murphy)
    fig = plt.gcf()
    cbar_ax = inset_axes(ax1, width='90%', height='4%', loc='lower center',
                         bbox_to_anchor=(0., 0.5, 1., 1.),
                         bbox_transform=fig.transFigure, borderpad=0)
    # cax = ax1.scatter([], [], s=1., c=[], cmap=cmap, norm=norm,
    #                  marker='o')
    cbar = fig.colorbar(cax2, cax=cbar_ax, orientation='horizontal',
                        use_gridspec=True, fraction=1., aspect=35.)
    cbar.set_label('$B_h$ [nT]', fontsize=12,
                   labelpad=4, rotation=0.)
    cbar.ax.tick_params(labelsize=12)




#
# Processing methods
#

# Separate out temporal B-field for E3A and E3B
def Bt_E3A(t):
    """
    Compute synthetic B-field time series from Equation 18 for E3A
    """
    func = (EA*kA)*(dawson(np.sqrt(aA*t))/np.sqrt(aA) -
                    dawson(np.sqrt(bA*t))/np.sqrt(bA))
    Bt = 2*np.sqrt(mu0*sigma/np.pi) * (func) * 1e9
    return Bt

def Bt_E3B(t):
    """
    Compute synthetic B-field time series from Equation 18 for E3AB
    """
    func = (EB*kB)*(dawson(np.sqrt(aB*t))/np.sqrt(aB) -
                    dawson(np.sqrt(bB*t))/np.sqrt(bB))
    Bt = -2*np.sqrt(mu0*sigma/np.pi) * (func) * 1e9
    return Bt

def Bf(f):
    """Function to compute the synthetic B frequency spectrum."""
    yf = np.sqrt((EA*(aB*bB*kA*(bA - aA) - 4*np.pi**2*f**2*kA*(bA - aA)) +
                  EB*(aA*bA*kB*(aB - bB) - 4*np.pi**2*f**2*kB*(aB - bB)))**2 +
                 (EA*(2*np.pi*aB*f*kA*(bA - aA) + 2*np.pi*bB*f*kA*(bA - aA)) +
                  EB*(2*np.pi*aA*f*kB*(aB - bB) + 2*np.pi*bA*f*kB*(aB - bB)))**2)
                    
    yf = yf/(np.sqrt(4*np.pi**2*f**2+aB**2)*np.sqrt(4*np.pi**2*f**2+bB**2)*
             np.sqrt(4*np.pi**2*f**2+aA**2)*np.sqrt(4*np.pi**2*f**2+bA**2))

    Bf = (1.0/np.sqrt(2.0))*np.sqrt(sigma*mu0/(np.pi*f))*yf * 1e9
    
    return Bf


# generate time-stationary, normalized spatial B-fields for E3A and E3B
def Br_E3A(lats, lons):
    """
    Compute normalized spatial vector field from Eq. 18 for E3A.
    """
    # generate a uniform northward B-field
    Bx = np.ones(lats.shape)
    By = np.zeros_like(Bx)
    Bz = np.zeros_like(Bx)
    return Bx, By, Bz

def Br_E3B(lats, lons, ret_secs=False):
    """
    Compute normalized spatial vector field from Eq. 18 for E3B.
    (if ret_secs is True, return the secs object for further analysis)
    
    Expects the following global variables to be defined:
    
    rad_iono  - radius of spherical earth in meters
    rad_earth - radius of spherical earth in meters
    delta     - separation of secs in meters

    """
    # use double SECS where each SEC is separated from the blast epicenter
    # by delta in the north(south) direction. The SEC amplitudes will be equal
    # and opposite, and produce a unit amplitude B-field at the epicenter.
    import pysecs
    
    # location of SECs
    delta_lat = (delta / rad_iono) * 180 / np.pi
    loc_sec1 = np.array([lat_epi + delta_lat / 2, lon_epi, rad_iono])
    loc_sec2 = np.array([lat_epi - delta_lat / 2, lon_epi, rad_iono])
    loc_secs = np.stack([loc_sec1, loc_sec2])
    
    secs = pysecs.SECS(sec_df_loc=loc_secs)

    # fit this B-field at epicenter
    Bx_epi = 1e-9 # Teslas
    By_epi = 0
    Bz_epi = 0
    loc_obs = np.array([[lat_epi, lon_epi, rad_earth]])
    B_obs = np.array([[Bx_epi, By_epi, Bz_epi]])
    secs.fit(loc_obs, B_obs)

    # predict B-field at specified locations
    loc_preds = np.empty((lats.size, 3))
    loc_preds[:, 0] = lats
    loc_preds[:, 1] = lons
    loc_preds[:, 2] = rad_earth
    B_pred = secs.predict(loc_preds)
    Bx = B_pred[:,0] * 1e9 # nanoTeslas
    By = B_pred[:,1] * 1e9 # nanoTeslas
    Bz = B_pred[:,2] * 1e9 # nanoTeslas

    if ret_secs:
        return Bx, By, Bz, secs
    else:
        return Bx, By, Bz


#
# Main method for command-line calls
#

def main():
    
    fig1 = plt.figure(figsize=(6.5, 2))
    gs1 = fig1.add_gridspec(ncols=1, nrows=1, height_ratios=[1])
    ax1 = fig1.add_subplot(gs1[0])
    
    plot_E3_timeseries(ax1)
    plt.subplots_adjust(left=0.11, right=0.99, top=0.77, bottom=0.33)
    plt.savefig('../figs/figXX_E3A_E3B_Bfields_timeseries.png', dpi=300)
    plt.savefig('../figs/figXX_E3A_E3B_Bfields_timeseries.pdf')

    fig2 = plt.figure(figsize=(4, 6))
    gs2 = fig2.add_gridspec(ncols=1, nrows=2, height_ratios=[1,1])
    ax3 = fig2.add_subplot(gs2[0], projection=projection)
    ax4 = fig2.add_subplot(gs2[1], projection=projection)

    plot_E3_map(ax3, ax4)

    # plt.subplots_adjust(left=0.11, right=0.99, top=0.9, bottom=0.05)
    plt.subplots_adjust(left=0.05, right=0.95, top=0.99, bottom=0.01, hspace=.52)


    plt.savefig('../figs/figXX_E3A_E3B_Bfields_basis.png', dpi=300)
    plt.savefig('../figs/figXX_E3A_E3B_Bfields_basis.pdf')
    
    plt.show()

    plt.close(fig1)
    plt.close(fig2)


if __name__ == "__main__":
    main()
