import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import bezpy
from scipy.interpolate import interp1d

plt.style.use(['seaborn-paper', 'tex.mplstyle'])

mt_data_folder = '../data'

list_of_files = sorted(glob.glob(mt_data_folder + '*.xml'))
MT_sites = {site.name: site for site in [bezpy.mt.read_xml(f)
                                         for f in list_of_files]}
MT_xys = [(site.latitude, site.longitude) for site in MT_sites.values()]

# site_1d = MT_sites['SFM06']
site_1d = 'SFM06'
filename_1d = '../data/SFM06.xml'
# site_3d = MT_sites['RFR111']
site_3d = 'RF111'
filename_3d = '../data/RF111.xml'


def e_polarization(a, z):
    # E polarization state from Berdichevsky & Dmitriev book
    l1 = (np.absolute(z[0, 0]) ** 2. + np.absolute(z[0, 1]) ** 2.) / \
         np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.
    l2 = 2. * np.real(z[0, 0] * z[1, 0].conj() + z[1, 1] * z[0, 1].conj()) / \
         np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.
    l3 = (np.absolute(z[1, 1]) ** 2. + np.absolute(z[1, 0]) ** 2.) / \
         np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.

    z_e = np.sqrt(1. / (l1 * np.sin(a) ** 2. -
                        l2 * np.sin(a) * np.cos(a) +
                        l3 * np.cos(a) ** 2.))

    return z_e


def h_polarization(a, z):
    # H polarization state from Berdichevsky & Dmitriev book
    k1 = np.absolute(z[0, 1]) ** 2. + np.absolute(z[1, 1]) ** 2.
    k2 = 2. * np.real(z[0, 0] * z[0, 1].conj() + z[1, 0] * z[1, 1].conj())
    k3 = np.absolute(z[0, 0]) ** 2. + np.absolute(z[1, 0]) ** 2.

    z_h = np.sqrt(k1 * np.sin(a) ** 2. +
                  k2 * np.sin(a) * np.cos(a) +
                  k3 * np.cos(a) ** 2.)

    return z_h


def main():

    # angles for ellipse plotting
    angles = np.linspace(0., 2. * np.pi, num=500)

    fig = plt.figure(figsize=(6.5, 7.))
    gs = GridSpec(2, 2)

    ax_1d_ze = fig.add_subplot(gs[0, 0], projection='polar')
    ax_1d_ze.set_theta_zero_location("N")
    ax_1d_ze.set_theta_direction(-1)
    ax_1d_ze.set_title('E-Polarization', pad=25., fontsize=12)
    ax_1d_ze.set_ylabel(site_1d, labelpad=32., fontsize=12)

    ax_1d_zb = fig.add_subplot(gs[0, 1], projection='polar')
    ax_1d_zb.set_theta_zero_location("N")
    ax_1d_zb.set_theta_direction(-1)
    ax_1d_zb.set_title('B-Polarization', pad=25., fontsize=12)

    ax_3d_ze = fig.add_subplot(gs[1, 0], projection='polar')
    ax_3d_ze.set_theta_zero_location("N")
    ax_3d_ze.set_theta_direction(-1)
    ax_3d_ze.set_ylabel(site_3d, labelpad=32., fontsize=12)

    ax_3d_zb = fig.add_subplot(gs[1, 1], projection='polar')
    ax_3d_zb.set_theta_zero_location("N")
    ax_3d_zb.set_theta_direction(-1)

    for sitename, filename, ax_ze, ax_zb, legend in zip(
            [site_1d, site_3d], [filename_1d, filename_3d],
            [ax_1d_ze, ax_3d_ze], [ax_1d_zb, ax_3d_zb], [False, True]):

        # load impedance tensor
        site = bezpy.mt.read_xml(filename)
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

        for period, color, label in zip([0.1, 1., 10., 100., 1000.],
                                        ['r-', 'g-', 'b-', 'm-', 'k-'],
                                        ['0.1 s', '1 s', '10 s', '100 s',
                                         '1000 s']):

            zxx = zxx_re_interpolator(period) + 1.j * zxx_im_interpolator(period)
            zxy = zxy_re_interpolator(period) + 1.j * zxy_im_interpolator(period)
            zyx = zyx_re_interpolator(period) + 1.j * zyx_im_interpolator(period)
            zyy = zyy_re_interpolator(period) + 1.j * zyy_im_interpolator(period)

            # evaluate polarization curves
            z_e = e_polarization(angles, np.array([[zxx, zxy], [zyx, zyy]]))
            z_h = h_polarization(angles, np.array([[zxx, zxy], [zyx, zyy]]))

            if legend:
                ax_zb.plot(angles, z_h, color, label=label)
            else:
                ax_zb.plot(angles, z_h, color)
            ax_ze.plot(angles, z_e, color)

        for ax in [ax_ze, ax_zb]:
            ax.set_rlabel_position(180.)
            if sitename == 'RF111':
                ax.set_rmin(0.)
                ax.set_rmax(800.)
                ax.set_rticks([200., 400., 600., 800.])
                # ax.set_rmax(120.)
                # ax.set_rticks([20., 40., 60., 80., 100.])
            elif sitename == 'SFM06':
                ax.set_rmin(0.)
                ax.set_rmax(20.)
                ax.set_rticks([5., 10., 15., 20.])
                # ax.set_rticks([2.5, 5., 7.5, 10., 12.5, 15.])
                # ax.set_yticklabels(['', '5'])
            ax.tick_params(axis='y', labelsize=8.)
            ax.tick_params(axis='x', labelsize=8.)
            ax.grid(True)

        if legend:
            fig.legend(loc='lower center', ncol=5, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.94, bottom=0.11, top=0.9,
                        wspace=0.4, hspace=0.4)
    plt.savefig('../figs/fig3_example_peanuts.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
