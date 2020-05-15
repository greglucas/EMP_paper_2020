import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import bezpy
# from zfile import ZFile

# ADD THIS FOR GREG STYLING
# plt.style.use('seaborn-paper')
plt.style.use(['seaborn-paper', 'tex.mplstyle'])

mt_data_folder = '../data/xml_files/'

list_of_files = sorted(glob.glob(mt_data_folder + '*.xml'))
MT_sites = {site.name: site for site in [bezpy.mt.read_xml(f)
                                         for f in list_of_files]}
MT_xys = [(site.latitude, site.longitude) for site in MT_sites.values()]


site_1d = MT_sites['SFM006']

site_3d = MT_sites['RFR111']


def e_polarization(a, z):
    # # E polarization state from Berdichevsky & Dmitriev book
    # l1 = (np.absolute(z[0, 0]) ** 2. + np.absolute(z[0, 1]) ** 2.) / \
    #      np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.
    # l2 = 2. * np.real(z[0, 0] * z[1, 0].conj() + z[1, 1] * z[0, 1].conj()) / \
    #      np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.
    # l3 = (np.absolute(z[1, 1]) ** 2. + np.absolute(z[1, 0]) ** 2.) / \
    #      np.absolute(z[0, 0] * z[1, 1] - z[0, 1] * z[1, 0]) ** 2.
    #
    # z_e = np.sqrt(1. / (l1 * np.sin(a) ** 2. -
    #                     l2 * np.sin(a) * np.cos(a) +
    #                     l3 * np.cos(a) ** 2.))

    # E polarization state from Berdichevsky & Dmitriev book
    l1 = (np.absolute(z[0, 0]) ** 2. +
          np.absolute(z[0, 1]) ** 2.) / \
         np.absolute(z[0, 0] *
                     z[1, 1] -
                     z[0, 1] *
                     z[1, 0]) ** 2.
    l2 = 2. * np.real(z[0, 0] *
                      z[1, 0].conj() +
                      z[1, 1] *
                      z[0, 1].conj()) / \
         np.absolute(z[0, 0] *
                     z[1, 1] -
                     z[0, 1] *
                     z[1, 0]) ** 2.
    l3 = (np.absolute(z[1, 1]) ** 2. +
          np.absolute(z[1, 0]) ** 2.) / \
         np.absolute(z[0, 0] *
                     z[1, 1] -
                     z[0, 1] *
                     z[1, 0]) ** 2.

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
    ax_1d_zb.set_title('H-Polarization', pad=25., fontsize=12)

    ax_3d_ze = fig.add_subplot(gs[1, 0], projection='polar')
    ax_3d_ze.set_theta_zero_location("N")
    ax_3d_ze.set_theta_direction(-1)
    ax_3d_ze.set_ylabel(site_3d, labelpad=32., fontsize=12)

    ax_3d_zb = fig.add_subplot(gs[1, 1], projection='polar')
    ax_3d_zb.set_theta_zero_location("N")
    ax_3d_zb.set_theta_direction(-1)

    for site, filename, ax_ze, ax_zb, legend in zip([site_1d, site_3d],
                                                    [filename_1d, filename_3d],
                                                    [ax_1d_ze, ax_3d_ze],
                                                    [ax_1d_zb, ax_3d_zb],
                                                    [False, True]):

        # load impedance tensor
        zfile = ZFile(filename)

        for period, color in zip([11.6, 102.4, 1092.3], ['r-', 'b-', 'k-']):

            i = np.argmin(np.absolute(zfile.periods - period))
            impedance, error = zfile.impedance(angle=0.)
            z = impedance[i, :, :]

            # evaluate polarization curves
            z_e = e_polarization(angles, z)
            z_h = h_polarization(angles, z)

            # if site == 'VAQ58' and period == 1092.3:
            #     print(np.vstack((angles * 180. / np.pi, z_e)).T)

            if legend:
                ax_zb.plot(angles, z_h, color, label='{:.1f} s'.format(period))
            else:
                ax_zb.plot(angles, z_h, color)
            ax_ze.plot(angles, z_e, color)

        for ax in [ax_ze, ax_zb]:
            ax.set_rlabel_position(180.)
            if site == 'KSR35':
                ax.set_rmin(0.)
                ax.set_rmax(5.)
                ax.set_rticks([1., 2., 3., 4., 5.])
            elif site == 'VAQ58':
                ax.set_rmin(0.)
                ax.set_rmax(100.)
                ax.set_rticks([20., 40., 60., 80., 100.])
                # ax.set_rticklabels(['10 mV/km', '20 mV/km', '30 mV/km', '40 mV/km'])
            ax.tick_params(axis='y', labelsize=8.)
            ax.tick_params(axis='x', labelsize=8.)
            ax.grid(True)

        if legend:
            fig.legend(loc='lower center', ncol=3, fontsize=12)

    plt.subplots_adjust(left=0.1, right=0.94, bottom=0.11, top=0.9,
                        wspace=0.4, hspace=0.4)
    plt.savefig('../figs/fig3_example_peanuts.png', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    main()
