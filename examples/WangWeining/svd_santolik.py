import numpy as np
from munch import Munch
import scipy
import time


def m_svd(wt_dict, lbg_dict):
    """magnetic field only SVD method"""
    num_t = wt_dict.epoch.size
    num_p = wt_dict.freq.size

    '''calculate A matrix'''
    a_arr = np.zeros((6, 3, num_p, num_t))
    a_arr[0, 0, ...] = np.real(wt_dict.x_wt * np.conj(wt_dict.x_wt))
    a_arr[0, 1, ...] = np.real(wt_dict.x_wt * np.conj(wt_dict.y_wt))
    a_arr[0, 2, ...] = np.real(wt_dict.x_wt * np.conj(wt_dict.z_wt))
    a_arr[1, 0, ...] = np.real(wt_dict.x_wt * np.conj(wt_dict.y_wt))
    a_arr[1, 1, ...] = np.real(wt_dict.y_wt * np.conj(wt_dict.y_wt))
    a_arr[1, 2, ...] = np.real(wt_dict.y_wt * np.conj(wt_dict.z_wt))
    a_arr[2, 0, ...] = np.real(wt_dict.x_wt * np.conj(wt_dict.z_wt))
    a_arr[2, 1, ...] = np.real(wt_dict.y_wt * np.conj(wt_dict.z_wt))
    a_arr[2, 2, ...] = np.real(wt_dict.z_wt * np.conj(wt_dict.z_wt))
    a_arr[3, 0, ...] = np.zeros([num_p, num_t])
    a_arr[3, 1, ...] = -np.imag(wt_dict.x_wt * np.conj(wt_dict.y_wt))
    a_arr[3, 2, ...] = -np.imag(wt_dict.x_wt * np.conj(wt_dict.z_wt))
    a_arr[4, 0, ...] = +np.imag(wt_dict.x_wt * np.conj(wt_dict.y_wt))
    a_arr[4, 1, ...] = np.zeros([num_p, num_t])
    a_arr[4, 2, ...] = -np.imag(wt_dict.y_wt * np.conj(wt_dict.z_wt))
    a_arr[5, 0, ...] = +np.imag(wt_dict.x_wt * np.conj(wt_dict.z_wt))
    a_arr[5, 1, ...] = +np.imag(wt_dict.y_wt * np.conj(wt_dict.z_wt))
    a_arr[5, 2, ...] = np.zeros([num_p, num_t])

    '''Initialize eigenvalue and eigenvector arrays'''
    s_v2_arr = np.zeros((3, num_p, num_t))
    vh_v2_arr = np.zeros((3, 3, num_p, num_t))

    '''calculate background magnetic field'''
    b0_arr = np.zeros((3, num_p, num_t))
    b0_arr[0, ...] = lbg_dict.bx_lbg_arr
    b0_arr[1, ...] = lbg_dict.by_lbg_arr
    b0_arr[2, ...] = lbg_dict.bz_lbg_arr

    '''calculate fac coordinates'''
    ez_arr = b0_arr / scipy.linalg.norm(b0_arr, axis=0)
    ex_arr = np.tile(np.array([1, 0, 0])[:, None, None], (1, num_p, num_t))
    ep1_arr = np.transpose(np.cross(ez_arr, ex_arr, axisa=0, axisb=0), (2, 0, 1))
    ep1_arr = ep1_arr / np.tile(scipy.linalg.norm(ep1_arr, axis=0)[None, ...], (3, 1, 1))
    ep2_arr = np.transpose(np.cross(ez_arr, ep1_arr, axisa=0, axisb=0), (2, 0, 1))
    ep2_arr = ep2_arr / np.tile(scipy.linalg.norm(ep2_arr, axis=0)[None, ...], (3, 1, 1))

    '''calculate wavelet coefficients perpendicular to the background magnetic field'''
    bx_fac_wt = wt_dict.x_wt * ep1_arr[0, ...] + wt_dict.y_wt * ep1_arr[1, ...] + wt_dict.z_wt * ep1_arr[2, ...]
    by_fac_wt = wt_dict.x_wt * ep2_arr[0, ...] + wt_dict.y_wt * ep2_arr[1, ...] + wt_dict.z_wt * ep2_arr[2, ...]

    '''calculate energy'''
    sxx_fac = bx_fac_wt * np.conj(bx_fac_wt)
    sxy_fac = bx_fac_wt * np.conj(by_fac_wt)
    syy_fac = by_fac_wt * np.conj(by_fac_wt)

    t = time.time()
    for it in range(num_t):
        for ip in range(num_p):
            a = a_arr[..., ip, it]
            '''calculation'''
            u, s, vh = scipy.linalg.svd(a)
            ascend_order = np.argsort(s)
            s_v2 = s[ascend_order]
            vh_v2 = vh[s.argsort()]
            s_v2_arr[:, ip, it] = s_v2
            vh_v2_arr[..., ip, it] = vh_v2
    print(time.time() - t)

    '''calculate angles'''
    k_arr = vh_v2_arr[0, ...] / np.linalg.norm(vh_v2_arr[0, ...], axis=0)
    db_arr = vh_v2_arr[2, ...] / np.linalg.norm(vh_v2_arr[2, ...], axis=0)

    '''calculate polarizations'''
    ellipticity = s_v2_arr[1, ...] / s_v2_arr[2, ...]
    sense_polar = np.real(2 * np.imag(sxy_fac) / (sxx_fac + syy_fac))
    th_k_b0 = np.rad2deg(np.arccos(np.abs(np.sum(k_arr * ez_arr, axis=0))))
    th_db_b0 = np.rad2deg(np.arccos(np.abs(np.sum(db_arr * ez_arr, axis=0))))
    planarity = 1 - np.sqrt(s_v2_arr[0, ...] / s_v2_arr[1, ...])
    degree_polar = s_v2_arr[2] / (s_v2_arr[0] + s_v2_arr[1] + s_v2_arr[2])

    '''save into a dictionary'''
    svd_dict = Munch()
    svd_dict.epoch = wt_dict.epoch
    svd_dict.period = wt_dict.period
    svd_dict.th_k_b0 = th_k_b0
    svd_dict.th_db_b0 = th_db_b0
    svd_dict.ellipticity = ellipticity
    svd_dict.sense_polar = sense_polar
    svd_dict.planarity = planarity
    svd_dict.degree_polar = degree_polar

    return svd_dict


def em_svd(b_dict, e_dict):
    """magnetic field only SVD method"""
    num_t = b_dict.epoch.size
    num_p = b_dict.period.size

    '''initialize parameters'''
    c_speed = 299792458.0
    i_up, j_up, k_up, l_up = 3, 3, 3, 6
    epsilon = np.zeros((3, 3, 3), dtype=float)
    epsilon[0, 1, 2] = +1
    epsilon[1, 2, 0] = +1
    epsilon[2, 0, 1] = +1
    epsilon[2, 1, 0] = -1
    epsilon[0, 2, 1] = -1
    epsilon[1, 0, 2] = -1

    '''initialize arrays'''
    eig_val_em_arr = np.zeros((3, num_p, num_t))
    eig_vect_em_arr = np.zeros((3, 3, num_p, num_t))

    zeta_arr = np.zeros((6, num_p, num_t), dtype=complex)
    q_arr = np.zeros((6, 6, num_p, num_t), dtype=complex)
    inv_w_em_arr = np.zeros((num_p, num_t, 3, 3))
    u_em_arr = np.zeros((num_p, num_t, 36, 3))
    v_em_arr = np.zeros((num_p, num_t, 3, 3))

    """create zeta vector"""
    zeta_arr[0, ...] = c_speed * b_dict.x_wt
    zeta_arr[1, ...] = c_speed * b_dict.y_wt
    zeta_arr[2, ...] = c_speed * b_dict.z_wt
    zeta_arr[3, ...] = e_dict.x_wt
    zeta_arr[4, ...] = e_dict.y_wt
    zeta_arr[5, ...] = e_dict.z_wt

    """create q matrix"""
    '''first row'''
    q_arr[0, 0, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[0, ...])
    q_arr[0, 1, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[1, ...])
    q_arr[0, 2, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[2, ...])
    q_arr[0, 3, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[3, ...])
    q_arr[0, 4, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[4, ...])
    q_arr[0, 5, ...] = zeta_arr[0, ...] * np.conj(zeta_arr[5, ...])
    '''second row'''
    q_arr[1, 0, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[0, ...])
    q_arr[1, 1, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[1, ...])
    q_arr[1, 2, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[2, ...])
    q_arr[1, 3, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[3, ...])
    q_arr[1, 4, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[4, ...])
    q_arr[1, 5, ...] = zeta_arr[1, ...] * np.conj(zeta_arr[5, ...])
    '''third row'''
    q_arr[2, 0, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[0, ...])
    q_arr[2, 1, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[1, ...])
    q_arr[2, 2, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[2, ...])
    q_arr[2, 3, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[3, ...])
    q_arr[2, 4, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[4, ...])
    q_arr[2, 5, ...] = zeta_arr[2, ...] * np.conj(zeta_arr[5, ...])
    '''fourth row'''
    q_arr[3, 0, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[0, ...])
    q_arr[3, 1, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[1, ...])
    q_arr[3, 2, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[2, ...])
    q_arr[3, 3, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[3, ...])
    q_arr[3, 4, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[4, ...])
    q_arr[3, 5, ...] = zeta_arr[3, ...] * np.conj(zeta_arr[5, ...])
    '''fifth row'''
    q_arr[4, 0, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[0, ...])
    q_arr[4, 1, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[1, ...])
    q_arr[4, 2, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[2, ...])
    q_arr[4, 3, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[3, ...])
    q_arr[4, 4, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[4, ...])
    q_arr[4, 5, ...] = zeta_arr[4, ...] * np.conj(zeta_arr[5, ...])
    '''sixth row'''
    q_arr[5, 0, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[0, ...])
    q_arr[5, 1, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[1, ...])
    q_arr[5, 2, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[2, ...])
    q_arr[5, 3, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[3, ...])
    q_arr[5, 4, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[4, ...])
    q_arr[5, 5, ...] = zeta_arr[5, ...] * np.conj(zeta_arr[5, ...])

    """a b complex matrix"""
    a_complex_arr = np.zeros((num_p, num_t, i_up * l_up, 3), dtype=complex)
    b_complex_arr = np.zeros((num_p, num_t, i_up * l_up), dtype=complex)

    for ii in range(i_up):
        for ll in range(l_up):
            il = ii * l_up + ll
            for jj in range(j_up):
                for kk in range(k_up):
                    a_complex_arr[..., il, jj] += epsilon[ii, jj, kk] * q_arr[3 + kk, ll, ...]
            b_complex_arr[..., il] = q_arr[ii, ll, ...]

    """a b real matrix"""
    a_real_arr = np.zeros((num_p, num_t, i_up * l_up * 2, 3))
    b_real_arr = np.zeros((num_p, num_t, i_up * l_up * 2, 1))

    a_real_arr[..., :i_up * l_up, :] = np.real(a_complex_arr)
    a_real_arr[..., (i_up * l_up): (i_up * l_up) * 2, :] = np.imag(a_complex_arr)
    b_real_arr[..., :(i_up * l_up), 0] = np.real(b_complex_arr)
    b_real_arr[..., (i_up * l_up): (i_up * l_up) * 2, 0] = np.imag(b_complex_arr)

    """initialize"""
    t = time.time()
    for it in range(num_t):
        for ip in range(num_p):
            u_em, w_em, v_em_t = scipy.linalg.svd(a_real_arr[ip, it, ...], full_matrices=False, lapack_driver='gesvd')
            ascend_order = np.argsort(w_em)
            eig_val_em_arr[:, ip, it] = w_em[ascend_order]
            eig_vect_em_arr[..., ip, it] = v_em_t[ascend_order, :]
            inv_w_em_arr[ip, it, ...] = np.diag(1 / w_em)
            u_em_arr[ip, it, ...] = u_em
            v_em_arr[ip, it, ...] = v_em_t.T
    print(time.time() - t)

    n_arr = v_em_arr @ inv_w_em_arr @ (np.transpose(u_em_arr, axes=[0, 1, 3, 2]) @ b_real_arr)
    # n_lst = v_em.T @ inv_w_em_arr @ (u_em.T @ b_real_arr)
    k_over_omega_arr = np.squeeze(n_arr) * (1.e-9 / 1.e-3) / c_speed  # unit: s/m

    '''save into a dictionary'''
    em_svd_dict = Munch()
    em_svd_dict.epoch = b_dict.epoch
    em_svd_dict.period = b_dict.period
    em_svd_dict.eig_val_em = eig_val_em_arr
    em_svd_dict.eig_vect_em = eig_vect_em_arr
    em_svd_dict.k2omega = np.transpose(k_over_omega_arr, axes=[2, 0, 1])

    return em_svd_dict


def cal_phase_k_omega_em_svd(svd_dict, v_bg_df):
    """calculate phase velocity in the plasma frame"""
    num_t = svd_dict.epoch.size
    num_p = svd_dict.period.size

    '''Calculate angular frequency in rad/s'''
    omega_sc_arr = np.tile(2 * np.pi / svd_dict.period[:, None], (1, num_t))

    '''extend background velocity 1d array to 2d array'''
    vx_mps_arr = np.tile(v_bg_df.vpr_mom, (num_p, 1)) * 1.e3
    vy_mps_arr = np.tile(v_bg_df.vpt_mom, (num_p, 1)) * 1.e3
    vz_mps_arr = np.tile(v_bg_df.vpn_mom, (num_p, 1)) * 1.e3

    '''Extract kx, ky, kz over omega in plasma frame (k/omega_pl)'''
    k2omega_x = svd_dict.k2omega[0, :, :]
    k2omega_y = svd_dict.k2omega[1, :, :]
    k2omega_z = svd_dict.k2omega[2, :, :]

    '''Calculate Omega_Flow (rad/s)'''
    omega_flow = (vx_mps_arr * k2omega_x + vy_mps_arr * k2omega_y + vz_mps_arr * k2omega_z)

    '''Calculate Omega_in_FLowFrame_arr'''
    omega_pl_arr = omega_sc_arr / (1.0 + omega_flow)

    '''Calculate kx, ky, kz (1/m)'''
    kx_arr = k2omega_x * omega_pl_arr
    ky_arr = k2omega_y * omega_pl_arr
    kz_arr = k2omega_z * omega_pl_arr
    absk_arr = np.sqrt(kx_arr ** 2 + ky_arr ** 2 + kz_arr ** 2)

    '''Calculate phase speed in the flow frame'''
    vphx_mps_arr = omega_pl_arr * kx_arr / absk_arr ** 2
    vphy_mps_arr = omega_pl_arr * ky_arr / absk_arr ** 2
    vphz_mps_arr = omega_pl_arr * kz_arr / absk_arr ** 2

    '''Convert phase-speed to kmps and add to the svd dictionary'''
    vph_pl = np.zeros((3, num_p, num_t))
    vph_pl[0, :, :] = vphx_mps_arr * 1e-3  # km
    vph_pl[1, :, :] = vphy_mps_arr * 1e-3
    vph_pl[2, :, :] = vphz_mps_arr * 1e-3
    k_pl = np.zeros((3, num_p, num_t))
    k_pl[0, :, :] = kx_arr * 1e3  # 1/km
    k_pl[1, :, :] = ky_arr * 1e3
    k_pl[2, :, :] = kz_arr * 1e3

    svd_dict.vph_pl = vph_pl
    svd_dict.k_pl = k_pl
    svd_dict.k_pl_origin = k_pl
    svd_dict.omega_pl = omega_pl_arr

    return svd_dict

def cal_phase_k_omega_em_svd_in_flow_frame(svd_dict, v_bg_df):
    """calculate phase velocity in the plasma frame"""
    num_t = svd_dict.epoch.size
    num_p = svd_dict.period.size

    '''Calculate angular frequency in rad/s'''
    omega_sc_arr = np.tile(2 * np.pi / svd_dict.period[:, None], (1, num_t))

    '''extend background velocity 1d array to 2d array'''
    vx_mps_arr = np.tile(v_bg_df.vpr_mom, (num_p, 1)) * 1.e3
    vy_mps_arr = np.tile(v_bg_df.vpt_mom, (num_p, 1)) * 1.e3
    vz_mps_arr = np.tile(v_bg_df.vpn_mom, (num_p, 1)) * 1.e3

    '''Extract kx, ky, kz over omega in plasma frame (k/omega_pl)'''
    k2omega_x = svd_dict.k2omega[0, :, :]
    k2omega_y = svd_dict.k2omega[1, :, :]
    k2omega_z = svd_dict.k2omega[2, :, :]

    '''Calculate Omega_Flow (rad/s)'''
    #omega_flow = (vx_mps_arr * k2omega_x + vy_mps_arr * k2omega_y + vz_mps_arr * k2omega_z)

    '''Calculate Omega_in_FLowFrame_arr'''
    omega_pl_arr = omega_sc_arr 

    '''Calculate kx, ky, kz (1/m)'''
    kx_arr = k2omega_x * omega_pl_arr
    ky_arr = k2omega_y * omega_pl_arr
    kz_arr = k2omega_z * omega_pl_arr
    absk_arr = np.sqrt(kx_arr ** 2 + ky_arr ** 2 + kz_arr ** 2)

    '''Calculate phase speed in the flow frame'''
    vphx_mps_arr = omega_pl_arr * kx_arr / absk_arr ** 2
    vphy_mps_arr = omega_pl_arr * ky_arr / absk_arr ** 2
    vphz_mps_arr = omega_pl_arr * kz_arr / absk_arr ** 2

    '''Convert phase-speed to kmps and add to the svd dictionary'''
    vph_pl = np.zeros((3, num_p, num_t))
    vph_pl[0, :, :] = vphx_mps_arr * 1e-3  # km
    vph_pl[1, :, :] = vphy_mps_arr * 1e-3
    vph_pl[2, :, :] = vphz_mps_arr * 1e-3
    k_pl = np.zeros((3, num_p, num_t))
    k_pl[0, :, :] = kx_arr * 1e3  # 1/km
    k_pl[1, :, :] = ky_arr * 1e3
    k_pl[2, :, :] = kz_arr * 1e3

    svd_dict.vph_pl = vph_pl
    svd_dict.k_pl = k_pl
    svd_dict.k_pl_origin = k_pl
    svd_dict.omega_pl = omega_pl_arr

    return svd_dict


def cal_propagation_em_svd(svd_dict, lbg_dict):
    """calculate a lot of angles using the em-svd results"""
    num_t = svd_dict.epoch.size
    num_p = svd_dict.period.size

    '''Initialize the output arrays'''
    theta_k_db = np.zeros((num_p, num_t))
    theta_k_b0 = np.zeros((num_p, num_t))
    theta_db_b0 = np.zeros((num_p, num_t))

    '''calculate wave-number at each point'''
    absk = scipy.linalg.norm(svd_dict.k_pl, axis=0)

    '''initialize background magnetic field array'''
    lbg_arr = np.zeros((3, num_p, num_t))
    lbg_arr[0, ...] = lbg_dict.bx_lbg_arr
    lbg_arr[1, ...] = lbg_dict.by_lbg_arr
    lbg_arr[2, ...] = lbg_dict.bz_lbg_arr

    '''calculate module of background magnetic field'''
    abs_lbg = scipy.linalg.norm(lbg_arr, axis=0)

    '''calculate propagation angle'''
    numerator = svd_dict.k_pl[0, ...] * lbg_arr[0, ...] + svd_dict.k_pl[1, ...] * lbg_arr[1, ...] + \
                svd_dict.k_pl[2, ...] * lbg_arr[2, ...]
    theta_k_b0 = np.rad2deg(np.arccos(numerator / (absk * abs_lbg)))
    k_para = numerator / abs_lbg
    k_perp = np.sqrt(absk ** 2 - k_para ** 2)
    '''add angles to svd dictionary'''
    svd_dict.theta_k_b0 = theta_k_b0
    svd_dict.k_pl_para = np.abs(k_para)
    svd_dict.k_pl_perp = np.abs(k_perp)

    return svd_dict

# '''initialize arrays'''
# th_k_b0 = np.zeros((num_p, num_t))
# th_db_b0 = np.zeros((num_p, num_t))
# ellipticity = np.zeros((num_p, num_t))
# sense_polar = np.zeros((num_p, num_t))
# planarity = np.zeros((num_p, num_t))
# degree_polar = np.zeros((num_p, num_t))
# t = time.time()
# for it in range(num_t):
#     for ip in range(num_p):
#         '''construct matrix'''
#         a = np.zeros((6, 3), dtype='float64')
#         a[0, 0] = np.real(wt_dict.x_wt[ip, it] * np.conj(wt_dict.x_wt[ip, it]))
#         a[0, 1] = np.real(wt_dict.x_wt[ip, it] * np.conj(wt_dict.y_wt[ip, it]))
#         a[0, 2] = np.real(wt_dict.x_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[1, 0] = np.real(wt_dict.x_wt[ip, it] * np.conj(wt_dict.y_wt[ip, it]))
#         a[1, 1] = np.real(wt_dict.y_wt[ip, it] * np.conj(wt_dict.y_wt[ip, it]))
#         a[1, 2] = np.real(wt_dict.y_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[2, 0] = np.real(wt_dict.x_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[2, 1] = np.real(wt_dict.y_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[2, 2] = np.real(wt_dict.z_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[3, 0] = 0.0
#         a[3, 1] = -np.imag(wt_dict.x_wt[ip, it] * np.conj(wt_dict.y_wt[ip, it]))
#         a[3, 2] = -np.imag(wt_dict.x_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[4, 0] = +np.imag(wt_dict.x_wt[ip, it] * np.conj(wt_dict.y_wt[ip, it]))
#         a[4, 1] = 0.0
#         a[4, 2] = -np.imag(wt_dict.y_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[5, 0] = +np.imag(wt_dict.x_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[5, 1] = +np.imag(wt_dict.y_wt[ip, it] * np.conj(wt_dict.z_wt[ip, it]))
#         a[5, 2] = 0.0
#
#         '''calculation'''
#         u, s, vh = scipy.linalg.svd(a)
#         ascend_order = np.argsort(s)
#         s_v2 = s[ascend_order]
#         vh_v2 = vh[s.argsort()]
#
#         '''project to fac coordinates'''
#         b0_vect = [lbg_dict.bx_lbg_arr[ip, it], lbg_dict.by_lbg_arr[ip, it], lbg_dict.bz_lbg_arr[ip, it]]
#         ez = b0_vect / np.linalg.norm(b0_vect)
#         ex = np.array([1., 0., 0.])
#         ep1 = np.cross(ez, ex)
#         ep1 = ep1 / scipy.linalg.norm(ep1)
#         ep2 = np.cross(ez, ep1)
#         ep2 = ep2 / scipy.linalg.norm(ep2)
#         bx_fac_wt = wt_dict.x_wt[ip, it] * ep1[0] + wt_dict.y_wt[ip, it] * ep1[1] + wt_dict.z_wt[ip, it] * ep1[2]
#         by_fac_wt = wt_dict.x_wt[ip, it] * ep2[0] + wt_dict.y_wt[ip, it] * ep2[1] + wt_dict.z_wt[ip, it] * ep2[2]
#
#         '''energy matrix'''
#         sxx_fac = bx_fac_wt * np.conj(bx_fac_wt)
#         sxy_fac = bx_fac_wt * np.conj(by_fac_wt)
#         syy_fac = by_fac_wt * np.conj(by_fac_wt)
#
#         '''calculate angles'''
#         k = vh_v2[0, :] / np.linalg.norm(vh_v2[0, :])
#         db = vh_v2[2, :] / np.linalg.norm(vh_v2[2, :])
#
#         '''calculate polarizations'''
#         ellipticity[ip, it] = s_v2[1] / s_v2[2]
#         sense_polar[ip, it] = np.real(2 * np.imag(sxy_fac) / (sxx_fac + syy_fac))
#         th_k_b0[ip, it] = np.rad2deg(np.arccos(np.abs(np.dot(k, ez))))
#         th_db_b0[ip, it] = np.rad2deg(np.arccos(np.abs(np.dot(db, ez))))
#         planarity[ip, it] = 1 - np.sqrt(s_v2[0] / s_v2[1])
#         degree_polar[ip, it] = s_v2[2] / (s_v2[0] + s_v2[1] + s_v2[2])
