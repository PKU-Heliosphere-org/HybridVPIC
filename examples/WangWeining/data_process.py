import numpy as np
import pandas as pd
import pycwt as wavelet
import julian
import matplotlib
from munch import Munch
from scipy.optimize import least_squares
from itertools import chain
from utilities.python_utils_JSHEPT import get_local_mean_variable as get_bg
from utilities.python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect
from utilities.python_utils_JSHEPT import get_plot_WaveletAnalysis_of_var_vect
from utilities.python_utils_JSHEPT import get_local_mean_variable
from utilities.python_utils_JSHEPT import dE_from_FaradayLaw
from utilities.python_utils_JSHEPT import wavelet_reconstruction
from utilities.python_utils_JSHEPT import get_PhaseAngle_and_PoyntingFlux_from_E_and_B
from utilities.python_utils_JSHEPT import get_PoyntingFlux_r_arr
from utilities.python_utils_JSHEPT import get_AlfvenSpeed
from utilities.python_utils_JSHEPT import get_ComplexOmega2k_arr

coeff_for_VxB = 1.e-3
from_VxB_in_kmpsxnT_to_E_in_mVpm = 1.e-3
from_E2B_in_mVpm2nT_to_V_in_mps = 1.e+6

def interp_and_smooth(df, t_win='0.02S', smooth=False, t_smooth=None):
    if t_smooth is None:
        t_smooth = t_win
    idx = pd.date_range(start=df.index[0], end=df.index[-1], freq=t_win)
    df_new = df.reindex(df.index.union(idx)).interpolate('time').loc[idx]
    if smooth:
        df_new = df_new.rolling(t_smooth, center=True).mean()
    return df_new


def interp_with_index(df, idx=1):
    return df.reindex(df.index.union(idx)).interpolate('time', limit_direction="both").loc[idx]


def cal_b_wavelet(index, bx, by, bz, s0=1.0, s1=100.0, num_periods=32):
    """calculate wavelet coefficients"""
    julian_lst = np.array([julian.to_jd(x) for x in index])
    time_vect = (julian_lst - julian_lst[0]) * (24. * 60. * 60)
    mwt = wavelet.Morlet(6)
    dt = index.to_series().diff().iloc[1].total_seconds()
    periods = np.logspace(np.log10(s0), np.log10(s1), num_periods)
    freqs = 1 / periods
    bx_wave, scales, freqs, coi, bx_fft, fftfreqs = wavelet.cwt(bx.values, dt, wavelet=mwt, freqs=freqs)
    by_wave, scales, freqs, coi, by_fft, fftfreqs = wavelet.cwt(by.values, dt, wavelet=mwt, freqs=freqs)
    bz_wave, scales, freqs, coi, bz_fft, fftfreqs = wavelet.cwt(bz.values, dt, wavelet=mwt, freqs=freqs)

    # period_range = [periods[0], periods[-1]]
    # var_vect = bx.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # bx_wave = WaveletCoeff_var_arr.T
    #
    # var_vect = by.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # by_wave = WaveletCoeff_var_arr.T
    #
    # var_vect = bz.values
    # time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = \
    #     get_plot_WaveletAnalysis_of_var_vect(time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    # bz_wave = WaveletCoeff_var_arr.T
    #
    # freqs = 1 / period_vect

    '''calculate background mean magnetic field and derive Bpara and Bperp'''
    var_vect = bx.values
    #width2period = 10.0
    width2period = 30.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    bx_lbg_arr = var_lbg_arr.T
    var_vect = by.values
    width2period = 30.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    by_lbg_arr = var_lbg_arr.T
    var_vect = bz.values
    width2period = 30.0
    time_vect, period_vect, var_lbg_arr = get_bg(time_vect, var_vect, period_range=(s0, s1),
                                                 num_periods=num_periods,
                                                 width2period=width2period)
    bz_lbg_arr = var_lbg_arr.T

    '''calculate unit direction vector '''
    babs_lbg_arr = np.sqrt(bx_lbg_arr ** 2 + by_lbg_arr ** 2 + bz_lbg_arr ** 2)
    ebx_lbg_arr = bx_lbg_arr / babs_lbg_arr
    eby_lbg_arr = by_lbg_arr / babs_lbg_arr
    ebz_lbg_arr = bz_lbg_arr / babs_lbg_arr

    '''calculate parallel wavelet'''
    bpara_wave = bx_wave * ebx_lbg_arr + by_wave * eby_lbg_arr + bz_wave * ebz_lbg_arr

    '''create wavelet dictionary '''
    wt_dict = Munch()
    wt_dict.epoch = index.to_pydatetime()
    wt_dict.x_wt = bx_wave
    wt_dict.y_wt = by_wave
    wt_dict.z_wt = bz_wave
    wt_dict.para_wt = bpara_wave
    wt_dict.freq = freqs
    wt_dict.period = periods
    wt_dict.coi = coi

    '''create local background magnetic field dictionary'''
    lbg_dict = Munch()
    lbg_dict.epoch = index.to_pydatetime()
    lbg_dict.freq = freqs
    lbg_dict.period = periods
    lbg_dict.bx_lbg_arr = bx_lbg_arr
    lbg_dict.by_lbg_arr = by_lbg_arr
    lbg_dict.bz_lbg_arr = bz_lbg_arr

    return wt_dict, lbg_dict


def cal_wavelet(index, bx, by, bz, s0=1.0, s1=100.0, num_periods=32):
    """calculate wavelet coefficients"""
    mwt = wavelet.Morlet(6)
    dt = index.to_series().diff().iloc[1].total_seconds()
    periods = np.logspace(np.log10(s0), np.log10(s1), num_periods)
    freqs = 1 / periods
    bx_wave, scales, freqs, coi, bx_fft, fftfreqs = wavelet.cwt(bx.values, dt, wavelet=mwt, freqs=freqs)
    by_wave, scales, freqs, coi, by_fft, fftfreqs = wavelet.cwt(by.values, dt, wavelet=mwt, freqs=freqs)
    bz_wave, scales, freqs, coi, bz_fft, fftfreqs = wavelet.cwt(bz.values, dt, wavelet=mwt, freqs=freqs)

    '''create dictionary for wavelets'''
    wt_dict = Munch()
    wt_dict.epoch = index.to_pydatetime()
    wt_dict.x_wt = bx_wave
    wt_dict.y_wt = by_wave
    wt_dict.z_wt = bz_wave
    wt_dict.freq = freqs
    wt_dict.period = periods
    wt_dict.coi = coi

    return wt_dict


def cal_para_wavelet(wt_dict, lbg_dict):
    """calculate parallel wavelet coefficients"""

    '''calculate unit direction vector '''
    babs_lbg_arr = np.sqrt(lbg_dict.bx_lbg_arr ** 2 + lbg_dict.by_lbg_arr ** 2 + lbg_dict.bz_lbg_arr ** 2)
    ebx_lbg_arr = lbg_dict.bx_lbg_arr / babs_lbg_arr
    eby_lbg_arr = lbg_dict.by_lbg_arr / babs_lbg_arr
    ebz_lbg_arr = lbg_dict.bz_lbg_arr / babs_lbg_arr

    wt = wt_dict.x_wt * ebx_lbg_arr + wt_dict.y_wt * eby_lbg_arr + wt_dict.z_wt * ebz_lbg_arr
    wt_perp = np.sqrt(np.abs(wt_dict.x_wt) ** 2 + np.abs(wt_dict.y_wt) ** 2 + np.abs(wt_dict.z_wt) ** 2 - np.abs(wt) ** 2)

    '''add to wavelet dictionary'''
    wt_dict.para_wt = wt
    wt_dict.perp_wt = wt_perp
    wt_dict.trace_wt = np.sqrt(np.abs(wt_dict.x_wt) ** 2 + np.abs(wt_dict.y_wt) ** 2 + np.abs(wt_dict.z_wt) ** 2)

    return wt_dict


def cal_psd(wt_dict):
    """calculate wavelet psd"""
    dt = np.diff(wt_dict.epoch)[0].total_seconds()
    x_psd_arr = np.abs(wt_dict.x_wt) ** 2 * 2 * dt
    y_psd_arr = np.abs(wt_dict.y_wt) ** 2 * 2 * dt
    z_psd_arr = np.abs(wt_dict.z_wt) ** 2 * 2 * dt
    trace_psd_arr = x_psd_arr + y_psd_arr + z_psd_arr
    para_psd_arr = np.abs(wt_dict.para_wt) ** 2 * 2 * dt
    perp_psd_arr = 0.5 * (trace_psd_arr - para_psd_arr)

    '''calculate 1d wavelet spectra'''
    x_psd_lst = np.mean(x_psd_arr, axis=1)
    y_psd_lst = np.mean(y_psd_arr, axis=1)
    z_psd_lst = np.mean(z_psd_arr, axis=1)
    para_psd_lst = np.mean(para_psd_arr, axis=1)
    perp_psd_lst = np.mean(perp_psd_arr, axis=1)
    trace_psd_lst = np.mean(trace_psd_arr, axis=1)

    x_lst = np.mean(np.abs(wt_dict.x_wt), axis=1)
    y_lst = np.mean(np.abs(wt_dict.y_wt), axis=1)
    z_lst = np.mean(np.abs(wt_dict.z_wt), axis=1)
    para_lst = np.mean(np.abs(wt_dict.para_wt), axis=1)
    trace_wt = np.sqrt(trace_psd_arr)
    trace_lst = np.mean(trace_wt, axis=1)
    perp_wt = np.sqrt(trace_wt**2 - np.abs(wt_dict.para_wt)**2)
    perp_lst = np.mean(perp_wt, axis=1)

    '''create psd dictionary'''
    psd_dict = Munch()
    psd_dict.epoch = wt_dict.epoch
    psd_dict.freq = wt_dict.freq
    psd_dict.period = wt_dict.period
    psd_dict.x_psd_arr = x_psd_arr
    psd_dict.y_psd_arr = y_psd_arr
    psd_dict.z_psd_arr = z_psd_arr
    psd_dict.para_psd_arr = para_psd_arr
    psd_dict.perp_psd_arr = perp_psd_arr
    psd_dict.trace_psd_arr = trace_psd_arr
    psd_dict.x_psd_lst = x_psd_lst
    psd_dict.y_psd_lst = y_psd_lst
    psd_dict.z_psd_lst = z_psd_lst
    psd_dict.para_psd_lst = para_psd_lst
    psd_dict.perp_psd_lst = perp_psd_lst
    psd_dict.trace_psd_lst = trace_psd_lst
    psd_dict.x_lst = x_lst
    psd_dict.y_lst = y_lst
    psd_dict.z_lst = z_lst
    psd_dict.trace_lst = trace_lst
    psd_dict.para_lst = para_lst
    psd_dict.perp_lst = perp_lst


    return psd_dict


def cal_helicity(epoch, period, wt_1, wt_2):
    """This is equivalent to the calculation of sense polarization"""
    helicity = np.real(2 * np.imag(wt_1 * np.conj(wt_2)) / (wt_1 * np.conj(wt_1) + wt_2 * np.conj(wt_2)))

    '''save into a dictionary'''
    hel_dict = Munch()
    hel_dict.epoch = epoch
    hel_dict.period = period
    hel_dict.helicity = helicity

    return hel_dict


def cal_bg_velocity(df, t_win='1000S'):
    v_bg_df = df.rolling(t_win, center=True).mean()
    return v_bg_df


def cal_plasma_frame_efield(e_df, b_df, v_bg_df):
    """calculate electric field in plasma frame with pre-interpolated data"""
    '''please use rtn data'''
    er_conv = -(v_bg_df.vpt_mom * b_df.bn - v_bg_df.vpn_mom * b_df.bt) * 1.e-3
    et_conv = -(v_bg_df.vpn_mom * b_df.br - v_bg_df.vpr_mom * b_df.bn) * 1.e-3
    en_conv = -(v_bg_df.vpr_mom * b_df.bt - v_bg_df.vpt_mom * b_df.br) * 1.e-3

    '''calculation'''
    e_df['er_pl'] = e_df.er - er_conv
    e_df['et_pl'] = e_df.et - et_conv 
    e_df['en_pl'] = e_df.en - en_conv

    return e_df

def calc_anisotropic_temp(T_tensor, mag, rotmat):
     T_para = []
     T_perp = []
     for i in range(len(T_tensor[:, 0])):
         T_arr = np.array(
             [[T_tensor[i, 0], T_tensor[i, 3], T_tensor[i, 4]], [T_tensor[i, 3], T_tensor[i, 1], T_tensor[i, 5]],
              [T_tensor[i, 4], T_tensor[i, 5], T_tensor[i, 2]]])
         mag_tmp = mag[:, i].reshape(-1,1)
         mag_tmp = np.dot(rotmat, mag_tmp).reshape(1,-1)
         # print(mag_tmp)
         # print('T_arr: ',T_arr)
         e1_mag = mag_tmp / np.linalg.norm(mag_tmp)
         e2_mag = np.cross(e1_mag, [1, 0, 0])
         e2_mag = e2_mag / np.linalg.norm(e2_mag)
         e3_mag = np.cross(e1_mag, e2_mag)
         E_mag = np.asmatrix(np.array([[e1_mag], [e2_mag], [e3_mag]]).T)
         T_mag = E_mag.T * np.asmatrix(T_arr) * E_mag
         # print(T_mag)
         T_para.append(T_mag[0, 0])
         T_perp.append((T_mag[1, 1] + T_mag[2, 2]) / 2)
     T_para = np.array(T_para)
     T_perp = np.array(T_perp)
     return T_para, T_perp


def cal_gamma_not_svd(mag_df_int, ele_df_int, mom_df_int, beg_t, end_t):
    JulDay_B_vect = mag_df_int.index.to_julian_date()
    Br_vect = mag_df_int.br
    Bt_vect = mag_df_int.bt
    Bn_vect = mag_df_int.bn
    Bmag_vect = np.sqrt(Br_vect ** 2 + Bt_vect ** 2 + Bn_vect ** 2)
    date_B_vect = matplotlib.dates.julian2num(JulDay_B_vect)

    JulDay_E_vect = ele_df_int.index.to_julian_date()
    Et_vect = ele_df_int.et
    En_vect = ele_df_int.en
    date_E_vect = matplotlib.dates.julian2num(JulDay_E_vect)

    JulDay_ion_vect = mom_df_int.index.to_julian_date()
    Np_vect = mom_df_int.np_mom
    Np_vect = np.reshape(Np_vect, len(Np_vect))
    Vr_vect = mom_df_int.vpr_mom
    Vt_vect = mom_df_int.vpt_mom
    Vn_vect = mom_df_int.vpn_mom
    date_ion_vect = matplotlib.dates.julian2num(JulDay_ion_vect)

    plot_details = 0

    '''slice data'''
    _beg_t = beg_t.strftime('%H%M%S')
    _end_t = end_t.strftime('%H%M%S')
    mag_df_int_temp = mag_df_int[beg_t:end_t]
    ele_df_int_temp = ele_df_int[beg_t:end_t]

    #<editor-fold desc="define 'datetime_beg/end', and get 'julday_beg/end'">
    ##define and set the datetime_beg & datetime_end for analysis and plot
    datetime_beg = beg_t
    datetime_end = end_t
    julday_beg = julian.to_jd(datetime_beg)
    julday_end = julian.to_jd(datetime_end)

    # print(julday_beg,julday_end, np.max(JulDay_vect), np.min(JulDay_vect))
    ###get 'TimeInterval_str' & 'date_str' to be as parts of the figure file names
    year_str = "{0:04d}".format(datetime_beg.year)
    month_str = "{0:02d}".format(datetime_beg.month)
    day_str = "{0:02d}".format(datetime_beg.day)
    hour_beg_str = "{0:02d}".format(datetime_beg.hour)
    minute_beg_str = "{0:02d}".format(datetime_beg.minute)
    second_beg_str = "{0:02d}".format(datetime_beg.second)
    hour_end_str = "{0:02d}".format(datetime_end.hour)
    minute_end_str = "{0:02d}".format(datetime_end.minute)
    second_end_str = "{0:02d}".format(datetime_end.second)
    TimeInterval_str = '(hhmmss=' + hour_beg_str + minute_beg_str + second_beg_str + '-' + \
                    hour_end_str + minute_end_str + second_end_str + ')'
    date_str = '(ymd=' + year_str + month_str + day_str + ')'
    title_datetime_str = hour_beg_str + ':' + minute_beg_str + ':' + second_beg_str + '-' + \
                        hour_end_str + ':' + minute_end_str + ':' + second_end_str + ' on ' + \
                        year_str + '-' + month_str + '-' + day_str
    print('TimeInterval_str: ', TimeInterval_str)
    print('date_str: ', date_str)
    #input('Press any key to continue...')
    #</editor-fold>

    #<editor-fold desc="get 'Julday_B/E/E_fit/ion_vect_interval', 'Br/Bt/Bn/Et/En/Np/Vr/Vt/Vn_vect_interval'">
    sub_B_interval = (JulDay_B_vect >= julday_beg) & (JulDay_B_vect <= julday_end)
    JulDay_B_vect_interval = JulDay_B_vect[sub_B_interval]
    Br_vect_interval = Br_vect[sub_B_interval]
    Bt_vect_interval = Bt_vect[sub_B_interval]
    Bn_vect_interval = Bn_vect[sub_B_interval]
    print('len(JulDay_B_vect_interval):', len(JulDay_B_vect_interval))
    del JulDay_B_vect, date_B_vect, Br_vect, Bt_vect, Bn_vect

    sub_E_interval = (JulDay_E_vect >= julday_beg) & (JulDay_E_vect <= julday_end)
    JulDay_E_vect_interval = JulDay_E_vect[sub_E_interval]
    Et_vect_interval = Et_vect[sub_E_interval]
    En_vect_interval = En_vect[sub_E_interval]
    print('len(JulDay_E_vect_interval):', len(JulDay_E_vect_interval))
    del JulDay_E_vect, date_E_vect, Et_vect, En_vect


    sub_ion_interval = (JulDay_ion_vect >= julday_beg) & (JulDay_ion_vect <= julday_end)
    JulDay_ion_vect_interval = JulDay_ion_vect[sub_ion_interval]
    Np_vect_interval = Np_vect[sub_ion_interval]
    Vr_vect_interval = Vr_vect[sub_ion_interval]
    Vt_vect_interval = Vt_vect[sub_ion_interval]
    Vn_vect_interval = Vn_vect[sub_ion_interval]
    print('len(JulDay_ion_vect_interval):', len(JulDay_ion_vect_interval), len(Np_vect_interval))
    del JulDay_ion_vect, date_ion_vect, Np_vect, Vr_vect, Vt_vect, Vn_vect
    #</editor-fold>

    #<editor-fold desc="get 'julday/datenum_vect_common', 'Br/Bt/Bn/Et/En/Np/Vr/Vt/Vn_vect_interval'.">
    dtime_common = mag_df_int.index.to_series().diff().mean().total_seconds()  # unit: s

    num_times_common = int((julday_end - julday_beg) / (dtime_common / (24. * 60 * 60))) + 1
    # a print('num_times_common: ', num_times_common)
    julday_vect_common = np.linspace(julday_beg, julday_end, num_times_common)
    datenum_vect_common = matplotlib.dates.julian2num(julday_vect_common)
    # a print('np.shape(julday_vect_common, JulDay_B_vect_interval, Br_vect_interval, Np_vect_interval):  ')
    # a print(np.shape(julday_vect_common), np.shape(JulDay_B_vect_interval), np.shape(Br_vect_interval), np.shape(Np_vect_interval))

    Br_vect_interval = np.interp(julday_vect_common, JulDay_B_vect_interval, Br_vect_interval)
    Bt_vect_interval = np.interp(julday_vect_common, JulDay_B_vect_interval, Bt_vect_interval)
    Bn_vect_interval = np.interp(julday_vect_common, JulDay_B_vect_interval, Bn_vect_interval)
    Et_vect_interval = np.interp(julday_vect_common, JulDay_E_vect_interval, Et_vect_interval)
    En_vect_interval = np.interp(julday_vect_common, JulDay_E_vect_interval, En_vect_interval)
    Np_vect_interval = np.interp(julday_vect_common, JulDay_ion_vect_interval, Np_vect_interval)
    Vr_vect_interval = np.interp(julday_vect_common, JulDay_ion_vect_interval, Vr_vect_interval)
    Vt_vect_interval = np.interp(julday_vect_common, JulDay_ion_vect_interval, Vt_vect_interval)
    Vn_vect_interval = np.interp(julday_vect_common, JulDay_ion_vect_interval, Vn_vect_interval)
    #</editor-fold>

    #<editor-fold desc="get 'Et/En_in_SW_vect_interval'.">
    ##get Et/En_in_SW_vect_interval
    Vr_mean = np.mean(Vr_vect_interval)
    Vt_mean = np.mean(Vt_vect_interval)
    Vn_mean = np.mean(Vn_vect_interval)
    Et_in_SW_vect_interval = Et_vect_interval + \
                            ((-Vr_mean * Bn_vect_interval) + (
                                        +Vn_mean * Br_vect_interval)) * from_VxB_in_kmpsxnT_to_E_in_mVpm
    En_in_SW_vect_interval = En_vect_interval + \
                            ((+Vr_mean * Bt_vect_interval) + (
                                        -Vt_mean * Br_vect_interval)) * from_VxB_in_kmpsxnT_to_E_in_mVpm
    #</editor-fold>

    time_vect = (julday_vect_common - julday_vect_common[0]) * (24. * 60. * 60)
    period_range = [0.1, 5]  # unit: s
    num_periods = 64
    #<editor-fold desc="get 'WaveletCoeff_Br/Bt/Bn_arr', 'SubWave_Br/Bt/Bn_arr', 'sigma_m_arr'.">
    ##get 'WaveletCoeff_Br_arr' & 'SubWave_Br_arr'
    var_vect = Br_vect_interval
    time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    WaveletCoeff_Br_arr = WaveletCoeff_var_arr
    SubWave_Br_arr = sub_wave_var_arr
    ##get 'WaveletCoeff_Bt_arr' & 'SubWave_Bt_arr'
    var_vect = Bt_vect_interval
    time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    WaveletCoeff_Bt_arr = WaveletCoeff_var_arr
    SubWave_Bt_arr = sub_wave_var_arr
    ##get 'WaveletCoeff_Bn_arr' & 'SubWave_Bn_arr'
    var_vect = Bn_vect_interval
    time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    WaveletCoeff_Bn_arr = WaveletCoeff_var_arr
    SubWave_Bn_arr = sub_wave_var_arr
    ##get magnetic helicity
    sigma_m_arr = 2 * np.imag(WaveletCoeff_Bt_arr * np.conj(WaveletCoeff_Bn_arr)) / \
                (np.abs(WaveletCoeff_Bt_arr) ** 2 + np.abs(WaveletCoeff_Bn_arr) ** 2)
    #</editor-fold>

    #<editor-fold desc="get 'WaveletCoeff_Et_arr' & 'SubWave_Et_arr' in the spacecraft (SC) reference frame.">
    ##get 'WaveletCoeff_Et_arr' & 'SubWave_Et_arr' in the spacecraft (SC) reference frame
    var_vect = Et_vect_interval
    time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    wavelet_obj_Et_arr = wavelet_obj_arr
    WaveletCoeff_Et_arr = WaveletCoeff_var_arr
    SubWave_Et_arr = sub_wave_var_arr
    print('scales, periods: ', wavelet_obj_Et_arr.scales, wavelet_obj_Et_arr.fourier_periods)
    #input('Press any key to continue...')
    ##get 'WaveletCoeff_En_arr' & 'SubWave_En_arr' in the spacecraft (SC) reference frame
    var_vect = En_vect_interval 
    time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods)
    wavelet_obj_En_arr = wavelet_obj_arr
    WaveletCoeff_En_arr = WaveletCoeff_var_arr
    SubWave_En_arr = sub_wave_var_arr
    #</editor-fold>

    #<editor-fold desc="get 'Br/Bt/Bn/Vr/Vt/Vn_LocalBG_arr'.">
    ##get 'Br_LocalBG_arr'
    var_vect = Br_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Br_LocalBG_arr = var_LocalBG_arr

    ##get 'Bt_LocalBG_arr'
    var_vect = Bt_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Bt_LocalBG_arr = var_LocalBG_arr

    ##get 'Bn_LocalBG_arr'
    var_vect = Bn_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Bn_LocalBG_arr = var_LocalBG_arr

    ##get 'Vr_LocalBG_arr'
    var_vect = Vr_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Vr_LocalBG_arr = var_LocalBG_arr

    ##get 'Vt_LocalBG_arr'
    var_vect = Vt_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Vt_LocalBG_arr = var_LocalBG_arr

    ##get 'Vn_LocalBG_arr'
    var_vect = Vn_vect_interval
    width2period = 10.0
    time_vect, period_vect, var_LocalBG_arr = get_local_mean_variable( \
        time_vect, var_vect, period_range=period_range, num_periods=num_periods, width2period=width2period)
    Vn_LocalBG_arr = var_LocalBG_arr
    #</editor-fold>

    #<editor-fold desc="get 'lambda_arr' (=Vsw_LocalBG_arr*period_arr).">
    ##get 'lambda_arr' (=Vsw_LocalBG_arr*period_arr)
    AbsV_LocalBG_arr = np.sqrt(Vr_LocalBG_arr ** 2 + Vt_LocalBG_arr ** 2 + Vn_LocalBG_arr ** 2)
    x_vect = datenum_vect_common
    y_vect = period_vect
    x_arr, y_arr = np.meshgrid(x_vect, y_vect)
    period_arr = np.transpose(y_arr)
    lambda_arr = AbsV_LocalBG_arr * period_arr
    lambda_vect = np.mean(lambda_arr, axis=0)  # unit: km
    #</editor-fold>

    #<editor-fold desc="get 'WaveletCoeff_Et/En_in_SW_arr', 'SubWave_Et/En_in_SW_arr'.">
    is_LocalBG_or_GlobalBG_flow = 1
    is_LocalBG_or_GlobalBG_flow_str = "{0:01d}".format(is_LocalBG_or_GlobalBG_flow)
    sub_file_str_for_BG_flow = '(is_LocalBG_or_GlobalBG_flow=' + is_LocalBG_or_GlobalBG_flow_str + ')'
    print('is_LocalBG_or_GlobalBG_flow: ', is_LocalBG_or_GlobalBG_flow)
    #input('Press any key to continue...')
    if (is_LocalBG_or_GlobalBG_flow == 1):
        ##get 'WaveletCoeff_Et/En_in_SW_arr', the wavelet coefficients for electric field fluctuations in the solar wind reference frame
        WaveletCoeff_Et_in_SW_arr = WaveletCoeff_Et_arr + \
                                    (
                                                -Vr_LocalBG_arr * WaveletCoeff_Bn_arr + Vn_LocalBG_arr * WaveletCoeff_Br_arr) * from_VxB_in_kmpsxnT_to_E_in_mVpm
        WaveletCoeff_En_in_SW_arr = WaveletCoeff_En_arr + \
                                    (
                                                +Vr_LocalBG_arr * WaveletCoeff_Bt_arr - Vt_LocalBG_arr * WaveletCoeff_Br_arr) * from_VxB_in_kmpsxnT_to_E_in_mVpm
        ##get 'SubWave_Et_in_SW_arr' through reconstruction based on
        ##the object 'wavelet_obj_Et_arr' & 'WaveletCoeff_Et_in_SW_arr' (the e-field wavelet coefficient in the solar wind reference frame
        wavelet_obj_Et_arr.fourier_periods = period_vect
        wavelet_obj_arr = wavelet_obj_Et_arr
        WaveletCoeff_var_arr = WaveletCoeff_Et_in_SW_arr  # in_SW_arr
        sub_wave_var_arr = wavelet_reconstruction(wavelet_obj_arr, WaveletCoeff_var_arr)
        SubWave_Et_in_SW_arr = sub_wave_var_arr
        ##get 'SubWave_En_in_SW_arr' through reconstruction based on
        ##the object 'wavelet_obj_En_arr' & 'WaveletCoeff_En_in_SW_arr' (the e-field wavelet coefficient in the solar wind reference frame
        wavelet_obj_En_arr.fourier_periods = period_vect
        wavelet_obj_arr = wavelet_obj_En_arr
        WaveletCoeff_var_arr = WaveletCoeff_En_in_SW_arr  # in_SW_arr
        sub_wave_var_arr = wavelet_reconstruction(wavelet_obj_arr, WaveletCoeff_var_arr)
        SubWave_En_in_SW_arr = sub_wave_var_arr
    elif (is_LocalBG_or_GlobalBG_flow == 2):
        ###get scalar values 'Vr/Vt/Vn_GlobalBG_scalar'
        Vr_GlobalBG_scalar = np.mean(Vr_vect_interval)
        Vt_GlobalBG_scalar = np.mean(Vt_vect_interval)
        Vn_GlobalBG_scalar = np.mean(Vn_vect_interval)
        ###get 'Et/En_in_SW_vect'
        Et_in_SW_vect = Et_vect_interval + \
                        (
                                    -Vr_GlobalBG_scalar * Bn_vect_interval + Vn_GlobalBG_scalar * Br_vect_interval) * from_VxB_in_kmpsxnT_to_E_in_mVpm
        En_in_SW_vect = En_vect_interval + \
                        (
                                    +Vr_GlobalBG_scalar * Bt_vect_interval - Vt_GlobalBG_scalar * Br_vect_interval) * from_VxB_in_kmpsxnT_to_E_in_mVpm
        ###get 'WaveletCoeff_En/Et_in_SW_arr' & 'SubWave_Et/En_in_SW_arr'
        var_vect = Et_in_SW_vect
        time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
            time_vect, var_vect, period_range=period_range, num_periods=num_periods)
        wavelet_obj_Et_arr = wavelet_obj_arr
        WaveletCoeff_Et_in_SW_arr = WaveletCoeff_var_arr
        SubWave_Et_in_SW_arr = sub_wave_var_arr
        var_vect = En_in_SW_vect
        time_vect, period_vect, wavelet_obj_arr, WaveletCoeff_var_arr, sub_wave_var_arr = get_plot_WaveletAnalysis_of_var_vect( \
            time_vect, var_vect, period_range=period_range, num_periods=num_periods)
        wavelet_obj_En_arr = wavelet_obj_arr
        WaveletCoeff_En_in_SW_arr = WaveletCoeff_var_arr
        SubWave_En_in_SW_arr = sub_wave_var_arr
    #</editor-fold>

    #<editor-fold desc="get 'polarization_E_in_SW_arr' & 'polarization_E_in_SC_arr'.">
    ##get electric field polarization 'polarization_E_in_SW_arr' & 'polarization_E_in_SC_arr'
    polarization_E_in_SW_arr = 2 * np.imag(WaveletCoeff_Et_in_SW_arr * np.conj(WaveletCoeff_En_in_SW_arr)) / \
                            (np.abs(WaveletCoeff_Et_in_SW_arr) ** 2 + np.abs(WaveletCoeff_En_in_SW_arr) ** 2)
    polarization_E_in_SC_arr = 2 * np.imag(WaveletCoeff_Et_arr * np.conj(WaveletCoeff_En_arr)) / \
                            (np.abs(WaveletCoeff_Et_arr) ** 2 + np.abs(WaveletCoeff_En_arr) ** 2)
    #</editor-fold>

    #<editor-fold desc="get 'PhaseAngle_EtEn_in_SW_arr' & 'PhaseAngle_BtBn_arr'.">
    ###get 'PhaseAngle_EtEn_in_SW_arr' & 'PhaseAngle_BtBn_arr'
    PhaseAngle_EtEn_in_SW_arr = np.arctan2(SubWave_Et_in_SW_arr, SubWave_En_in_SW_arr) / np.pi * 180.
    PhaseAngle_BtBn_arr = np.arctan2(SubWave_Bt_arr, SubWave_Bn_arr) / np.pi * 180.
    #</editor-fold>

    #<editor-fold desc="get 'PhaseAngle_from_E_to_B_arr', "PoyntingFlux_r_arr".">
    ##get the phase angle between dE_in_SW and dB
    ##According to wave theory, ICW is growing when (ExB.e_r >0 & |phi(dE, dB)|<90) or (ExB.e_r <0 & |phi(dE, dB)|>90)
    ##According to wave theory, ICW is damping when (ExB.e_r >0 & |phi(dE, dB)|>90) or (ExB.e_r <0 & |phi(dE, dB)|<90)
    PhaseAngle_from_E_to_B_arr, PoyntingFlux_r_arr, sub_dJidE_gt_0, sub_dJidE_lt_0, sub_dJedE_gt_0, sub_dJedE_lt_0 = \
        get_PhaseAngle_and_PoyntingFlux_from_E_and_B(SubWave_Et_in_SW_arr, SubWave_En_in_SW_arr, SubWave_Bt_arr,
                                                    SubWave_Bn_arr, Br_LocalBG_arr)
    PoyntingFlux_r_arr_from_SubWaves = PoyntingFlux_r_arr
    ##get 'PoyntingFlux_r_arr'
    PoyntingFlux_r_arr = get_PoyntingFlux_r_arr(time_vect, \
                                                WaveletCoeff_Et_in_SW_arr, WaveletCoeff_En_in_SW_arr, \
                                                WaveletCoeff_Bt_arr, WaveletCoeff_Bn_arr)
    #</editor-fold>

    #<editor-fold desc="get 'omega2k/gamma2k/gamma2omega/omega/gamma_arr'.">
    ##get 'omega2k/gamma2k_time_period_arr'
    ComplexOmega2k_time_period_arr_from_dEn2dBt = -WaveletCoeff_En_in_SW_arr / WaveletCoeff_Bt_arr
    ComplexOmega2k_time_period_arr_from_dEt2dBn = +WaveletCoeff_Et_in_SW_arr / WaveletCoeff_Bn_arr
    ComplexOmega2k_time_period_arr_from_dEn2dBt *= from_E2B_in_mVpm2nT_to_V_in_mps * 1.e-3  # unit: km/s
    ComplexOmega2k_time_period_arr_from_dEt2dBn *= from_E2B_in_mVpm2nT_to_V_in_mps * 1.e-3  # unit: km/s
    ###revise 'omega2k' & 'gamma2k' to be their opposite ones at the time & period when Poynting Flux is negative
    sub_PoyntingFlux_lt_0 = (PoyntingFlux_r_arr < 0)
    ComplexOmega2k_time_period_arr_from_dEn2dBt[sub_PoyntingFlux_lt_0] *= -1.
    ComplexOmega2k_time_period_arr_from_dEt2dBn[sub_PoyntingFlux_lt_0] *= -1.
    ###get 'omega2k/gamma2k/gamma2omega'
    omega2k_arr_from_dEn2dBt = np.real(ComplexOmega2k_time_period_arr_from_dEn2dBt)
    gamma2k_arr_from_dEn2dBt = np.imag(ComplexOmega2k_time_period_arr_from_dEn2dBt)
    omega2k_arr_from_dEt2dBn = np.real(ComplexOmega2k_time_period_arr_from_dEt2dBn)
    gamma2k_arr_from_dEt2dBn = np.imag(ComplexOmega2k_time_period_arr_from_dEt2dBn)
    gamma2omega_arr_from_dEn2dBt = gamma2k_arr_from_dEn2dBt / np.abs(omega2k_arr_from_dEn2dBt)
    gamma2omega_arr_from_dEt2dBn = gamma2k_arr_from_dEt2dBn / np.abs(omega2k_arr_from_dEt2dBn)
    print('max(omega2k), min(omega2k): ', np.max(omega2k_arr_from_dEn2dBt), np.min(omega2k_arr_from_dEn2dBt))
    gamma_arr_from_dEn2dBt = gamma2k_arr_from_dEn2dBt / lambda_arr  # unit: 1/s
    gamma_arr_from_dEt2dBn = gamma2k_arr_from_dEt2dBn / lambda_arr  # unit: 1/s
    print('period_vect: ', period_vect)
    #input('Press any key to continue...')

    ##get 'ComplexOmega2k' based on 'omega/k=dExdB*/dB.dB*' under the  assumption that ek//dExdB
    ComplexOmega2k_time_period_arr = get_ComplexOmega2k_arr(time_vect, \
                                                            WaveletCoeff_Et_in_SW_arr, WaveletCoeff_En_in_SW_arr, \
                                                            WaveletCoeff_Bt_arr, WaveletCoeff_Bn_arr)
    omega2k_arr = np.real(ComplexOmega2k_time_period_arr)  # unit: km/s
    gamma2k_arr = np.imag(ComplexOmega2k_time_period_arr)
    lambda_arr_v2 = (AbsV_LocalBG_arr + omega2k_arr) * period_arr
    is_add_omega2k_to_lambda = 1
    print('is_add_omega2k_to_lambda:', is_add_omega2k_to_lambda)
    #input('Press any key to continue...')
    if (is_add_omega2k_to_lambda == 1):
        lambda_arr = lambda_arr_v2
    else:
        lambda_arr = lambda_arr
    lambda_vect = np.mean(lambda_arr, axis=0) # update 'lambda_vect'
    omega_arr = omega2k_arr / lambda_arr  # unit: 1/s
    gamma_arr = gamma2k_arr / lambda_arr  # unit: 1/s
    gamma2omega_arr = gamma2k_arr / np.abs(omega2k_arr)
    #</editor-fold>

    #<editor-foldd desc="get 'theta_BR_arr', 'PSD_Btrace/Bperp/Bpara/_arr', 'PSD_Et/En_in_SW/SC_arr'.">
    theta_BR_arr = np.arctan2(np.sqrt(Bt_LocalBG_arr ** 2 + Bn_LocalBG_arr ** 2), Br_LocalBG_arr) * 180 / np.pi
    AbsB_LocalBG_arr = np.sqrt(Br_LocalBG_arr ** 2 + Bt_LocalBG_arr ** 2 + Bn_LocalBG_arr ** 2)
    ebr_LocalBG_arr = Br_LocalBG_arr / AbsB_LocalBG_arr
    ebt_LocalBG_arr = Bt_LocalBG_arr / AbsB_LocalBG_arr
    ebn_LocalBG_arr = Bn_LocalBG_arr / AbsB_LocalBG_arr
    WaveletCoeff_Bpara_arr = WaveletCoeff_Br_arr * ebr_LocalBG_arr + \
                            WaveletCoeff_Bt_arr * ebt_LocalBG_arr + \
                            WaveletCoeff_Bn_arr * ebn_LocalBG_arr
    dtime = np.diff(time_vect).mean()
    PSD_Bpara_arr = np.abs(WaveletCoeff_Bpara_arr) ** 2 * (2 * dtime)
    PSD_Btrace_arr = (np.abs(WaveletCoeff_Br_arr) ** 2 + np.abs(WaveletCoeff_Bt_arr) ** 2 + np.abs(
        WaveletCoeff_Bn_arr) ** 2) * (2 * dtime)
    PSD_Bperp_arr = PSD_Btrace_arr - PSD_Bpara_arr
    ##
    PSD_Et_in_SC_arr = np.abs(WaveletCoeff_Et_arr) ** 2 * (2 * dtime)
    PSD_En_in_SC_arr = np.abs(WaveletCoeff_En_arr) ** 2 * (2 * dtime)
    PSD_Et_in_SW_arr = np.abs(WaveletCoeff_Et_in_SW_arr) ** 2 * (2 * dtime)
    PSD_En_in_SW_arr = np.abs(WaveletCoeff_En_in_SW_arr) ** 2 * (2 * dtime)
    PSD_Etn_in_SC_arr = PSD_Et_in_SC_arr + PSD_En_in_SC_arr
    PSD_Etn_in_SW_arr = PSD_Et_in_SW_arr + PSD_En_in_SW_arr
    #</editor-fold>

    #<editor-fold desc="get omega2k & gamma2k & gamma2omega using scipy.optimize.least_squares">
    ## get omega2k & gamma2k & gamma2omega using scipy.optimize.least_squares
    """
    (omega/k+i*gamma/k)*[dBt_complex_obs, dBn_complex_obs]^T = [-dEn_complex_obs, +dEt_complex_obs]^T
    [[dBt_complex_obs, 0],[0, dBn_compllex_obs]]x[omega/k+i*gamma/k,omega/k+i*gamma/k]^T = [-dEn_complex_obs, +dEt_compllex_obs]^T
    """
    omega2k_kmps_vect = np.zeros(num_periods)
    gamma2k_kmps_vect = np.zeros(num_periods)
    gamma2omega_vect = np.zeros(num_periods)
    for i_period in range(0, num_periods):
        dEt_complex_observe = WaveletCoeff_Et_in_SW_arr[:, i_period]
        dEn_complex_observe = WaveletCoeff_En_in_SW_arr[:, i_period]
        dBt_complex_observe = WaveletCoeff_Bt_arr[:, i_period]
        dBn_complex_observe = WaveletCoeff_Bn_arr[:, i_period]
        VA_kmps = 200.
        omega2k_gamma2k_ini = [VA_kmps, VA_kmps * 0.1]


        def residual_between_dE_obs_and_dE_from_FaradayLaw(omega2k_gamma2k):
            dEt_complex_predict, dEn_complex_predict = dE_from_FaradayLaw(omega2k_gamma2k, dBt_complex_observe,
                                                                        dBn_complex_observe)
            residual_Re_En = np.real(dEn_complex_predict - dEn_complex_observe)
            residual_Im_En = np.imag(dEn_complex_predict - dEn_complex_observe)
            residual_Re_Et = np.real(dEt_complex_predict - dEt_complex_observe)
            residual_Im_Et = np.imag(dEt_complex_predict - dEt_complex_observe)
            residual_vect_tmp = [residual_Re_En, residual_Im_En, residual_Re_Et, residual_Im_Et]
            # a print(type(residual_vect_tmp), np.shape(residual_vect_tmp))
            residual_vect = list(chain.from_iterable(residual_vect_tmp))
            # a print(np.shape(residual_vect))
            return residual_vect


        result_LeastSquares = least_squares(residual_between_dE_obs_and_dE_from_FaradayLaw, omega2k_gamma2k_ini)
        omega2k_gamma2k_fit = result_LeastSquares.x
        omega2k_mps = omega2k_gamma2k_fit[0] * from_E2B_in_mVpm2nT_to_V_in_mps
        gamma2k_mps = omega2k_gamma2k_fit[1] * from_E2B_in_mVpm2nT_to_V_in_mps
        gamma2omega = omega2k_gamma2k_fit[1] / np.abs(omega2k_gamma2k_fit[0])
        # a print('type(omega2k_mps), omega2k_mps: ',type(omega2k_mps), omega2k_mps)
        omega2k_kmps_vect[i_period] = omega2k_mps * 1.e-3
        gamma2k_kmps_vect[i_period] = gamma2k_mps * 1.e-3
        gamma2omega_vect[i_period] = gamma2omega
    gamma_s_vect = gamma2k_kmps_vect / lambda_vect
    return gamma_arr
