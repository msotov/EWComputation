"""
EW_computation.py

Module to compute the equivalent width (EW) of a list of absorption lines,
from high-resolution echelle spectra.
The lines are fitted with Gaussian-shaped profiles,
and then the EWs are estimated based on the line fit.

Usage:

from EW_computation import EW_calc

EW_calc('sun', wave, flux)

"""
from __future__ import print_function
from __future__ import division

__author__ = "Maritza Soto (marisotov@gmail.com)"
__date__ = "2019-08-08"

from builtins import zip
from builtins import range
import os
import pickle
import logging
import warnings
import sys
import matplotlib
matplotlib.use('TkAgg')
from scipy import interpolate
from astropy.convolution import convolve, Box1DKernel
from astropy.io import ascii as astropyascii
from pyspeckit import Spectrum
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.pyplot as plt

warnings.simplefilter("ignore")

########################################################


def EW_calc(
        starname,
        wave,
        flux,
        linelist='linelist.dat',
        snr=100.,
        makeplot=True,
        path_plots='./EW/plots_EW',
        from_dic=False,
        save_line_data=False):
    """
    Main function of the module. It receives the wavelength
    and flux arrays, as well as the linelist to use,
    and return the EW for all the lines in linelist,
    as long as the result is a valid number (not nan or inf),
    and the line is visible in the spectrum.
    """

    print('\t\tCreating EW file for %s' % starname)

    if from_dic is False or os.path.isfile(
            'EW/%s_line_data.pkl' %
            starname) is False:
        norders = wave.ndim
        lines = astropyascii.read('Spectra/%s' % linelist, comment='-')['WL']
        final_eqws = np.zeros(lines.size)
        final_eqws_mean = np.zeros(lines.size)
        final_eqws_err1 = np.zeros(lines.size)
        final_eqws_err2 = np.zeros(lines.size)

        dic = {}
        space = 3.0
        if norders == 1:
            logging.info('Only one order detected')

        for i, l in enumerate(lines):
            # Select a window of 3 AA around the line
            if norders is 1:
                iline = np.where((l - space < wave) & (wave < l + space))[0]
                wave_l = wave[iline]
                flux_l = flux[iline]
                rejt = 1. - (1. / snr)
                snr_o = snr
                if wave_l.size == 0:
                    logging.warning('Line not found in spectrum.')
                    continue

            else:
                iline_coords = np.where(
                    (l - space < wave) & (wave < l + space))
                # Check is the line is present in more than one order
                if np.unique(iline_coords[0]).size > 1:
                    # Check that the order's limits are within the line
                    ix = np.unique(iline_coords[0])
                    order_limits = np.zeros(ix.size)
                    for x_i, xorder in enumerate(ix):
                        wave_nonzero = wave[xorder][np.nonzero(wave[xorder])]
                        if wave_nonzero[0] < (l - space):
                            order_limits[x_i] += 1
                        if wave_nonzero[-1] > (l + space):
                            order_limits[x_i] += 1

                    if len(np.where(order_limits == max(order_limits))[0]) > 1:
                        # Choose the redder order
                        order_mean = np.zeros(ix.size)
                        for x_i, xorder in enumerate(ix):
                            wave_nonzero = wave[xorder][np.nonzero(
                                wave[xorder])]
                            order_mean[x_i] = np.mean(wave_nonzero)
                        iorder = ix[np.where(order_mean == max(order_mean))[0]]

                    else:
                        iorder = ix[np.where(
                            order_limits == max(order_limits))[0]]

                    x_order = iline_coords[0][np.where(
                        iline_coords[0] == iorder)[0]]
                    y_order = iline_coords[1][np.where(
                        iline_coords[0] == iorder)[0]]

                    if hasattr(snr, "__len__"):
                        rejt = 1. - (1. / snr[iorder])
                        snr_o = snr[iorder]
                    else:
                        rejt = 1. - (1. / snr)
                        snr_o = snr

                    wave_l = wave[x_order, y_order]
                    flux_l = flux[x_order, y_order]

                else:
                    wave_l = wave[iline_coords]
                    flux_l = flux[iline_coords]

                    if wave_l.size == 0:
                        logging.warning('Line not found in spectrum')
                        continue

                    if hasattr(snr, "__len__"):
                        iorder = np.unique(iline_coords[0])
                        rejt = 1. - (1. / snr[iorder])
                        snr_o = snr[iorder]
                    else:
                        rejt = 1. - (1. / snr)
                        snr_o = snr

                del iline_coords

            l, final_eqw, final_eqw_mean, final_eqw_err1, final_eqw_err2, dicl = EW_for_line(
                l, wave_l, flux_l, rejt, snr_o)
            if final_eqw != 0.0:
                final_eqws[i] = final_eqw
                final_eqws_mean[i] = final_eqw_mean
                final_eqws_err1[i] = final_eqw_err1
                final_eqws_err2[i] = final_eqw_err2
                dic['%f' % l] = dicl

            del dicl, wave_l, flux_l

        astropyascii.write([lines,
                            final_eqws,
                            final_eqws_mean,
                            final_eqws_err1,
                            final_eqws_err2],
                           'EW/%s.txt' % starname,
                           format='fixed_width_no_header',
                           delimiter=' ',
                           overwrite=True)

        if save_line_data:
            fl = open('EW/%s_line_data.pkl' % starname, 'wb')
            pickle.dump(dic, fl)
            fl.close()
            del fl
        del lines, final_eqws, final_eqws_mean, final_eqws_err1, final_eqws_err2

    else:
        print('\t\tReading EW information from EW/%s_line_data.pkl' % starname)
        dic = pickle.load(open('EW/%s_line_data.pkl' % starname, 'rb'))

    if makeplot:
        plot_lines(starname, dic, path_plots)

    del dic


def error_ew(sp):
    """
    Estimate the uncertainty in the EW by changing the
    parameters of the Gaussian-shaped fit within their
    uncertainties
    """

    values = sp.specfit.parinfo.values
    errors = sp.specfit.parinfo.errors
    par0 = np.random.normal(values[0], errors[0], 1000)
    par1 = np.random.normal(values[1], errors[1], 1000)
    par2 = np.random.normal(values[2], errors[2], 1000)
    del values, errors

    i = np.where(par2 > 0.0)[0]

    par0 = par0[i]
    par1 = par1[i]
    par2 = par2[i]
    parvalues = list(zip(par0, par1, par2))

    EW = np.zeros(i.size)
    xarr = sp.xarr.value
    cdelt = np.mean(xarr[1:] - xarr[:-1])

    for k in range(i.size):
        EW[k] = -par0[k] * np.exp(-(xarr - par1[k]) **
                                  2. / (2. * par2[k]**2.)).sum() * cdelt * 1000.

    del par0, par1, par2, parvalues, i, xarr

    inonan = np.where((~np.isnan(EW)) & (~np.isinf(EW)))[0]
    dist_ranges = np.percentile(EW[inonan], [16, 50, 84])

    return dist_ranges, EW[inonan]


def normalize(x, y, rejt):
    """ Substracts the continuum and normalizes """

    p = np.poly1d(np.polyfit(x, y, 3))
    ynorm = p(x)

    for _ in range(5):
        dif = np.hstack((np.abs((y[:-1] - y[1:]) / y[:-1]), [1.0]))
        i = np.where(((y - ynorm * rejt) > 0) & (dif < 0.1))[0]
        vecx = x[i]
        vecy = y[i]
        p = np.poly1d(np.polyfit(vecx, vecy, 3))
        ynorm = p(x)

    yfit = p(x)
    del p

    if np.any(yfit <= 0.0):
        i_nonzero = np.where(y > (y.min() + 0.05 * (y.max() - y.min())))[0]
        xmed = np.mean(x)
        i_in_line = np.where((x >= (xmed - 1.0)) & (x <= (xmed + 1.0)))[0]

        if np.any(np.isin(i_in_line, i_nonzero)):
            logging.info('line not visible')
            yfit = np.ones(y.size) * np.nan

        del i_nonzero, i_in_line

    return np.array(y / yfit) - 1.0, vecx, vecy


def detect_lines(wave, flux):
    """ Try to detect lines using derivates (from ARES) """

    ysmooth = convolve(flux, Box1DKernel(4))
    for n in range(3):
        gradient = np.gradient(ysmooth)
        ysmooth = convolve(gradient, Box1DKernel(4))
        if n == 1:
            tckn = interpolate.UnivariateSpline(wave, ysmooth, k=5, s=0)

    tck = interpolate.UnivariateSpline(wave, ysmooth, k=3, s=0)
    zeros = tck.roots()
    flux_zeros = interpolate.UnivariateSpline(wave, flux, k=5, s=0)(zeros)
    ifinal = np.where((tckn(zeros) > 0.0) & (flux_zeros < -0.04))[0]

    del ysmooth, gradient, tckn, tck, flux_zeros

    return zeros[ifinal]


def get_exc_list(lines, width=0.15):
    """ Get the regions surrounding the lines that won't
    be considered in the fit"""

    e_list = []
    if lines.size == 0:
        return e_list
    e_list_l = [[li - width, li + width] for li in lines]
    e1, e2 = lines[0] - width, lines[0] - width
    for ii in range(len(e_list_l) - 1):
        if e_list_l[ii][0] > e2:
            e1 = e_list_l[ii][0]
        if e_list_l[ii][1] > e2:
            e2 = e_list_l[ii][1]
        if e2 < e_list_l[ii + 1][0]:
            e_list.append(e1)
            e_list.append(e2)
    if e_list_l[-1][0] > e2:
        e1 = e_list_l[-1][0]
    e2 = e_list_l[-1][1]
    e_list.append(e1)
    e_list.append(e2)

    del e_list_l
    return e_list

def check_data(l, x):
    """ Check that the line is not at the edges of the window"""

    x_right = np.where(x <= l)[0]
    x_left = np.where(x >= l)[0]
    if x_right.size < 5 or x_left.size < 5:
        return False
    return True


########################################################
########################################################
########################################################


def EW_for_line(l, wave_l, flux_l, rejt, snr_o):
    """
    Function that computes the EW (with its uncertainty)
    for a single line.
    """

    dicl = {}
    try:
        # Check that the line is visible
        if check_data(l, wave_l) is False:
            logging.error('Line %.2f is not visible in data. Skipping.', l)
            return l, 0.0, 0.0, 0.0, 0.0, dicl

        # Normalize the continuum
        flux_l_norm, vecx, vecy = normalize(wave_l, flux_l, rejt)
        if np.all(np.isnan(flux_l_norm)) or np.any(np.isinf(flux_l_norm)):
            return l, 0.0, 0.0, 0.0, 0.0, dicl

        dicl['wave'] = wave_l
        dicl['flux'] = flux_l_norm
        dicl['rejt'] = rejt
        dicl['vecx'] = vecx
        dicl['vecy'] = vecy
        dicl['flux_no_norm'] = flux_l

        del vecx, vecy

        # Detect lines and the regions surrounding it
        lines_in_data = detect_lines(wave_l, flux_l_norm)
        dicl['lines'] = lines_in_data
        exc_list = get_exc_list(lines_in_data, width=0.15)

        # Create the spectrum object, with the normalized data
        sp = Spectrum(
            xarr=wave_l,
            data=flux_l_norm + 1.0,
            xarrkwargs={
                'unit': 'angstrom'},
            header={},
            debug=False,
            interactive=False,
            verbose=False,
            fit_plotted_area=False)
        sp.xarr.xtype = 'wavelength'

        # Compute the continuum
        sp.baseline(exclude=exc_list, order=0, subtract=False, annotate=False,
                    debug=False, interactive=False,
                    reset_selection=False)
        sp.baseline.set_basespec_frompars(baselinepars=[1.0])

        i_no_l = np.where(np.abs(lines_in_data - float(l)) > 0.1)[0]
        lines_no_l = np.sort(lines_in_data[i_no_l])
        dicl['lines_no_l'] = lines_no_l
        exc_list_fit = get_exc_list(lines_no_l, width=0.15)
        del i_no_l, lines_no_l

        fluxmin = min(flux_l_norm[np.where(np.abs(wave_l - l) <= 0.15)[0]])

        sp.specfit(guesses=[fluxmin, l, 0.02],
                   fittype='gaussian',
                   limits=[(fluxmin - 0.03, fluxmin + 0.03), (l - 0.1, l + 0.1), (0, 0)],
                   limited=[(True, True), (True, True), (True, False)],
                   exclude=exc_list_fit, annotate=False, interactive=False,
                   xmin=wave_l[0], xmax=wave_l[-1], verbose=False,
                   fit_plotted_area=False)

        sp.specfit(guesses=sp.specfit.parinfo.values,
                   fittype='gaussian',
                   exclude=exc_list_fit, annotate=False, interactive=False,
                   xmin=wave_l[0], xmax=wave_l[-1], fit_plotted_area=False,
                   verbose=False)

        amp = sp.specfit.parinfo.values[0]
        mean = sp.specfit.parinfo.values[1]
        sigma = sp.specfit.parinfo.values[2]

        # If the fit has the correct parameters:
        if (amp <= 0.0) and (np.abs(mean - l) <= 0.10) and (sigma < 0.10) and\
                np.all(np.array(sp.specfit.parinfo.errors) < 0.3):

            # Compute the EW for the fit
            try:
                EW = sp.specfit.EQW(
                    fitted=True,
                    components=False,
                    xunits='angstrom',
                    xmin=l - 3.0,
                    xmax=l + 3.0,
                    annotate=False)
            except ValueError:
                EW = sp.specfit.EQW(
                    fitted=True,
                    components=False,
                    xunits='angstrom',
                    xmin=l - 0.2,
                    xmax=l + 0.2,
                    annotate=False)

            final_eqws = EW * 1000.
            parvalues = sp.specfit.parinfo.values
            parerrors = sp.specfit.parinfo.errors

        else:
            # Modify the portions of the spectra we have to exclude from the
            # fit
            logging.warning('First fit with the wrong parameters')

            # If the only line present in the spectrum is the one we're trying
            # to fit
            if not exc_list or len(exc_list) == 2:
                logging.info('Only one line was detected')
                exc_list_fit = [wave_l[0], l - 0.5, l + 0.5, wave_l[-1]]

            else:
                # If the line it too shallow
                if fluxmin > -0.03:
                    exc_list_fit = exc_list
                    exc_list_fit.append(wave_l[0])
                    exc_list_fit.append(l - 0.2)
                    exc_list_fit.append(l + 0.2)
                    exc_list_fit.append(wave_l[-1])
                    logging.info(
                        'Line is too shalow. New exc_list has length %d',
                        len(exc_list_fit))

            if np.where(np.abs(lines_in_data - float(l)) <= 0.10)[0].size == 0:
                lines_in_data = np.append(lines_in_data, float(l))
                lines_in_data = np.array(sorted(lines_in_data))
                dicl['lines'] = lines_in_data

            # Only consider the lines close to the data
            lines_close = lines_in_data[np.where(
                np.abs(lines_in_data - float(l)) < 1.0)[0]]
            lines_far = lines_in_data[np.where(
                np.abs(lines_in_data - float(l)) >= 1.0)[0]]
            exc_list_fit = get_exc_list(lines_far, width=0.20)

            tck_lines = interpolate.UnivariateSpline(
                wave_l, flux_l_norm, k=5, s=0)
            fluxlines = tck_lines(lines_close)
            del tck_lines
            guesses = []
            for il, ll in enumerate(lines_close):
                guesses.append(fluxlines[il])
                guesses.append(ll)
                guesses.append(0.02)

            sp.specfit(guesses=guesses,
                       fittype='gaussian',
                       annotate=False, interactive=False,
                       xmin=wave_l[0], xmax=wave_l[-1], verbose=False,
                       fit_plotted_area=False,
                       exclude=exc_list_fit)

            parinfo = sp.specfit.parinfo
            lg = len(guesses)

            parlines = np.array([parinfo.values[i:i + 3]
                                 for i in range(lg) if i % 3 == 0])
            parerror = np.array([parinfo.errors[i:i + 3]
                                 for i in range(lg) if i % 3 == 0])
            iline = np.argmin(np.abs(parlines.T[1] - l))

            dicl['parlines'] = parlines

            amp = parlines[iline][0]
            mean = parlines[iline][1]
            sigma = parlines[iline][2]
            logging.info(
                'New values are: l=%.2f, a=%.2f, mu=%.1f, s=%.3f',
                l,
                amp,
                mean,
                sigma)
            del guesses, lines_close, lines_far, parinfo, lg

            # If the fit parameters are correct
            if (amp <= 0.0) and (np.abs(mean - l) <= 0.15) and (sigma < 0.15) and\
                np.all(np.array(parerror[iline]) < 0.3):
                logging.info('Correct fit values were found')

                parvalues = parlines[iline]
                parerrors = parerror[iline]

                # Compute the EW using the line fit
                try:
                    EW = sp.specfit.EQW(
                        fitted=True,
                        components=True,
                        xunits='angstrom',
                        xmin=l - 3.0,
                        xmax=l + 3.0,
                        annotate=False)[iline]
                except ValueError:
                    EW = sp.specfit.EQW(
                        fitted=True,
                        components=True,
                        xunits='angstrom',
                        xmin=l - 0.2,
                        xmax=l + 0.2,
                        annotate=False)[iline]

                final_eqws = EW * 1000.
            else:
                logging.error('Couldnt find correct line parameters')
                final_eqws = 0.0

            del parlines, parerror

        if final_eqws == 0.0:
            del sp, exc_list, exc_list_fit, wave_l, flux_l, flux_l_norm
            return l, 0.0, 0.0, 0.0, 0.0, dicl

        # Compute the error in the EW
        e_ranges, EW_dist = error_ew(sp)
        final_eqws_mean = e_ranges[1]
        final_eqws_err1 = e_ranges[2] - e_ranges[1]
        final_eqws_err2 = e_ranges[1] - e_ranges[0]

        print(
            '\t\t\tLine {:6.2f}: a={: 3.2f}, mu={:5.1f}, s={:4.3f}, EW={:6.2f}, rejt={:4.3f}, '
            'snr={:5.2f}'.format(
                l,
                *
                parvalues,
                final_eqws,
                rejt,
                snr_o))
        logging.info(
            '{:6.2f}: a={: 3.2f}, mu={:5.1f}, s={:4.3f}, EW={:6.2f}, rejt={:4.3f}, '
            'snr={:5.2f}'.format(
                l,
                *parvalues,
                final_eqws,
                rejt,
                snr_o))

        dicl['exc_list'] = exc_list
        dicl['exc_list_fit'] = exc_list_fit
        dicl['EW'] = final_eqws
        dicl['snr'] = snr_o
        dicl['EW_dist'] = EW_dist
        dicl['parvalues'] = parvalues
        dicl['parerrors'] = parerrors

        del sp, exc_list, exc_list_fit, flux_l_norm, EW_dist, e_ranges

        if ~np.isnan(final_eqws) and ~np.isinf(final_eqws):
            return l, final_eqws, final_eqws_mean, final_eqws_err1, final_eqws_err2, dicl

    except Exception as e:
        logging.error('Problem with line %.2f. Ignoring.', l)
        _, _, exc_tb = sys.exc_info()
        logging.error('line %d: %s', exc_tb.tb_lineno, e)
        return l, 0.0, 0.0, 0.0, 0.0, dicl



def plot_full_model(x, values):
    """ Returns the model produced by one or several Gaussian profiles """

    nlines = len(values)
    yfit = np.zeros(x.size)
    for l in range(nlines):
        yfit += values[l][0] * \
            np.exp(-(x - values[l][1])**2 / (2.0 * values[l][2]**2))
    return yfit


def plot_fit_distribution(wave, parvalues, parerrors):
    """ Returns the confidence levels of the Gaussian fit,
    based on the uncertainties of the fit parameters."""

    def fit_for_par(w, p):
        return p[0] * np.exp(-(w - p[1])**2 / (2.0 * p[2]**2))

    n = 1000
    y = np.zeros((n, wave.size))
    y16 = np.zeros(wave.size)
    y50 = np.zeros(wave.size)
    y84 = np.zeros(wave.size)
    a_dist = np.random.normal(parvalues[0], parerrors[0], n)
    m_dist = np.random.normal(parvalues[1], parerrors[1], n)
    s_dist = np.random.normal(parvalues[2], parerrors[2], n)
    for i in range(n):
        y[i] = fit_for_par(wave, [a_dist[i], m_dist[i], s_dist[i]])
    yT = y.T
    for i in range(wave.size):
        y16[i], y50[i], y84[i] = np.percentile(yT[i], [16, 50, 84])

    del yT, y, a_dist, m_dist, s_dist
    return y16, y50, y84


def plot_lines(starname, dic, path_plots, ncols=6):
    """
    Makes a plot of all the lines fitted, including the
    continuum estimation, the Gaussian-shaped fit and
    its uncertainty, the other absorption lines
    detected, and the EW for each line.
    It also plots the distribution of EWs based on
    the fit parameters.
    """

    plt.style.use(['seaborn-muted'])
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    class PlotGrid:
        """Plot object, formed by N-individual axes, with N the number of lines"""

        def __init__(self, ncols, nrows, path_plots='.', islines=True):
            self.ncols = ncols
            self.nrows = nrows
            if islines:
                self.fig = plt.figure(figsize=(6 * 6, 2 * nrows))
            else:
                self.fig = plt.figure(figsize=(3 * 6, 2 * nrows))
            self.grid = GridSpec(nrows, ncols, figure=self.fig)
            self.coordinates = np.arange(ncols * nrows).reshape((nrows, ncols))
            self.islines = islines
            self.path_plots = path_plots

        def __call__(self, i):
            irow, icol = np.where(self.coordinates == i)
            if self.islines:
                gs0 = GridSpecFromSubplotSpec(
                    2, 2, subplot_spec=self.grid[irow, icol], hspace=0)
                axi1 = plt.Subplot(self.fig, gs0[0, :])
                self.fig.add_subplot(axi1)
                axi = plt.Subplot(self.fig, gs0[1, :])
                self.fig.add_subplot(axi)
                return axi1, axi, irow, icol

            ax_err = self.fig.add_subplot(self.nrows, self.ncols, i + 1)
            return ax_err, irow, icol

        def finish_plot(self, starname):
            """Saves and closes the figures"""
            if self.islines:
                self.fig.subplots_adjust(bottom=1. /
                                         (self.nrows *
                                          4), top=1 -
                                         1. /
                                         (self.nrows *
                                          6), left=0.015, right=0.99, wspace=0.1)
                self.fig.savefig(
                    os.path.join(
                        os.path.relpath(
                            self.path_plots,
                            '.'),
                        '%s' %
                        starname),
                    format='pdf')
            else:
                self.fig.subplots_adjust(
                    bottom=0.02, top=0.99, left=0.05, right=0.98, wspace=0.2)
                self.fig.savefig(
                    os.path.join(
                        os.path.relpath(
                            self.path_plots,
                            '.'),
                        '%s_error' %
                        starname),
                    format='pdf')
            plt.close('all')
            del self.fig

    try:
        lines = list(dic.keys())
        nrows = int(np.ceil(len(lines) / 6.))

        LinesPlot = PlotGrid(ncols, nrows, path_plots=path_plots, islines=True)
        ErrorPlot = PlotGrid(
            ncols,
            nrows,
            path_plots=path_plots,
            islines=False)

        for i, l in enumerate(lines):
            d = dic[l]
            ax_cont, ax, irow, icol = LinesPlot(i)

            ####### Plot the fits to the lines ########
            wave_l = d['wave']
            flux_l = d['flux'] + 1.0
            parvalues = d['parvalues']
            parerrors = d['parerrors']

            # Continuum
            vecx = d['vecx']
            vecy = d['vecy']
            flux_no_norm = d['flux_no_norm']
            ax_cont.plot(wave_l, flux_no_norm, lw=0.5, color='black')
            p = np.poly1d(np.polyfit(vecx, vecy, 3))
            xcont = np.linspace(wave_l[0], wave_l[-1], 100)
            ycont = p(xcont)
            ax_cont.plot(xcont, ycont, lw=0.5, color='steelblue')
            ax_cont.plot(
                vecx,
                vecy,
                ls='None',
                marker='x',
                color='orangered',
                markersize=1.2)
            del vecx, vecy, p, xcont, ycont

            # Line

            ax.plot(wave_l, flux_l, lw=0.4, color='black')
            ax.axvline(float(l), color='red', ls='-', lw=0.5, alpha=0.5)

            xfit = np.hstack((np.linspace(min(wave_l), float(l) - 1.0, 50),
                              np.linspace(float(l) - 1.0, float(l) + 1.0, 200),
                              np.linspace(float(l) + 1.0, max(wave_l), 50)))
            yfit = parvalues[0] * \
                np.exp(-(xfit - parvalues[1])**2 / (2.0 * parvalues[2]**2))
            ax.plot(xfit, yfit + 1.0, color='steelblue', lw=0.5)

            if 'parlines' in d:
                yfull = plot_full_model(xfit, d['parlines'])
                ax.plot(xfit, yfull + 1.0, color='green', lw=0.5)
                del yfull

            exc_list = d['exc_list_fit']
            for j, e in enumerate(exc_list):
                if (j % 2) == 0:
                    ax.axvspan(exc_list[j],
                               exc_list[j + 1],
                               facecolor='magenta',
                               alpha=0.2,
                               edgecolor='white',
                               lw=0.1)

            yfit16, yfit50, yfit84 = plot_fit_distribution(
                xfit, parvalues, parerrors)
            ax.fill_between(
                xfit,
                yfit16 + 1.0,
                y2=yfit50 + 1.0,
                alpha=0.4,
                edgecolor='white',
                lw=0.1,
                color='orangered')
            ax.fill_between(
                xfit,
                yfit50 + 1.0,
                y2=yfit84 + 1.0,
                alpha=0.4,
                edgecolor='white',
                lw=0.1,
                color='orangered')
            ax.plot(xfit, yfit50 + 1.0, color='red', lw=0.3)

            del xfit, yfit, yfit16, yfit50, yfit84, exc_list

            ax.axhline(1.0, ls='-', color='gray')

            # Lines found
            ll = [
                ax.axvline(
                    l,
                    color='gray',
                    ls='-',
                    alpha=0.5,
                    lw=0.5) for l in d['lines']]
            if 'lines_no_l' in d:
                ll = [
                    ax.axvline(
                        l,
                        color='gray',
                        ls='-',
                        alpha=0.5,
                        lw=0.5) for l in d['lines_no_l']]
            del ll

            # Set limits, label sizes, number of ticks, and text to be added

            sx, ex = min(wave_l), max(wave_l)
            sy, ey = ax_cont.get_ylim()
            ax_cont.set_xlim(sx, ex)
            ax_cont.xaxis.set_ticks(np.arange(np.ceil(sx), ex, 1))
            if ex - sx > 6.0:
                ax_cont.xaxis.set_ticks(np.arange(np.ceil(sx), ex, 2))
            ax_cont.yaxis.set_ticks(
                np.linspace(
                    min(flux_no_norm), max(flux_no_norm), 5)[1:])
            ax_cont.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax_cont.tick_params(labelsize='x-small', labelbottom=False)
            del flux_no_norm

            sy, ey = ax.get_ylim()
            ew = d['EW'] / 1000.
            ew16, ew50, ew84 = np.percentile(
                d['EW_dist'], [16, 50, 84]) / 1000.
            midpt = float(l)
            ax.fill_between([midpt - ew50 / 2.0, midpt + ew50 / 2.0], [0, 0],
                            [1.0, 1.0], color='g', alpha=0.3, edgecolor='white', lw=0.1)
            ax.text(sx + (ex - sx) / 20.,
                    sy + (ey - sy) / 7.,
                    r'EW$\,=\,%.1f^{+%.1f}_{-%.1f}\,$mA' % (ew50 * 1000.,
                                                            (ew84 - ew50) * 1000.,
                                                            (ew50 - ew16) * 1000.),
                    fontsize='x-small',
                    bbox=dict(facecolor='white',
                              alpha=0.8,
                              edgecolor='None',
                              linestyle='None'))

            ax.text(ex - (ex - sx) / 4,
                    sy + (ey - sy) / 6.,
                    r'$A\,=\,%.3f\,\pm\,%.3f$'
                    '\n'
                    r'$\mu\,=\,%.1f\,\pm\,%.3f$'
                    '\n'
                    r'$\sigma\,=\,%.3f\,\pm\,%.3f$' % (parvalues[0],
                                                       parerrors[0],
                                                       parvalues[1],
                                                       parerrors[1],
                                                       parvalues[2],
                                                       parerrors[2]),
                    fontsize='x-small',
                    bbox=dict(facecolor='white',
                              alpha=0.8,
                              edgecolor='None',
                              linestyle='None'))

            ax.set_ylim(max(sy, 0.0), min(1.30, ey))
            ax.set_xlim(sx, ex)

            ax.xaxis.set_ticks(np.arange(np.ceil(sx), ex, 1))
            ax.xaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%d"))

            sy, ey = ax.get_ylim()
            ax.yaxis.set_ticks(np.linspace(sy, ey, 6))
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_ylim(sy, ey)

            ax.tick_params(labelsize='x-small')
            if icol == 0:
                ax.set_ylabel('Flux', fontsize='small')
            else:
                ax.set_ylabel(' ')
            if irow == nrows - 1:
                ax.set_xlabel('Wavelength', fontsize='small')
            elif (irow == nrows - 2) and (icol >= (ncols - ((ncols * nrows) % len(lines)))):
                ax.set_xlabel('Wavelength', fontsize='small')
            else:
                ax.set_xlabel(' ')

            ####### Plot the error distributions ########

            ax_err, irow, icol = ErrorPlot(i)

            if ~np.all(np.isnan(d['EW_dist'])):
                inonan = np.where(~np.isnan(d['EW_dist']))[0]
                ax_err.hist(
                    d['EW_dist'][inonan],
                    bins=30,
                    histtype='stepfilled',
                    color='steelblue')
                del inonan
            ax_err.axvline(ew16 * 1000., color='orange')
            ax_err.axvline(ew50 * 1000., color='orange')
            ax_err.axvline(ew84 * 1000., color='orange')

            ax_err.tick_params(labelsize='small')

            start_x, end_x = ax_err.get_xlim()
            ax_err.xaxis.set_ticks(np.linspace(start_x, end_x, 5))
            ax_err.axvline(ew * 1000., color='orangered')

            del wave_l, flux_l, d, parvalues, parerrors, ax_cont, ax, ax_err

        LinesPlot.finish_plot(starname)
        ErrorPlot.finish_plot(starname)

        # plt.close('all')
        del LinesPlot, ErrorPlot

    except Exception as e:
        logging.error('Error while plotting:')
        _, _, exc_tb = sys.exc_info()
        logging.error('line %d: %s', exc_tb.tb_lineno, e)
        logging.error('Skipping...')
