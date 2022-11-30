import math
import numpy as np
from matplotlib import pyplot as plt
from skrf.media import DefinedGammaZ0
from skrf import Network, Frequency, connect, innerconnect, mathFunctions
from lmfit import minimize, Parameters, Model
from pathlib import Path
from skrf_extensions import *
from aux import *


"""
Inductor class.
 - Holds the data to be fitted as Network object
 - Creates a fitted Network object
 - Contains useful functions
"""

class Inductor():
    def __init__(self, data=None, name='smi_inductor', verbose=False):
        # Input data can be either an skrf.Network or an S2P file.
        if isinstance(data, Network):
            self.data = data
        elif isinstance(data, str):
            if Path(data).is_file():
                self.data = Network(file=data, unit='Hz')
        else:
            raise ValueError("data must be either an scikit-RF Network object of a Touchstone file.")

        # Options.
        self.verbose = verbose
        self.name = name    # Name of the inductor used to write Spice, Spectre, or Xyce.

        # Useful values.
        self.f = self.data.frequency.f
        self.omega = 2 * math.pi * self.f
        self.designSpace = DefinedGammaZ0(frequency=self.data.frequency, Z0=50)
        self.lf_limit_ratio = 0.01  # ratio of Yshunt/Yseries. Defines the low frequency range.
        self.lf_limit, self.lf_limit_idx = self.get_lf_limit()
        self.srf, self.sromega, self.srf_idx, self.srf_series, self.sromega_series, self.srf_series_idx = self.get_srf()
        self.Ldc_nH, self.Rdc_ohms = self.get_Ldc_Rdc()

        # Used to extrapolate Cp, Rsub, and Csub.
        self.eps_sub = 11.9
        self.eps_imd = 3.9
        self.eps_ratio = self.eps_imd / self.eps_sub

        if self.srf is not None:
            self.mf_idx = int((self.srf_idx + self.lf_limit_idx) / 2)  # Arbitrary mid-frequency index. Used to estimate Cp.
            self.mf = self.f[self.mf_idx]
        else:
            self.mf = 10 * self.lf_limit
            self.mf_idx = get_idx_at(self.mf, self.f)

        # Perform the fit.
        self.Cox_fF, self.Rsi_ohms, self.Csi_fF = self.get_Cox_Rsi_Csi(optimize=True)
        known_parameters = {'Cox_fF': self.Cox_fF, 'Rsi_ohms': self.Rsi_ohms, 'Csi_fF': self.Csi_fF}
        estimated_Ls0_Rs0_Ls1_Rs1_dict = self.estimate_Ls0_Rs0_Ls1_Rs1()
        estimated_Cp_Rsub_Csub_dict = self.estimate_Cp_Rsub_Csub(known_parameters, estimated_Ls0_Rs0_Ls1_Rs1_dict)
        estimated_parameters = dict(estimated_Ls0_Rs0_Ls1_Rs1_dict)
        estimated_parameters.update(estimated_Cp_Rsub_Csub_dict)

        # Full optimization.
        self.model_parameters = self.full_optimization(known_parameters, estimated_parameters)
        self.model = self.build_full_model(self.model_parameters)
        #plotLR(self.data, self.model, type='series')
        #plotLR(self.data, self.model, type='in')
        #plt.show()

    def get_Ldc_Rdc(self):
        """
        Calculates Ldc and Rdc.
        :return: Ldc_nH and Rdc_ohms
        """
        Y12_data = self.data.y[:, 0, 1]
        Rseries = (-1 / Y12_data).real
        Lseries = (-1 / Y12_data).imag / self.omega

        # DC series resistance and inductance. Use lowest frequency data point.
        Rdc_ohms = Rseries[0]
        Ldc_nH = 1e9 * Lseries[0]

        return Ldc_nH, Rdc_ohms

    def get_lf_limit(self):
        """
        Gets the index and frequency where the ratio of mag(Yshunt)/mag(Yseries) = lf_limit_threshold.
        This ratio tells you when series portion of the impedance dominates.
        When the ratio is small, we can ignore the shunt network (e.g. Cox to ground). Chen uses 0.02.
        :param ntwk: Network object. <SKRF Network>
        :param limit_ratio: Maximum allowed value of mag(Yshunt)/mag(Yseries). <float>
        :return: low_frequency_limit, index. <float, int>
        """
        f = self.data.frequency.f
        Y12 = self.data.y[:, 0, 1]
        Y11 = self.data.y[:, 0, 0]
        Yseries = Y12
        Yshunt = Y11 + Y12
        Yseries_mag = mathFunctions.complex_2_magnitude(Yseries)
        Yshunt_mag = mathFunctions.complex_2_magnitude(Yshunt)
        ratio = Yshunt_mag / Yseries_mag

        comp_bool = (ratio > self.lf_limit_ratio)

        if np.all(comp_bool == False):  # i.e. purely series network, with no shunt parasitics.
            self.lf_limit_idx = len(ratio) - 1  # Return the last index. The slope calculations below will use all data.
            print("Warning: Found purely series network, with no shunt parasitics. Using all data.")
        else:
            self.lf_limit_idx = np.where(comp_bool == True)[0][0]

        # Stop operation if there is no series inductor branch.
        if np.all(comp_bool) == True or self.lf_limit_idx == 0:
            raise Exception("Not enough data points have Yshunt/Yseries < {}. No series branch found between ports 1 and 2. Check s2p file.".format(self.lf_limit_ratio))

        self.lf_limit = f[self.lf_limit_idx]
        if self.verbose:
            print("Low Frequency limit = {} GHz".format(self.lf_limit / 1e9))

        return self.lf_limit, self.lf_limit_idx

    def get_srf(self, data=None):
        """
        Finds the self-resonant frequency using where L11 and L12 changes sign from positive to negative.
        Both 'in' and 'series' SRF is calculated.
        :param data: 2-port Network. <scikit-RF Network>
        :return:
        """
        # Define data Network.
        if data is None:
            data = self.data

        # Check where the sign changes.
        #quickplot(self.f, L12(data))
        #quickplot(self.f, L11(data))
        #plt.show()

        # By default set SRF to the maximum frequency. Some inductors may not have measured data up to SRF.
        srf = self.f[-1]
        sromega = 2 * math.pi * srf
        srf_idx = len(self.f)
        srf_series = self.f[-1]
        sromega_series = 2 * math.pi * srf
        srf_series_idx = len(self.f)

        for n, y in enumerate(L11(data)):
            if y < 0 and n > self.lf_limit_idx:
                srf = self.f[n]
                srf_idx = n
                sromega = self.omega[n]
                break

        for n, y in enumerate(L12(data)):
            if y < 0 and n > self.lf_limit_idx:
                srf_series = self.f[n]
                srf_series_idx = n
                sromega_series = self.omega[n]
                break

        return srf, sromega, srf_idx, srf_series, sromega_series, srf_series_idx

    def main_branch(self, params=None, f_array=None):
        """
        Creates the main (series) branch of the inductor, an L-R network.
        Skin / Proximity effects are modeled by an RL ladder.
        :param f_array:
        :param Ls_nH:
        :param Rs_ohms:
        :return:
        """
        # If f_array is given, use that.
        if f_array is not None:
            freq = rf.Frequency.from_f(f_array, unit='hz')
            designSpace = DefinedGammaZ0(frequency=freq, Z0=50)
        else:
            designSpace = self.designSpace

        # Define components.
        Rs0 = designSpace.resistor(R=params['Rs0_ohms'], name="Rs0")
        Rs1 = designSpace.resistor(R=params['Rs1_ohms'], name="Rs1")
        Ls0 = designSpace.inductor(L=params['Ls0_nH'] * 1e-9, name="Ls0")
        Ls1 = designSpace.inductor(L=params['Ls1_nH'] * 1e-9, name="Ls1")

        ntwk = Ls0 ** parallel((Ls1 ** Rs1), Rs0)
        ntwk.name = "main branch"
        return ntwk

    def parallel_branch(self, params=None, f_array=None):
        '''
        Creates parasitic 2-port network that is in parallel to the main series inductor.
        Three paths:
            - Path1 from port1/2 to gnd: port1/2 --> Cox-->(Rsi||Csi-->gnd) (the shunt branch)
            - Path2 from port1 to port2: port1 --> Cox --> (Rsub||Csub) --> Cox --> port2
            - Path3 from port1 to port2: port1 --> Cp --> port2 (ToDo)
        :param params: Dictionary or Parameters object of Cox, Csi, Rsi, Rsub, Csub, Csi, and CP. <dict or lmfit Parameters>
        :param f_array: frequency as 1D np.array.
        :return: 2-port network object
        '''
        # If f_array is given, use that.
        if f_array is not None:
            freq = rf.Frequency.from_f(f_array, unit='hz')
            designSpace = DefinedGammaZ0(frequency=freq, Z0=50)
        else:
            designSpace = self.designSpace

        # Create components.
        Cox = designSpace.capacitor(C=params['Cox_fF'] * 1e-15)
        Csi = designSpace.capacitor(C=params['Csi_fF'] * 1e-15)
        Rsi = designSpace.resistor(R=params['Rsi_ohms'])
        Csub = designSpace.capacitor(C=params['Csub_fF'] * 1e-15)
        Rsub = designSpace.resistor(R=params['Rsub_ohms'])
        Cp = designSpace.capacitor(C=params['Cp_fF'] * 1e-15)
        gnd = designSpace.short()

        # Parallel combination of Csi and Rsi going to ground.
        ntwk = parallel(Rsi, Csi) ** gnd  # 1-port series network...
        ntwk = designSpace.shunt(ntwk)  # ...convert to 2-port shunt network.

        # Add in Cox and Rsub||Csub.
        ntwk = Cox ** ntwk ** parallel(Rsub, Csub) ** ntwk ** Cox
        ntwk = parallel(ntwk, Cp)
        ntwk.name = "parallel branch"
        return ntwk

    def build_full_model(self, params):
        """
        Builds the full inductor model given the required parameters.
        :param params: Dictionary containing values of Ls0, Rs0, Ls1, Rs1, Cox, Rsi, Csi, Cp, Rsub, Csub.
        :return:
        """
        main_branch_model = self.main_branch(params)
        parallel_branch_model = self.parallel_branch(params)
        full_model = parallel(main_branch_model, parallel_branch_model)
        return full_model

    def get_Cox_Rsi_Csi(self, data=None, optimize=False):
        '''
        Gets Cox, Rsi, and Csi from the data. These parameters can be exactly determined.
        :param data: Network object. <scikit-RF Network>
        :return:
        '''
        # Define data Network.
        if data is None:
            data = self.data

        # Isolate Cox + Rsi||Csi path.
        # Solve Z-matrix with V2=V1. This condition eliminates all series paths.
        Z11 = data.z[:, 0, 0]
        Z12 = data.z[:, 0, 1]
        Z21 = data.z[:, 1, 0]
        Z22 = data.z[:, 1, 1]
        Z_Cox_Rsi_Csi = (Z11 - Z12 * Z21 / Z22) * (1 / (1 - Z12 / Z22))

        # Estimate Rsi and Csi. Algebra gives a polynomial expression for 1/Real(Z_Cox_Rsi_Csi) = f(w) = 1/Rsi + Csi^2*Rsi*w^2.
        # This polynomial expression is valid from DC to a frequency past the LF limit, but well below SRF and lower than MF.
        # To get a good polynomial fit, sufficient data points past LF are needed.
        # As a heuristic, we select a frequency (fh) between MF and LF.
        fl = self.lf_limit
        fl_idx = 0
        fh = (self.lf_limit + self.mf)/2
        fh_idx = get_idx_at(fh, self.f)
        y = (1 / Z_Cox_Rsi_Csi.real)[fl_idx:fh_idx]
        x = self.omega[fl_idx:fh_idx]

        # Use np.polyfit to get an intial estimate for c0 and c2. This fit will include c1 term, which should be zero.
        c = np.polyfit(x, y, deg=2)

        # Use lmfit to get more accurate.
        def poly(omega, c0, c1, c2):
            return c0*omega**2 + 0*c1*omega + c2

        pmodel = Model(poly)
        params = pmodel.make_params(c0=c[0], c1=0, c2=c[2])
        params['c2'].min = 0.1e-6   # Rsi is 1/c2. Set Rsi to some max limit, otherwise optimizer might make this very small, and Csi will be incorrect also.
        #params['c0'].min = 1e-31    # Related to Csi. Ideally, set to a minimum value, but it messes up lmfit.
        params['c1'].vary = False
        result = pmodel.fit(y, params, omega=x)
        Rsi_ohms_estimate = 1 / result.params['c2']
        Csi_fF_estimate = 1e15 * math.sqrt(result.params['c0'] / Rsi_ohms_estimate)

        # Estimate Cox. Subtract out Z_Rsi_Csi.
        Z_Csi = 1 / (1j * self.omega * Csi_fF_estimate * 1e-15)
        Z_Rsi_Csi = 1 / (1 / Rsi_ohms_estimate + 1 / Z_Csi)
        Z_Cox = Z_Cox_Rsi_Csi - Z_Rsi_Csi
        Cox = -1e15 / (self.omega * Z_Cox.imag)
        Cox_fF_estimate = Cox[0]

        # If Cox is negative or below a certain value, then it needs to be calculated separately.
        # This might be the case for SOI where Cox is close to zero and the high resistivity substrate makes Rsi||Csi high impedance.
        Cox_fF_min = 1e15
        if Cox_fF_estimate <= Cox_fF_min:
            Cox_fF_estimate = -1e15 / (self.omega[0] * Z_Cox_Rsi_Csi[0].imag)

        if self.verbose:
            print("Estimated Cox_fF, Rsi_ohms, Csi_fF = {}, {}, {}".format(Cox_fF_estimate, Rsi_ohms_estimate, Csi_fF_estimate))

        # Optimize. Needed because real data doesn't follow Cox + Rsi||Csi.
        if optimize:
            params = Parameters()
            params.add('Cox_fF', value=Cox_fF_estimate, min=0.5*Cox_fF_estimate, max=1.5*Cox_fF_estimate, vary=True)
            params.add('Csi_fF', value=Csi_fF_estimate, min=0.5*Csi_fF_estimate, max=1.5*Csi_fF_estimate, vary=True)
            params.add('Rsi_ohms', value=Rsi_ohms_estimate, min=0.01*Rsi_ohms_estimate, max=10*Rsi_ohms_estimate, vary=True)
            params.add('Csub_fF', value=0.1, vary=False)  # Set to some low value. Not needed in fitting Cox, Rsi, and Csi.
            params.add('Rsub_ohms', value=10000, vary=False)  # Set to some high value. Not needed in fitting Cox, Rsi, and Csi.
            params.add('Cp_fF', value=0.001, vary=False)  # Set to some low value. Not needed in fitting Cox, Rsi, and Csi.
            out = minimize(self.residual_Cox_Rsi_Csi, params, args=(self.f, self.data))
            if self.verbose:
                print("Fitted Cox_fF, Rsi_ohms, Csi_fF = {}, {}, {}".format(out.params['Cox_fF'].value, out.params['Rsi_ohms'].value, out.params['Csi_fF'].value))
            return out.params['Cox_fF'].value, out.params['Rsi_ohms'].value, out.params['Csi_fF'].value

        return Cox_fF_estimate, Rsi_ohms_estimate, Csi_fF_estimate

    def residual_Cox_Rsi_Csi(self, params, f_array, data=None):
        """
        Residual to optimize for Cox, Rsi, and Csi.
        :param params: Parameters used to create the model network (e.g. Cox, Rsi, Csi, Rsub, Csub). <lmfit Params>
        :param f_array: Frequency array in Hz. <numpy array>
        :param data: Data of network to be fitted. <skrf Network>
        :return:
        """
        if data is None:
            data = self.data

        # Solve Z-matrix with V2=V1 for data and model networks. This condition eliminates series paths.
        # Data.
        Z11 = data.z[:, 0, 0]
        Z12 = data.z[:, 0, 1]
        Z21 = data.z[:, 1, 0]
        Z22 = data.z[:, 1, 1]
        Z_Cox_Rsi_Csi_data = (Z11 - Z12 * Z21 / Z22) * (1 / (1 - Z12 / Z22))

        # Model.
        model = self.parallel_branch(params, f_array)
        Z11 = model.z[:, 0, 0]
        Z12 = model.z[:, 0, 1]
        Z21 = model.z[:, 1, 0]
        Z22 = model.z[:, 1, 1]
        Z_Cox_Rsi_Csi_model = (Z11 - Z12 * Z21 / Z22) * (1 / (1 - Z12 / Z22))

        # Error to be minimized.
        error = Z_Cox_Rsi_Csi_data - Z_Cox_Rsi_Csi_model
        error = error[0:self.lf_limit_idx].view(np.float)

        if self.verbose:
            print("    Optimizing Cox, Rsi, Csi...")

        return error.view()

    def residual_Ls0_Rs0_Ls1_Rs1(self, params=None, Ldc_nH=None, Rdc_ohms=None, f_array=None, data=None):
        """
        Residual to do initial estimate of Ls0, Rs0, Ls1, and Rs1.
        :param params: Parameters used to create the model network (e.g. Ls0, Rs0, Ls1, Rs1). <lmfit Params>
        :param Rdc: Resistance at DC or very low frequency. <float>
        :param Ldc: Inductance at DC or very low frequency. <float>
        :param f_array: Frequency array in Hz. <numpy array>
        :param data: Data of network to be fitted. <skrf Network>
        :return:
        """
        # Create the model.
        if data is None:
            data = self.data

        # Calculate Rs1 and Ls0.
        Rs0_ohms = params['Rs0_ohms'].value
        Rs1_ohms = (Rdc_ohms * Rs0_ohms) / (Rs0_ohms - Rdc_ohms)
        Rt = Rs0_ohms + Rs1_ohms
        Ls1_nH = params['Ls1_nH'].value
        Ls0_nH = Ldc_nH - ((Ls1_nH * Rs0_ohms ** 2) / Rt ** 2)

        # Build the inductor.
        params_series = {}
        params_series['Rs0_ohms'] = params['Rs0_ohms'].value
        params_series['Rs1_ohms'] = Rs1_ohms
        params_series['Ls0_nH'] = Ls0_nH
        params_series['Ls1_nH'] = params['Ls1_nH'].value
        main_branch = self.main_branch(params_series)
        #parallel_branch = self.parallel_branch(params)

        #model = parallel(main_branch, parallel_branch)
        model = self.main_branch(params_series)

        # Error.
        error = model.y[:, 0, 1] - data.y[:, 0, 1]
        error = error.view(np.float)

        return error

    def residual_Cp_Rsub_Csub(self, params=None, f_array=None, data=None):
        """
        Residual to determine Cp, Rsub, and Csub.
        :param f_array:
        :param Ls_nH:
        :param Rs_ohms:
        :return:
        """
        # Create the model.
        # Copy the lmfit params to a simple dictionary.
        parameters2 = {}
        parameters2['Rs0_ohms'] = params['Rs0_ohms'].value
        parameters2['Rs1_ohms'] = params['Rs1_ohms'].value
        parameters2['Ls0_nH'] = params['Ls0_nH'].value
        parameters2['Ls1_nH'] = params['Ls1_nH'].value
        parameters2['Cox_fF'] = params['Cox_fF'].value
        parameters2['Csi_fF'] = params['Csi_fF'].value
        parameters2['Rsi_ohms'] = params['Rsi_ohms'].value
        parameters2['Cp_fF'] = params['Cp_fF'].value
        parameters2['Rsub_ohms'], parameters2['Csub_fF'] = self.calculate_Rsub_Csub(parameters2['Cp_fF'])

        # Build the model.
        main_branch = self.main_branch(parameters2)
        parallel_branch = self.parallel_branch(parameters2)
        model = parallel(main_branch, parallel_branch)

        # Error. Minimize the excess capacitance by increasing Cp and/or Csub.
        Yseries_data = data.y[:, 0, 1]
        Yseries_model = model.y[:, 0, 1]
        Y_Rsub_Csub_Cp = Yseries_model - Yseries_data  # "data" has more parasitic cap than "model", so admittance Ycap,model > Ycap,data
        excess_cap = 1e15 * Y_Rsub_Csub_Cp.imag / self.omega
        error = excess_cap[self.lf_limit_idx:self.srf_idx]

        return error

    def estimate_Ls0_Rs0_Ls1_Rs1(self, data=None):
        '''
        Estimates Ls0, Rs0, Ls1, Rs1.
        It is an estimate because the parasitive capacitive branches are unknown and assumed to be open at low frequencies.
        :param data: Network object. <scikit-RF Network>
        :return: Ls0, Rs0, Ls1, Rs1
        '''
        # Define data Network.
        if data is None:
            data = self.data

        # Isolate the series path. This includes the LR path and the Cox+(Rsub||Csb)+Cox path and the Cp path.
        # For initial estimate, assume L-R path is dominant at low frequencies.
        Y12_data = data.y[:, 0, 1]
        Y12_data = Y12_data[0:self.lf_limit_idx]
        Ru = (-1 / Y12_data).real
        Lu = (-1 / Y12_data).imag / self.omega[0:self.lf_limit_idx]

        # DC series resistance and inductance. Use lowest frequency data point.
        Rdc = Ru[0]
        Ldc = Lu[0]
        Rdc_ohms = Rdc
        Ldc_nH = Ldc * 1e9

        # Calculate some slopes based on Chen's paper.
        y = Ru - Rdc
        x = Ldc - Lu
        bestfit_slope, _, _, _ = np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)    # Force y-intercept to 0.
        T = bestfit_slope[0]

        x2 = 1 / (1 + (T ** 2) / (self.omega[0:self.lf_limit_idx]) ** 2)
        M, b = np.polyfit(x2, y, 1)

        # Estimate parameters.
        Rs0_ohms_est = M + Rdc
        Rs1_ohms_est = (Rs0_ohms_est * Rdc) / M
        Ls0_nH_est = 1e9 * (Ldc - M / T)
        Ls1_nH_est = 1e9 * (Rs0_ohms_est + Rs1_ohms_est) / T
        if self.verbose:
            print("Estimated: Ls0_est={}, Rs0_est={}, Ls1_est={}, Rs1_est={}, Ldc={}, Rdc={}".format(Ls0_nH_est, Rs0_ohms_est, Ls1_nH_est, Rs1_ohms_est, Ldc, Rdc))

        # Optimize. Rs1 and Ls0 are calculated in the residual from Rs1 and Ls0.
        params = Parameters()
        params.add('Rs0_ohms', value=Rs0_ohms_est, min=0, max=20, vary=True)
        # params.add('Rs1_ohms', value=12, min=0, max=20, vary=False)
        # params.add('Ls0_nH', value=9, min=0, max=20, vary=False)
        params.add('Ls1_nH', value=Ls1_nH_est, min=0, max=20, vary=True)
        out = minimize(self.residual_Ls0_Rs0_Ls1_Rs1, params, args=(self.f, Ldc_nH, Rdc_ohms, self.data))

        # Build the model and check.
        # Calculate Rs1_ohms and Ls0_ohms from fitted/known values of Rs0_ohms, Ls0_ohms, Rdc_ohms, and Ldc_nH.

        Rs0_ohms = params['Rs0_ohms'].value
        Rs1_ohms = (Rdc_ohms * Rs0_ohms) / (Rs0_ohms - Rdc_ohms)
        Rt = Rs0_ohms + Rs1_ohms
        Ls1_nH = params['Ls1_nH'].value
        Ls0_nH = Ldc_nH - ((Ls1_nH * Rs0_ohms ** 2) / Rt ** 2)

        out.params.add('Rs1_ohms', value=Rs1_ohms, vary=False)
        out.params.add('Ls0_nH', value=Ls0_nH, vary=False)

        if self.verbose:
            print("Optimized: Ls0_nH_est={}, Rs0_ohms_est={}, Ls1_nH_est={}, Rs1_ohms_est={}".format(out.params['Ls0_nH'].value, out.params['Rs0_ohms'].value, out.params['Ls1_nH'].value, out.params['Rs1_ohms'].value))

        # Return the parameters as a simple dictionary.
        params = {}
        for param in out.params:
            params[param] = out.params[param].value

        return params

    def estimate_Cp_Rsub_Csub(self, known_params, main_branch_params, data=None):
        '''
        Once the main branch parameters are known, then these parameters can be extracted by
        subtracting the main branch parameters from Y12 (or Y21).
        :param main_branch_params: Network parameters of main LR branch of fitted data. <dict>
        :param data: Network parameters of data to be fitted. <scikit-RF Network>
        :return:
        '''
        # Define data network.
        if data is None:
            data = self.data

        # Define model main branch network.
        model_main_branch = self.main_branch(main_branch_params)

        # Deembed the main branch from the series network.
        Yseries_data = data.y[:, 0, 1]
        Yseries_model = model_main_branch.y[:, 0, 1]    # No shunt branch so Y11=Y12=Y21=Y22.

        # Calculate the excess admittance seen in data vs model.
        Y_Rsub_Csub_Cp = Yseries_model - Yseries_data   # The data has more parasitic cap than the model, so admittance Ycap,model > Ycap,data
        excess_admittance = Y_Rsub_Csub_Cp  # Gives Rsub||Csub||Cp.
        excess_cap = excess_admittance.imag / self.omega
        # excess_res = -1 / excess_admittance.real    # Doesn't give good results because estimated main branch parameters are not fully accurate.

        # Excess cap > 0, otherwise that means there is no further capacitance to add to the model.
        # Return a small Cp and Csub value and large value of Rsub just to build model without any impact on electricals.
        if np.any(excess_cap[self.lf_limit_idx:self.srf_series_idx] < 0):
            if self.verbose: print('No excess cap found. Setting minimum values for Cp_fF, Rsub_ohms, and Csub_fF.')
            Cp_fF_estimate = 1e-6
            Csub_fF_estimate = 1e-6
            Rsub_ohms_estimate = 1e9

        else:
            # Estimate the excess cap.
            # This excess cap models the SRF so it is useful to choose a frequency between LF and SRF.
            excess_cap_at_mf = 1e15 * excess_cap[self.mf_idx]
            if self.verbose: print('Excess cap =', excess_cap_at_mf)

            # Optimize. Use the excess cap as an estimate to Cp. Rsub and Csub are calculated.
            params = Parameters()
            for param_name in main_branch_params:
                params.add(param_name, value=main_branch_params[param_name], vary=False)
            for param_name in known_params:
                params.add(param_name, value=known_params[param_name], vary=False)
            params.add('Cp_fF', value=excess_cap_at_mf, min=0.5*excess_cap_at_mf, max=1.5*excess_cap_at_mf, vary=True)
            #params.add('Csub_fF', value=0.1, vary=False)  # Set to some low value. Will be estimated later.
            #params.add('Rsub_ohms', value=10000, vary=False)  # Set to some high value. Will be estimated later.
            out = minimize(self.residual_Cp_Rsub_Csub, params, args=(self.f, self.data))

            # Estimate Csub by using dielectric ratio of Si vs SiO2.
            Cp_fF_estimate = out.params['Cp_fF'].value
            Rsub_ohms_estimate, Csub_fF_estimate = self.calculate_Rsub_Csub(Cp_fF_estimate)

            if self.verbose:
                print(Cp_fF_estimate, Csub_fF_estimate, Rsub_ohms_estimate)

        # Return the parameters as a simple dictionary.
        params = {}
        params['Cp_fF'] = Cp_fF_estimate
        params['Rsub_ohms'] = Rsub_ohms_estimate
        params['Csub_fF'] = Csub_fF_estimate

        return params

    def residual_full_model(self, params=None, f_array=None, data=None):
        """
        Residual to fit full model using Y12.
        Shunt parameters (Cox, Rsi, Csi) are already determined (known) so fitting Y11 is not needed.
        :param params:
        :param f_array:
        :param data:
        :return:
        """
        # Build the series branch model. Ls1_nH and Rs1_ohms are calculated.
        Rs0_ohms = params['Rs0_ohms'].value
        Ls0_nH = params['Ls0_nH'].value
        Ls1_nH, Rs1_ohms = self.calculate_Ls1_Rs1(Ls0_nH, Rs0_ohms)

        params_series = {}
        params_series['Rs0_ohms'] = Rs0_ohms
        params_series['Rs1_ohms'] = Rs1_ohms
        params_series['Ls0_nH'] = Ls0_nH
        params_series['Ls1_nH'] = Ls1_nH
        main_branch = self.main_branch(params_series)

        # Build the parallel branch model. Rsub_ohms and Csub_fF are calculated.
        Cp_fF = params['Cp_fF'].value
        Rsub_ohms, Csub_fF = self.calculate_Rsub_Csub(Cp_fF)

        params_parallel = {}
        params_parallel['Cox_fF'] = params['Cox_fF'].value
        params_parallel['Csi_fF'] = params['Csi_fF'].value
        params_parallel['Rsi_ohms'] = params['Rsi_ohms'].value
        params_parallel['Cp_fF'] = Cp_fF
        params_parallel['Csub_fF'] = Csub_fF
        params_parallel['Rsub_ohms'] = Rsub_ohms
        parallel_branch = self.parallel_branch(params_parallel)

        model = parallel(main_branch, parallel_branch)

        # Minimize Y12 error up to SRF. Shunt branch (Cox, Rsi, Csi) is already determined.
        Y12_data = data.y[:, 0, 1]
        Y12_model = model.y[:, 0, 1]
        error = Y12_model - Y12_data
        error = error[0:self.srf_idx].view(np.float)

        return error

    def full_optimization(self, known_parameters=None, estimated_parameters=None, data=None):
        """
        Final full optimization after estimates for all component parameters are known.
        :param estimated_parameters: <dict>
        :param data: Network parameters of data to be fitted. <scikit-RF Network>
        :return: Optimized parameters
        """
        # Build the model.
        # 'Ls1_nH', 'Rs1_nH', 'Rsub_ohms', 'Csub_fF' are dependent parameters and calculated.
        params = Parameters()
        for param_name in known_parameters:
            params.add(param_name, value=known_parameters[param_name], vary=False)

        params.add('Ls0_nH', value=estimated_parameters['Ls0_nH'], min=0.5*estimated_parameters['Ls0_nH'], max=0.99*self.Ldc_nH, vary=True)
        params.add('Rs0_ohms', value=estimated_parameters['Rs0_ohms'], min=1.01*self.Rdc_ohms, max=1.5*estimated_parameters['Rs0_ohms'], vary=True)
        params.add('Cp_fF', value=estimated_parameters['Cp_fF'], min=0.5*estimated_parameters['Cp_fF'], max=1.5*estimated_parameters['Cp_fF'], vary=True)

        if self.verbose:
            print("Prior to optimizing:")
            for parameter in estimated_parameters:
                print(parameter, estimated_parameters[parameter])

        # Optimize.
        if self.verbose: print("Running full optimization. This might take some time.")
        out = minimize(self.residual_full_model, params, args=(self.f, self.data))

        fully_fitted_params = {}
        fully_fitted_params['Ls0_nH'] = out.params['Ls0_nH'].value
        fully_fitted_params['Rs0_ohms'] = out.params['Rs0_ohms'].value
        fully_fitted_params['Ls1_nH'], fully_fitted_params['Rs1_ohms'] = self.calculate_Ls1_Rs1(out.params['Ls0_nH'], out.params['Rs0_ohms'])
        fully_fitted_params['Cp_fF'] = out.params['Cp_fF'].value
        fully_fitted_params['Rsub_ohms'], fully_fitted_params['Csub_fF'] = self.calculate_Rsub_Csub(out.params['Cp_fF'])
        fully_fitted_params['Cox_fF'] = out.params['Cox_fF'].value
        fully_fitted_params['Rsi_ohms'] = out.params['Rsi_ohms'].value
        fully_fitted_params['Csi_fF'] = out.params['Csi_fF'].value

        if self.verbose:
            print("After optimization:")
            for param in fully_fitted_params:
                print(param, fully_fitted_params[param])


        return fully_fitted_params

    def calculate_Ls1_Rs1(self, Ls0_nH, Rs0_ohms):
        """
        Given Ls0_nH and Rs0_ohms, Ls1_nH and Rs1_ohms can be calculated.
        :param Ls1_nH:
        :param Rs0_ohms:
        :param Ldc_nH:
        :param Rdc_ohms:
        :return:
        """
        if Rs0_ohms < self.Rdc_ohms:
            raise ValueError("Rs0_ohms ({}) is less than Rdc_ohms ({}). Would result in negative Rs1_ohms.".format(Rs0_ohms, self.Rdc_ohms))

        if Ls0_nH > self.Ldc_nH:
            raise ValueError("Ls0_nH ({}) is greater than Ldc_nH ({}). Would result in negative Ls1_nH.".format(Ls0_nH, self.Ldc_nH))

        Rs1_ohms = (self.Rdc_ohms * Rs0_ohms) / (Rs0_ohms - self.Rdc_ohms)
        Rt = Rs0_ohms + Rs1_ohms
        Ls1_nH = ((self.Ldc_nH - Ls0_nH) * Rt**2) / (Rs0_ohms**2)
        #print(Ls0_nH, Rs0_ohms, Ls1_nH, Rs1_ohms, self.Ldc_nH, self.Rdc_ohms)
        return Ls1_nH, Rs1_ohms

    def calculate_Rsub_Csub(self, Cp_fF):
        """
        Given Cp_fF, Rsub and Csub can be estimated.
        :param Cp_fF:
        :return: Rsub_ohms, Csub_fF
        """
        Csub_fF = (1 - self.eps_ratio) * Cp_fF
        Rsub_ohms = (self.Rsi_ohms * self.Csi_fF) / Csub_fF

        return(Rsub_ohms, Csub_fF)

    def write_spice(self, filename=None, subckt=True):
        """
        Creates a spice netlist using the modeled parameters.
        :return:
        """
        if filename is None:
            filename = "{}.cir".format(self.name)
            path = Path(filename)

        text = ("Ls0 PLUS N001 {Ls0_nH}e-9\n"
               "Rs0 N001 MINUS {Rs0_ohms}\n"
               "Ls1 N001 N002 {Ls1_nH}e-9\n"
               "Rs1 N002 MINUS {Rs1_ohms}\n"
               "Cox1L PLUS N003L {Cox_fF}e-15\n"
               "Rsi1L N003L 0 {Rsi_ohms}\n"
               "Csi1L N003L 0 {Csi_fF}e-15\n"
               "Cox1R MINUS N003R {Cox_fF}e-15\n"
               "Rsi1R N003R 0 {Rsi_ohms}\n"
               "Csi1R N003R 0 {Csi_fF}e-15\n"
               "Csub N003L N003R {Csub_fF}e-15\n"
               "Rsub N003L N003R {Rsub_ohms}\n"
               "Cp PLUS MINUS {Cp_fF}").format(**self.model_parameters)

        if subckt:
            text = ".subckt {} PLUS MINUS\n".format(self.name) + text
            text = text + "\n.ends"

        with open(path, mode='w') as fid:
            fid.write(text)

