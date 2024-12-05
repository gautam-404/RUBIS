from rubis.helpers                 import DotDict, give_me_a_name, assign_method, write_model, init_2D
from rubis.plot                    import get_cmap_from_proplot
from rubis.rotation_profiles       import solid, lorentzian, plateau, la_bidouille
from rubis.model_deform_radial     import radial_method, init_1D
from rubis.model_deform_spheroidal import spheroidal_method

# from MESAcontroller import ProjectOps, MesaAccess

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import glob
import matplotlib.pyplot as plt
import h5py    
import numpy as np    
import os, psutil, sys
from contextlib import redirect_stdout

from tqdm import tqdm
import multiprocessing as mp

from mesaport import ProjectOps, MesaAccess

class RUBISmod:
    """Class to implement rubis modifications to MESA models
    """
    def __init__(self, name, data_format="GSM"):
        """Initialise the RUBISmod class

        Parameters
        ----------
        name : str
            Path to the MESA project
        data_format : str
            Format of the MESA data. Either GSM or GYRE

        Raises
        ------
        AssertionError
            If the path to the MESA project does not exist
        ValueError
            If the data_format is not GSM or GYRE
        """
        assert os.path.exists(name), "The path to the MESA project does not exist"

        self.name = name
        self.data_format = data_format
        if data_format == "GSM":
            self.profiles = sorted(glob.glob(os.path.join(name, 'LOGS/*profile*.data.GSM')), key=lambda x: int(x.split(".")[-3].split("profile")[1]))
            if self.profiles == []:
                self.profiles = sorted(glob.glob(os.path.join(name, '*profile*.data.GSM')), key=lambda x: int(x.split('/')[-1].split(".")[-3].split("profile")[1]))
        elif data_format == "GYRE":
            self.profiles = sorted(glob.glob(os.path.join(name, 'LOGS/*profile*.data.GYRE')), key=lambda x: int(x.split(".")[-3].split("profile")[1]))
            if self.profiles == []:
                self.profiles = sorted(glob.glob(os.path.join(name, '*profile*.data.GYRE')), key=lambda x: int(x.split('/')[-1].split(".")[-3].split("profile")[1]))
        else:
            raise ValueError("data_format must be either GSM or GYRE")
        if self.profiles == []:
            raise ValueError(f"No profiles found in the MESA project {name} with the data format .{data_format}")


    def write_rubis_profile(self, df, header, profile, angular_resolution, file_format='GSM'):
        """Function to write a rubis profile to a file

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame containing the rubis profile
        header : str
            Header of the rubis profile
        profile : str
            Path to the profile to be modified
        file_format : str, optional
            Format of the profile, by default 'GSM'
        """
        dirname = os.path.dirname(profile)
        outfile = profile+".rubis"
        outheader = pd.read_csv(outfile, sep=' ', nrows=1, header=None, names=['N', 'res','M', 'R', 'Omega_eq', 'G'])
        r_zeta_theta = np.arange(0, angular_resolution)
        outdf = pd.read_csv(outfile, sep=' ', skiprows=1, names=[f'r_{i}' for i in r_zeta_theta]+['zeta', 'P', 'rho', 'phi_eff', 'Omega'])
        outdf = pd.concat([outdf, df.Gamma_1], axis=1)
        outheader.to_csv(outfile, sep=' ', index=False, header=None)
        outdf.to_csv(outfile, sep=' ', index=False, float_format='%.18e', mode='a', header=None)

    
    def rubis_model(self, profile, angular_resolution, **kwargs):
        """Function to implement rubis modifications to a MESA profile

        Parameters
        ----------
        profile : str
            Path to the MESA profile
        angular_resolution : int
            Angular resolution of the model

        Keyword Arguments
        -----------------
        save_model : bool, default=False
        show_model : bool, default=False
        show_T_eff : bool, default=False
        show_harmonics: bool, default=False
        write_to_log: bool, default=True

        Returns
        -------
        DotDict
            Dictionary containing the output of the model
        tuple
            Tuple containing the output of the model.   
            N, M, mass, radius, rotation_target, G, map_n, additional_var, zeta, P, rho, phi_eff, rotation_profile
        """
        assert angular_resolution % 2 == 1, "Angular resolution must be odd to include the equator"

        # kwargs
        save_model, show_model, show_T_eff, show_harmonics, write_to_log = kwargs.pop("save_model", False), kwargs.pop("show_model", False), \
                                            kwargs.pop("show_T_eff", False), kwargs.pop("show_harmonics", False), kwargs.pop("write_to_log", True)
        
        if ".data.GYRE" in profile:
            df_head = pd.read_csv(profile, nrows=1, delim_whitespace=True, names=["N", "M", "R", "L", "verison"])
            df = pd.read_csv(profile, skiprows=1, delim_whitespace=True, 
                            names=['k', 'r', 'Mr', 'Lr', 'P', 'T', 'rho', 'grad',
                                    'N2', 'adexpo', 'ad_temp_grad', 'thermo_coeff',
                                    'opacity', 'opacity_partial1', 'opacity_partial2',
                                    'nuc_E_gen_rate', 'nuc_E_gen_rate_partial1', 'nuc_E_gen_rate_partial2',
                                    'omega'])
            N = df_head.N.iloc[-1]
            R = df_head.R.iloc[-1]
            M = df_head.M.iloc[-1]
            omega = df.omega.iloc[-1]
            surface_pressure = df.P.iloc[-1]
            r1D = df.r.values
            rho1D = df.rho.values
        elif ".data.GSM" in profile:
            file = h5py.File(profile, 'r')
            N = file.attrs['n']
            R = file.attrs['R_star']
            M = file.attrs['M_star']
            omega = file['Omega_rot'][-1]
            surface_pressure = file['P'][-1]
            r1D = file['r'][:]
            rho1D = file['rho'][:]
        G = 6.6743e-8
        omega_crit = np.sqrt((G*M)/R**3)
        method_choice = 'radial'

        #### MODEL CHOICE ####
        radial_res = N
        # additional_var = (df[col].values for col in df.columns)
        additional_var = ()
        model_choice = (surface_pressure, radial_res, r1D, rho1D, *additional_var)

        #### ROTATION PARAMETERS ####      
        rotation_profile = solid
        rotation_target = omega/omega_crit
        central_diff_rate = 1.0
        rotation_scale = 1.0

        #### SOLVER PARAMETERS ####
        max_degree = angular_resolution
        full_rate = 3
        mapping_precision = 1e-10
        lagrange_order = 3
        spline_order = 5

        #### OUTPUT PARAMETERS ####
        output_params = DotDict(
            # Tests
            show_harmonics = show_harmonics,
            virial_test = True,
            # Model
            show_model = show_model,
            plot_resolution = 501,
            plot_surfaces = False,
            plot_cmap_f = sns.color_palette("viridis", as_cmap=True),
            # plot_cmap_f = sns.color_palette("viridis_r", as_cmap=True),
            plot_cmap_surfaces = get_cmap_from_proplot("Greys"),
            gravitational_moments = False,
            # Radiative flux
            radiative_flux = False,
            plot_flux_lines = False,
            flux_origin = 0.05,
            flux_lines_number = 30,
            show_T_eff = show_T_eff,
            flux_res = (200, 100),
            flux_cmap = get_cmap_from_proplot("Stellar_r"),
            # Model writting
            dim_model = True,
            save_model = save_model,
            save_name = profile+".rubis"
        )

        #### SPHEROIDAL PARAMETERS ####
        external_domain_res = 201
        rescale_ab = False

        # Choosing the method to call
        method_func = assign_method(
            method_choice, model_choice, radial_method, spheroidal_method
        )

        # Performing the deformation
        if write_to_log:
            with open(self.name+"/rubis.log", "a+") as f:
                f.write(f"\nRunning rubis on {profile}\n")
                with redirect_stdout(f):
                    try:
                        rubis_out = method_func(
                            model_choice, 
                            rotation_profile, rotation_target, central_diff_rate, rotation_scale, 
                            max_degree, angular_resolution, full_rate,
                            mapping_precision, spline_order, lagrange_order,
                            output_params, 
                            external_domain_res, rescale_ab
                        )
                        f.write(f"Rubis on {profile} completed\n")
                    except Exception as e:
                        f.write(f"Rubis on {profile} failed\n")
                        f.write(f"Error: {e}\n")
                        return output_params, None
        else:
            rubis_out = method_func(
                            model_choice, 
                            rotation_profile, rotation_target, central_diff_rate, rotation_scale, 
                            max_degree, angular_resolution, full_rate,
                            mapping_precision, spline_order, lagrange_order,
                            output_params, 
                            external_domain_res, rescale_ab
                        )
        if show_model:
            plt.show()
        return output_params, rubis_out
    

    def run_rubis(self, profile, angular_resolution, **kwargs):
        """Function to implement rubis modifications to a MESA profile
        
        Parameters
        ----------
        profile : str
            Path to the profile to be modified
        angular_resolution : int, optional
            Angular resolution of the model, by default 181
        save_model : bool, optional
            Whether to save the model or not, by default False
        show_model : bool, optional
            Whether to show the model or not, by default False
        show_T_eff : bool, optional
            Whether to show the effective temperature or not, by default False
        show_harmonics : bool, optional
            Whether to show the harmonics or not, by default False
        write_to_log : bool, optional
            Whether to write the output to a log file or not, by default True
        """
        save_model, show_model, show_T_eff, show_harmonics, write_to_log = kwargs.pop("save_model", False), kwargs.pop("show_model", False), \
                                            kwargs.pop("show_T_eff", False), kwargs.pop("show_harmonics", False), kwargs.pop("write_to_log", True)
        output_params, rubis_out = self.rubis_model(profile, angular_resolution, save_model=save_model, show_model=show_model,
                                                    show_T_eff=show_T_eff, show_harmonics=show_harmonics, write_to_log=write_to_log)
        if rubis_out is None:
            return
        N, M, mass, radius, rotation_target, G, map_n, additional_var, zeta, P, rho, phi_eff, rota = rubis_out
        params = (N, M, mass, radius, rotation_target, G)
        args = zeta, P, rho, phi_eff, rota

        cols = [f'r_{i}' for i in range(0, angular_resolution)]
        other_cols = ['zeta', 'Pressure', 'Density', 'Potential', 'Omega_s']

        df = pd.DataFrame(np.hstack((map_n, np.vstack(args + (*additional_var,)).T)), columns=cols+other_cols)

        if self.data_format == "GSM":
            with h5py.File(profile, 'r') as file:
                dictionary = {}
                header = []
                for key in file.keys():
                    dictionary[key] = file[key][()]
                header.append(file.attrs['n'])
                header.append(file.attrs['M_star'])
                header.append(file.attrs['R_star'])
                header.append(file.attrs['L_star'])
                header.append(101)
                header = pd.DataFrame([header], columns=['N', 'M', 'R', 'L', 'v'])
            dfx = pd.DataFrame.from_dict(dictionary).drop(columns=['P', 'rho']).rename(columns={'r': 'r_MESA'})
        elif self.data_format == "GYRE":
            dfx = pd.read_csv(profile, skiprows=1, delim_whitespace=True,
                        names = ['k', 'r_MESA', 'M_r', 'L_r', 'Pressure', 'T', 'Density', 'nabla',
                            'N2', 'Gamma_1', 'nabla_ad', 'delta',
                            'kap', 'kap_kap_T', 'kap_kap_rho',
                            'eps', 'eps_eps_T', 'eps_eps_rho',
                            'Omega_rot']).drop(columns=['Pressure', 'Density'])
            header = pd.DataFrame([[len(dfx), dfx["M_r"].iloc[-1], dfx["r_MESA"].iloc[-1], dfx["L_r"].iloc[-1], 101]], columns=['N', 'M', 'R', 'L', 'v'])
        else:
            raise ValueError("Data format not recognized")
        names = cols+other_cols+list(dfx.columns)
        names = [name+' '*(15-len(name)) for name in names]
        df = pd.concat([df, dfx], axis=1, names=names)
        self.write_rubis_profile(df, header, profile, angular_resolution)
        return df, header

    def create_rubis_profiles(self, angular_resolution=181, n_processes=None, **kwargs):
        os.environ['HDF5_USE_FILE_LOCKING'] = 'False'
        os.environ['OMP_NUM_THREADS'] = '1'
        n_processes = n_processes if n_processes is not None else psutil.cpu_count(logical=False)
        args = [(profile, angular_resolution, kwargs) for profile in self.profiles]

        with tqdm(total=len(self.profiles), desc="Running RUBIS...") as progressbar:
            with mp.Pool(n_processes) as pool:
                for _ in pool.imap_unordered(self.run_rubis_wrapper, args):
                    progressbar.update()

    def run_rubis_wrapper(self, args):
        profile, angular_resolution, kwargs = args
        # Call the actual method, ensure minimal overhead in this wrapper
        return self.run_rubis(profile, angular_resolution, **kwargs)


def get_gyre_params(archive_name, suffix=None, zinit=None, run_on_cool=True, file_format="GYRE"):
    '''
    Compute the GYRE input parameters for a given MESA model.

    Parameters
    ----------
    name : str
        Name of the MESA model.
    zinit : float, optional
        Initial metallicity of the MESA model. The default is None. If None, the metallicity is read from the MESA model.
    run_on_cool : bool, optional
        If True, run GYRE on all models, regardless of temperature. The default is False.
    file_format : str, optional
        File format of the MESA model. The default is "GYRE".
    
    Returns
    -------
    profiles : list
        List of MESA profile files to be run with GYRE.
    gyre_input_params : list
    '''
    archive_name = os.path.abspath(archive_name)
    if suffix == None:
        histfile = os.path.join(archive_name, "histories", "history.data")
        pindexfile = os.path.join(archive_name, "profile_indexes", "profiles.index")
    else:
        histfile = os.path.join(archive_name, "histories", f"history_{suffix}.data")
        pindexfile = os.path.join(archive_name, "profile_indexes", f"profiles_{suffix}.index")
    h = pd.read_csv(histfile, sep='\s+', skiprows=5)
    h["Zfrac"] = 1 - h["average_h1"] - h["average_he4"]
    h["Myr"] = h["star_age"]*1.0E-6
    h["density"] = h["star_mass"]/np.power(10,h["log_R"])**3
    p = pd.read_csv(pindexfile, skiprows=1, names=['model_number', 'priority', 'profile_number'], sep='\s+')
    h = pd.merge(h, p, on='model_number', how='inner')
    h = h.sort_values(by='Myr')
    gyre_start_age = 0
    gyre_intake = h.query(f"Myr > {gyre_start_age/1e6}")
    profiles = []
    gyre_input_params = []
    for i,row in gyre_intake.iterrows():
        p = int(row["profile_number"])
        gyre_profile = f"profile{p}.data.{file_format}"
            
        ###Checks###
        # if not os.path.exists(profile_file):
        #     raise FileNotFoundError("Profile not found. Possible file format mismatch")
        if row["log_Teff"] < 3.778 and not run_on_cool:
            continue
        ############

        if row["Myr"] > 40:
            diff_scheme = "COLLOC_GL2"
        else:
            diff_scheme = "MAGNUS_GL2"
        freq_min = 0.8
        if zinit < 0.003:
            freq_max = 150
        else:
            freq_max = 95
        profiles.append(gyre_profile)
        gyre_input_params.append({"freq_min": freq_min, "freq_max": freq_max, "diff_scheme": diff_scheme})
    return profiles, gyre_input_params

def run_gyre(name):
    proj = ProjectOps(name)
    print("[bold green]Running GYRE...[/bold green]")
    os.environ['HDF5_USE_FILE_LOCKING'] = 'False'
    os.environ['OMP_NUM_THREADS'] = '1'
    file_format = "GYRE"
    profiles, gyre_input_params = get_gyre_params(name, zinit=None, run_on_cool=True, file_format=file_format)
    if len(profiles) > 0:
        profiles = [profile.split('/')[-1] for profile in profiles]
        proj.runGyre(gyre_in=os.path.expanduser("./templates/gyre_rot_template_dipole.in"), files=profiles, data_format=file_format, 
                    logging=False, parallel=True, n_cores=None, gyre_input_params=gyre_input_params)
    else:
        with open(f"{name}/run.log", "a+") as f:
            f.write(f"GYRE skipped: no profiles found, possibly because all models had T_eff < 6000 K\n")


if __name__ == "__main__":
    obj = RUBISmod("test_benchmark/m1.7_z0.015_v10", data_format="GSM")
    # obj.create_rubis_profiles(angular_resolution=45, show_output=False, 
    #                           save_model=True, show_model=False, show_T_eff=False, 
    #                           show_harmonics=False, write_to_log=True)

    obj.create_rubis_profiles(angular_resolution=200, show_output=False, 
                              save_model=True, show_model=False, show_T_eff=False, 
                              show_harmonics=False, write_to_log=True)