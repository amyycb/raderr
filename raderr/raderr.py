import h5py
import numpy as np
import wradlib as wrl
import atten_rain as atten_r
import h5py
import numpy as np
import os
from fft import spectral_sim as sp_sim
from scipy.stats import norm, gamma
import wradlib as wrl
from os.path import split
import scipy.stats


class SimRad:
    def __init__(self,
                 rainfall_path,
                 adv=(0, 0), # vel_x, vel_y
                 radar_config=None,
                 cloud_config=None,
                 vertical_vario=None,
                 noise=None,
                 dsd_a=None,
                 dsd_b=None,
                 ref_atten=None,
                 rain_atten=None,
                 elev_config = "uk",
                 mem_limit=5):

        # read in rainfall field
        with h5py.File(rainfall_path, "r") as hf:
            self.rain_xy = hf["rainfall"][:]

        # meta data
        self.dur = self.rain_xy.shape[0]
        self.domain = (self.rain_xy.shape[2], self.rain_xy.shape[1]) # x dim, y dim

        # advection shifts
        self.adv = adv 
        self.nu_shifts = np.zeros([2, self.dur], dtype=int)
        for i in range(2):
            # convert to pixels
            shifts = - (np.arange(self.dur) * self.adv[i] / 1000 * 300)
            # convert to integers
            self.nu_shifts[i] = (shifts - np.min(shifts)).astype(int)

        # config
        self.sigma_e = None
        self.radar_config = radar_config
        self.cloud_config = cloud_config
        self.vertical_vario = vertical_vario
        self.noise = noise
        self.dsd_a = dsd_a
        self.dsd_b = dsd_b
        self.ref_atten = ref_atten
        self.rain_atten = rain_atten

        # radar elevation set up (UK, Germany or manual elevation specification)
        if elev_config == "uk":
            self.radar_config["elevs"] = np.array([0.2, 0.5, 1, 1.5, 2, 2])  # radar rays - UK configuration
        elif elev_config == "deu":
            self.radar_config["elevs"] = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8, 12, 17, 25])  # radar rays - German configuration
        else:
            self.radar_config["elevs"] = elev_config

        self.radar_config["n_bins"] = int(np.ceil((radar_config["range"] - radar_config["bin_start"]) / radar_config["bin_width"]))

        # coordinates
        self.coords = {
            "thetas" : np.arange(self.radar_config["n_thetas"]),
            "bins" : self.radar_config["bin_start"] + self.radar_config["bin_width"] * 0.5 + self.radar_config["bin_width"] * np.arange(self.radar_config["n_bins"]),
            "x" : self.radar_config["bng_coords"][0] + np.arange(- 1000 * self.radar_config["range"], 1000 * self.radar_config["range"], 1000),
            "y" : self.radar_config["bng_coords"][1] + np.arange(- 1000 * self.radar_config["range"], 1000 * self.radar_config["range"], 1000)
        }

        # domain area indicator (radar field circle)
        x_grid, y_grid = np.meshgrid(self.coords["x"], self.coords["y"])
        self.domain_indicator = np.sqrt(((x_grid - self.radar_config["bng_coords"][0]) * 0.001) ** 2 + ((y_grid - self.radar_config["bng_coords"][1]) * 0.001) ** 2).round(0).astype(int) <= self.radar_config["range"]
        
        # fields
        for attr in ["sigma_e", "estimation_variance", "rain_polar", "atten_rain_polar", "pia_rain_polar"
                     "atten_rain_xy", "pia_rain_xy", "ref_xy", "atten_ref_xy", "pia_ref_xy", "atten_post_ref_xy",
                     "ref_polar", "atten_ref_polar", "atten_post_ref_polar", "pia_ref_polar", "noise_polar",
                     "pia_ref_polar", "noise_polar", "noise_xy"]:
            setattr(self, attr, None)
            
    def get_estimation_variance(self,
                                radar_config=None,
                                cloud_config=None,
                                vertical_vario=None):
        
        if radar_config is not None:
            self.radar_config = radar_config
        if cloud_config is not None:
            self.cloud_config = cloud_config
        if vertical_vario is not None:
            self.vertical_vario = vertical_vario
        
        # calculate estimation variance
        radar = {}
        bins = np.arange(100) #self.coords["bins"]  # distances from radar
        curv = 6371 * (1 - np.cos(np.arcsin(bins / 6371)))  # curvature of the earth

        for i, el in enumerate(self.radar_config["elevs"]):
            el_low = curv + self.radar_config["height"] + bins * np.tan(np.radians(el - 0.5))  # beam minimum
            el_high = curv + self.radar_config["height"] + bins * np.tan(np.radians(el + 0.5))  # beam maximum
            el_high[el_high > self.cloud_config["bright_band"]] = self.cloud_config["bright_band"]  # clip at bright band
            el_high[el_low > self.cloud_config["bright_band"]] = np.nan  # clip above bright band
            el_low[el_low > self.cloud_config["bright_band"]] = np.nan
            radar[str(i)] = {}
            radar[str(i)]["min"] = el_low
            radar[str(i)]["max"] = el_high

        # calculate variogram over whole domain
        hs = np.arange(self.cloud_config["ground"], self.cloud_config["cloud_top"], self.vertical_vario["interval_size"])  # discetize heights

        dists = []
        for j in hs:
            for k in hs:
                dists.append(abs(j - k))
        dists = np.array(dists)
        def exponential(h, r, c0, b=0):
            a = r / 3
            return b + c0 * (1. - np.exp(-(h / a)))

        gamma_bar_V = np.nanmean(exponential(dists, r=self.vertical_vario["range"], c0=self.vertical_vario["sill"], b=self.vertical_vario["nugget"]))
        gamma_bar_v = np.full(bins.shape[0], np.nan)

        # calculate variogram for rays
        for i in range(len(bins)):

            visible = []

            for key in radar.keys():
                r_min = radar[key]["min"][i]
                r_max = radar[key]["max"][i]

                if not np.isnan(r_min):
                    visible.extend(hs[(hs >= r_min) & (hs < r_max)])

            dists = []
            for j in visible:
                for k in visible:
                    dists.append(abs(j - k))
            dists = np.array(dists)
            gammas = exponential(dists, r=self.vertical_vario["range"], c0=self.vertical_vario["sill"], b=self.vertical_vario["nugget"])
            gamma_bar_v[i] = np.nanmean(gammas)
            
        self.estimation_variance = gamma_bar_V - gamma_bar_v
        self.sigma_e = np.full(self.rain_xy[0].shape, np.nan)
        x_grid, y_grid = np.meshgrid(self.coords["x"], self.coords["y"])
        dists = np.sqrt(((x_grid - self.radar_config["bng_coords"][0]) * 0.001) ** 2 + ((y_grid - self.radar_config["bng_coords"][1]) * 0.001) ** 2).round(0).astype(int)

        for i in range(100):
            self.sigma_e[dists == i] = self.estimation_variance[i] ** 0.5
            
        # print("Estimation variance complete.")
        
    def simulate_ref(self,
                     radar_config=None, # check these
                     cloud_config=None,
                     vertical_vario=None,
                     ref_atten=None,
                     rain_atten=None):
       
    
        for attr in [radar_config, cloud_config, vertical_vario, ref_atten, rain_atten]:
            if attr is not None:
                setattr(self, f'{attr=}'.split('=')[0], attr)
                
          
        polar_dim = (self.dur, self.radar_config["n_thetas"], self.radar_config["n_bins"])
        cart_dim = (self.dur, self.domain[1], self.domain[0])
        
        # reproject rainfall to polar
        x_grid, y_grid = np.meshgrid(self.coords["x"], self.coords["y"])
        grid_xy = np.vstack((x_grid.ravel(), y_grid.ravel())).transpose()
        rs, thetas = np.meshgrid(self.coords["bins"], self.coords["thetas"])
        xs = self.radar_config["bng_coords"][0] + 1000 * rs * np.cos(np.radians(thetas))
        ys = self.radar_config["bng_coords"][1] + 1000 * rs * np.sin(np.radians(thetas))
        new_grid = np.stack([xs, ys]).transpose()
        
        self.rain_polar = np.full(polar_dim, np.nan)
        for t in range(self.dur):
            self.rain_polar[t] = wrl.ipol.cart_to_irregular_interp(
                grid_xy, 
                self.rain_xy[t].flatten(), 
                new_grid).transpose()      
        self.rain_polar[self.rain_polar < 0] = 0
        
        self.pia_rain_polar = np.zeros(polar_dim)
        if self.rain_atten is not None:    
            self.rain_atten["gate_length"] = self.radar_config["bin_width"]
            
            # estimate attenuation
            for i in range(self.dur):
                self.pia_rain_polar[i] = atten_r.correct_attenuation_constrained(
                    self.rain_polar[i],
                    b_min=self.rain_atten["b_min"],
                    b_max=self.rain_atten["b_max"],
                    a_min=self.rain_atten["a_min"],
                    a_max=self.rain_atten["a_max"],
                    n_a=self.rain_atten["n_a"],
                    n_b=self.rain_atten["n_b"],
                    gate_length=self.rain_atten["gate_length"],
                    constraints=self.rain_atten["constraints"],
                    constraint_args=self.rain_atten["constraint_args"]
                )
            
            self.atten_rain_polar = np.zeros(polar_dim)
            self.atten_rain_polar[self.pia_rain_polar <= self.rain_polar] = (self.rain_polar - self.pia_rain_polar)[self.pia_rain_polar <= self.rain_polar]
            # print("Advection effects (rainfall) estimated.")
        else:
            self.atten_rain_polar = self.rain_polar
            # print("No advection effects (rainfall) estimated.")
            
        # simulate noise field
        if self.noise is not None:
            if self.noise["advected"]:
                noise_domain = (
                    self.dur, 
                    self.nu_shifts[1].max() - self.nu_shifts[1].min() + self.domain[1], 
                    self.nu_shifts[0].max() - self.nu_shifts[0].min() + self.domain[0])
                if not self.noise["temporal"]:
                    pass # print("Advected fields must have temporal dimension.")
            else:
                if self.noise["temporal"]:
                    noise_domain = (self.dur, self.domain[1], self.domain[0])
                else:
                    noise_domain = (self.domain[1], self.domain[0])
            
            self.noise_xy = np.zeros(cart_dim)
            noise_covariance_model = str(self.noise["nugget"]) + " Nug(0) + " + str(self.noise["sill"] - self.noise["nugget"]) + " " + self.noise["model"] + "(" + str(self.noise["range"]) + ")"
            srf = sp_sim.SpectralRandomField(noise_domain, noise_covariance_model, periodic=False)
            noise_raw = np.exp(self.noise["mu"] + self.noise["sigma"] * srf.new_simulation())
                
            
            if self.noise["advected"]:
                for t in range(self.dur):
                    field = noise_raw[
                        t,
                        self.nu_shifts[1, t] : self.nu_shifts[1, t] + self.domain[1],
                        self.nu_shifts[0, t] : self.nu_shifts[0, t] + self.domain[0]]
                    self.noise_xy[t, :, :] = field 
            else:
                if not self.noise["temporal"]:
                    for t in range(self.dur):
                        self.noise_xy[t, :, :] = noise_raw    
                else: 
                    self.noise_xy = noise_raw 
                    
            # print("Noise field estimated.")
        else: 
            pass # print("No noise field estimated.")
        
        if self.vertical_vario is not None:    
            if self.sigma_e is None:
                self.get_estimation_variance()
                
        # convert to reflectivity
        if isinstance(self.dsd_a, dict):
            a_covariance_model = str("1 " + self.dsd_a["model"] + "(" + str(self.dsd_a["range"]) + ")")
            srf = sp_sim.SpectralRandomField((self.domain[1], self.domain[0]), a_covariance_model, periodic=False)
            a_noise = self.dsd_a["sigma"] * srf.new_simulation()
            a_mu = self.dsd_a["mu"]
            a = a_mu + a_noise * (1 + self.sigma_e)
            
        elif (isinstance(self.dsd_a, float) | isinstance(self.dsd_a, int)):
            a_mu = self.dsd_a # specified estimation constant
            a = a_mu * (1 + self.sigma_e)
        else:
            a_mu = 200 # standard estimation constant
            a = a_mu * (1 + self.sigma_e)
        
        self.a_xy = a
            
        if isinstance(self.dsd_b, dict):
            if self.dsd_b["params"] is None:
                rv = eval("scipy.stats." + self.dsd_b["dist"] + "(loc=self.dsd_b['mu'], scale=self.dsd_b['sigma']" + ")")
            else:
                rv = eval("scipy.stats." + self.dsd_b["dist"] + "(self.dsd_b['params'], loc=self.dsd_b['mu'], scale=self.dsd_b['sigma']" + ")")
            b = rv.rvs(size=1)
        elif (isinstance(dsd_b, float) | isinstance(dsd_b, int)):
            b = dsd_b # specified estimation constant
        else:
            b = 1.6 # standard estimation constant
        
        
        self.b = b 
        
        # polar a
        if len(self.a_xy.shape) == 2:
            self.a_polar = wrl.ipol.cart_to_irregular_interp(
                grid_xy, 
                self.a_xy.flatten(), 
                new_grid).transpose()   
        else:
            self.a_polar = np.full(polar_dim, np.nan)
            for t in range(self.dur):
                self.a_polar[t] = wrl.ipol.cart_to_irregular_interp(
                    grid_xy, 
                    self.a_xy[t].flatten(), 
                    new_grid).transpose()   
        
        # polar noise
        if len(self.noise_xy.shape) == 2:
            self.noise_polar = wrl.ipol.cart_to_irregular_interp(
                grid_xy, 
                self.noise.flatten(), 
                new_grid).transpose()   
        else:
            self.noise_polar = np.full(polar_dim, np.nan)
            for t in range(self.dur):
                self.noise_polar[t] = wrl.ipol.cart_to_irregular_interp(
                    grid_xy, 
                    self.noise_xy[t].flatten(), 
                    new_grid).transpose()  
                 
        # conver to reflectivity
        # print("Converting to reflectivity.")
        Z = 10 * np.log10(self.a_xy * (self.rain_xy + self.noise_xy) ** self.b)
        self.ref_xy = np.full(Z.shape, -32)
        self.ref_xy[Z >= -32] = Z[Z >= -32]
        #self.ref_xy[Z < -32] = -32
        
        self.ref_polar = np.full(polar_dim, -32.)
        self.pia_ref_polar = np.zeros(polar_dim)
        
        Z = 10 * np.log10(self.a_polar * (self.atten_rain_polar + self.noise_polar) ** self.b)
        self.atten_ref1_polar = np.full(Z.shape, -32)
        self.atten_ref1_polar[Z >= -32] = Z[Z >= -32]
        #self.ref_xy[Z < -32] = -32
        
        # apply attenuation effects
        if self.ref_atten is not None:
            self.ref_atten["gate_length"] = self.radar_config["bin_width"]
            for t in range(self.dur):
                self.ref_polar[t] = wrl.ipol.cart_to_irregular_interp(grid_xy, self.ref_xy[t], new_grid).transpose()

                # add in attenuation effects
                if self.ref_atten is not None:
                    self.pia_ref_polar[t] = atten.correct_attenuation_constrained(
                        self.ref_polar[t],
                        b_min=self.ref_atten["b_min"],
                        b_max=self.ref_atten["b_max"],
                        a_min=self.ref_atten["a_min"],
                        a_max=self.ref_atten["a_max"],
                        n_a=self.ref_atten["n_a"],
                        n_b=self.ref_atten["n_b"],
                        gate_length=self.ref_atten["gate_length"],
                        constraints=self.ref_atten["constraints"],
                        constraint_args=self.ref_atten["constraint_args"])

            z_atten = self.ref_polar - self.pia_ref_polar
        
            self.atten_ref_polar = np.full(z_atten.shape, -32)
            self.atten_ref_polar[z_atten >= -32.] = z_atten[z_atten >= -32.]
        else:
            self.atten_ref_polar = self.ref_polar

        # grid reflectivity
        # print("Gridding polar fields.")
        # create polar grid in lon/lat
        polargrid = np.meshgrid(self.coords["bins"], self.coords["thetas"])
        coords, rad = wrl.georef.spherical_to_xyz(
            polargrid[0], 
            polargrid[1], 
            1, 
            (self.radar_config["latlon_coords"][0], self.radar_config["latlon_coords"][1], self.radar_config["height"])
        )
        # convert to BNG
        rs, thetas = np.meshgrid(self.coords["bins"], self.coords["thetas"])
        xs = self.radar_config["bng_coords"][0] + 1000 * rs * np.cos(np.radians(thetas))
        ys = self.radar_config["bng_coords"][1] + 1000 * rs * np.sin(np.radians(thetas))
        x_grid, y_grid = np.meshgrid(self.coords["x"], self.coords["y"])

        src = np.stack([xs.flatten(), ys.flatten()]).transpose()
        trg = np.vstack((x_grid.ravel(), y_grid.ravel())).transpose()
        
        for to_grid in ["pia_ref_polar", "ref_polar", "atten_ref_polar", "atten_rain_polar", "atten_ref1_polar"]:

            data = getattr(self, to_grid)
            gridded = np.full([self.dur, self.domain[1], self.domain[0]], np.nan)
            for t in range(self.dur):
                gridded_t = wrl.comp.togrid(
                    src=src, 
                    trg=trg, 
                    radius=1000 * (self.coords["bins"].max() + self.radar_config["bin_width"] / 2), 
                    center=self.radar_config["bng_coords"],
                    data=data[t].ravel(),
                    interpol=wrl.ipol.Nearest).reshape([self.domain[1], self.domain[0]])
                gridded[t, self.domain_indicator] = gridded_t[self.domain_indicator]
            setattr(self, to_grid.split("polar")[0] + "xy", gridded)