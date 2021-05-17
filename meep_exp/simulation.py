import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd



C = 299_792_458



def fresnel(freq, neffe, zeffe, d):

    n1 = np.ones(freq.shape) * 1.0
    n2 = neffe
    n3 = n1

    k2 = 2.0 * np.pi * freq * n2 / C

    Z1 = 1 / (8.854e-12 * C)
    Z2 = zeffe
    Z3 = 1 / (8.854e-12 * C)

    t12 = 2 * Z2 / ( Z1 + Z2 )
    t23 = 2 * Z3 / ( Z2 + Z3 )

    r12 = ( Z2 - Z1 ) / ( Z1 + Z2 )
    r23 = ( Z3 - Z2 ) / ( Z2 + Z3 )

    P2 = np.exp( 1j * k2 * d )

    t123 = ( t12 * t23 * P2 ) / ( 1 + r12 * r23 * P2**2 )
    r123 = ( r12 + r23 * P2**2 ) / ( 1 + r12 * r23 * P2**2 )

    return t123, r123


class Simulation():

    def __init__(
        self,
        # Source frequency settings.
        f_center, f_width, source_center,
        # Simulation field setting in tuple (in um).
        field_size, resolution,
        # PML setting.
        pml_width,
        # Medium setting.
        medium, medium_width,
        # Monitor position.
        monitor_position,
        # Simulation time.
        sim_time=100e3, n_freq=300,
    ):

        self.sim_time = sim_time
        self.n_freq = n_freq
        self.f_center = 1/((C/f_center)*1e6)
        self.f_width = 1/((C/f_width)*1e6)

        self.medium = medium
        self.medium_width = medium_width

        self.field = field_size

        self.refl_straight = None

        source = [
            mp.Source(
                mp.GaussianSource(
                    self.f_center, fwidth=self.f_width, is_integrated=True
                ),
                component=mp.Ex,
                center=mp.Vector3(0, 0, source_center),
                size=mp.Vector3(x=self.field[0], y=self.field[0]),
            ),
        ]

        self.simulator = lambda geo: \
            mp.Simulation(
                cell_size=mp.Vector3(*field_size),
                boundary_layers=[
                    mp.PML(pml_width, direction=mp.Z)
                ],
                geometry=geo,
                sources=source,
                resolution=resolution,
                k_point=mp.Vector3()
            )

        self.tran_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, monitor_position), 
            direction=mp.Y
        )
        self.refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, -monitor_position), 
            direction=mp.Y,
            weight=-1
        )


    def get_tran_incident(self):

        sim = self.simulator([])
        tran_monitor = sim.add_flux(
            self.f_center, self.f_width, 
            self.n_freq, self.tran_fr
        )
        refl_monitor = sim.add_flux(
            self.f_center, self.f_width,
            self.n_freq, self.refl_fr
        )
        sim.run(until=self.sim_time)

        self.refl_straight = sim.get_flux_data(refl_monitor)

        return np.array(sim.get_flux_data(tran_monitor).E[:self.n_freq])


    def get_refl_incident(self, debug=False):

        geometry = [
            mp.Block(
                size=mp.Vector3(mp.inf, mp.inf, self.medium_width),
                center=mp.Vector3(0,0,0),
                material=mp.perfect_electric_conductor
            ),
        ]
        sim = self.simulator(geometry)
        refl_monitor = sim.add_flux(
            self.f_center, self.f_width, 
            self.n_freq, self.refl_fr
        )
        sim.load_minus_flux_data(refl_monitor, self.refl_straight)
        sim.run(until=self.sim_time)

        if debug:
            sim.plot2D(
                fields=mp.Ex, 
                output_plane=mp.Volume(
                    center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(self.field[0], 0, self.field[2])
                )
            )
            plt.savefig('Ex_xz.png')

        return np.array(sim.get_flux_data(refl_monitor).E[:self.n_freq])


    def get_signal(self):

        trans = []
        refls = []

        for angle in [0, 90]:

            sim = self.simulator(self.medium[angle])
            tran_monitor = sim.add_flux(
                self.f_center, self.f_width, 
                self.n_freq, self.tran_fr
            )
            refl_monitor = sim.add_flux(
                self.f_center, self.f_width, 
                self.n_freq, self.refl_fr
            )
            sim.load_minus_flux_data(refl_monitor, self.refl_straight)
            sim.run(until=self.sim_time)

            trans.append(
                sim.get_flux_data(tran_monitor).E[:self.n_freq]
            )
            refls.append(
                sim.get_flux_data(refl_monitor).E[:self.n_freq]
            )

        return \
            np.array(trans), \
            np.array(refls), \
            np.array(mp.get_flux_freqs(tran_monitor))


    def run(self, tran_inc=None, refl_inc=None, refl_straight=None):

        self.refl_straight = np.load(refl_straight) if refl_straight else None
        tran_incidnet = np.load(tran_inc) if tran_inc and refl_straight else self.get_tran_incident()
        refl_incident = np.load(refl_inc) if refl_inc else self.get_refl_incident()
        tran_signal, refl_signal, freq = self.get_signal()

        self.freq = C/(1/freq*1e-6)
        self.S11 = refl_signal / refl_incident
        self.S11_0 = self.S11[0]
        self.S11_90 = self.S11[1]
        self.S21 = tran_signal / tran_incidnet
        self.S21_0 = self.S21[0]
        self.S21_90 = self.S21[1]
        self.k = 2*np.pi*self.freq / C

        self.Neff, self.Zeff, self.mu, self.eps = self.extract_params(
            self.S11, self.S21, self.medium_width*1e-6, self.freq
        )
        # self.Neff = self.Neff.real + 1j * np.abs(self.Neff.imag)
        self.deltaN = self.Neff[0].real - self.Neff[1].real

        self.validation()

        self.incidents = np.array([
            tran_incidnet,
            refl_incident,
            self.refl_straight
        ])

    
    def validation(self):

        N = self.Neff.real + 1j*np.abs(self.Neff.imag)

        tran, refl = fresnel(
            self.freq, N, 
            self.Zeff, self.medium_width*1e-6
        )

        self.fresnel_tran = tran
        self.fresnel_refl = refl


    def extract_params(self, S11, S21, d, freq):

        imp_0 = 1 / (8.854e-12 * C)
        imp = imp_0 * np.sqrt(((S11 + 1)**2 - S21**2) / ((S11 - 1)**2 - S21**2))
        Z = S21 * ( imp + imp_0 ) / ( ( imp + imp_0 ) - S11 * ( imp - imp_0 ) )
        k = 1j / d * np.log(Z)
        mu = k * imp / (2*np.pi*freq)
        eps = k / (2*np.pi*freq * imp)
        n = C*np.sqrt(mu*eps)

        return n, imp, mu, eps


    def save_incidents(self):
        np.save('tran_incident.npy', self.incidents[0])
        np.save('refl_incident.npy', self.incidents[1])
        np.save('refl_straight.npy', self.incidents[2])


    def save_spectrm(self, prefix, freq_range):

        plt.figure()
        plt.plot(self.freq*1e-12, np.abs(self.S21_0)
                ** 2, 'r-', label='tran0')
        plt.plot(self.freq*1e-12, np.abs(self.S11_0)
                ** 2, 'b-', label='refl0')
        plt.plot(self.freq*1e-12, np.abs(self.S21_90)
                ** 2, 'r--', label='tran90')
        plt.plot(self.freq*1e-12, np.abs(self.S11_90)
                ** 2, 'b--', label='refl90')
        plt.ylim(0, 1)
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("Transmittance/Reflectance (-)")
        plt.legend()
        plt.savefig('{}_amp.png'.format(prefix))
        plt.close()

        plt.figure()
        plt.plot(self.freq*1e-12,
                np.angle(self.S21_0), 'r-', label='tran0')
        plt.plot(self.freq*1e-12,
                np.angle(self.S11_0), 'b-', label='refl0')
        plt.plot(self.freq*1e-12, np.angle(self.S21_90),
                'r--', label='tran90')
        plt.plot(self.freq*1e-12, np.angle(self.S11_90),
                'b--', label='refl90')
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("Phase shift (rad)")
        plt.legend()
        plt.savefig('{}_phase.png'.format(prefix))
        plt.close()

        plt.figure()
        plt.plot(self.freq*1e-12,
                 self.deltaN, 'r-')
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("$\Delta N$ (-)")
        plt.savefig('{}_deltaN.png'.format(prefix))
        plt.close()


        plt.figure()
        plt.plot(self.freq*1e-12, np.abs(self.fresnel_tran[0])
                 ** 2, 'r-', label='tran0')
        plt.plot(self.freq*1e-12, np.abs(self.fresnel_refl[0])
                 ** 2, 'b-', label='refl0')
        plt.plot(self.freq*1e-12, np.abs(self.fresnel_tran[1])
                 ** 2, 'r--', label='tran90')
        plt.plot(self.freq*1e-12, np.abs(self.fresnel_refl[1])
                 ** 2, 'b--', label='refl90')
        plt.ylim(0, 1)
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("Transmittance/Reflectance (-)")
        plt.legend()
        plt.savefig('{}_fresnel_amp.png'.format(prefix))
        plt.close()

        plt.figure()
        plt.plot(self.freq*1e-12,
                 np.angle(self.fresnel_tran[0]), 'r-', label='tran0')
        plt.plot(self.freq*1e-12,
                 np.angle(self.fresnel_refl[0]), 'b-', label='refl0')
        plt.plot(self.freq*1e-12, np.angle(self.fresnel_tran[1]),
                 'r--', label='tran90')
        plt.plot(self.freq*1e-12, np.angle(self.fresnel_refl[1]),
                 'b--', label='refl90')
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("Phase shift (rad)")
        plt.legend()
        plt.savefig('{}_fresnel_phase.png'.format(prefix))
        plt.close()

    
    def save_data(self, prefix):

        with open('{}_data.csv'.format(prefix), 'w') as f:

            f.write(
                'freq, \
                S11_0.real,\
                S11_0.imag,\
                S11_90.real,\
                S11_90.imag,\
                S21_0.real,\
                S21_0.imag,\
                S21_90.real,\
                S21_90.imag,\
                z_0.real,\
                z_0.imag,\
                z_90.real,\
                z_90.imag,\
                n_0.real,\
                n_0.imag,\
                n_90.real,\
                n_90.imag\n'
            )

            for i in range(len(self.freq)):
                f.write(
                    '{}, \
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {},\
                    {}\n'.format(
                        self.freq[i],
                        self.S11_0[i].real,
                        self.S11_0[i].imag,
                        self.S11_90[i].real,
                        self.S11_90[i].imag,
                        self.S21_0[i].real,
                        self.S21_0[i].imag,
                        self.S21_90[i].real,
                        self.S21_90[i].imag,
                        self.Zeff[0,i].real,
                        self.Zeff[0,i].imag,
                        self.Zeff[1,i].real,
                        self.Zeff[1,i].imag,
                        self.Neff[0,i].real,
                        self.Neff[0,i].imag,
                        self.Neff[1,i].real,
                        self.Neff[1,i].imag,
                    )
                )




