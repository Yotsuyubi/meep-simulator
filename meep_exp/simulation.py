import meep as mp
import numpy as np
import matplotlib.pyplot as plt
import sys
import torch as th


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


def fresnel_torch(freq, neffe, zeffe, d):

    n2 = neffe

    k2 = 2.0 * np.pi * freq * n2 / C

    Z1 = 1 / (8.854e-12 * C)
    Z2 = zeffe
    Z3 = 1 / (8.854e-12 * C)

    t12 = 2 * Z2 / ( Z1 + Z2 )
    t23 = 2 * Z3 / ( Z2 + Z3 )

    r12 = ( Z2 - Z1 ) / ( Z1 + Z2 )
    r23 = ( Z3 - Z2 ) / ( Z2 + Z3 )

    P2 = th.exp( 1j * k2 * d )

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
        a=1e-6 # meep unit of length is defined as 1 [um].
    ):

        self.a = a
        self.resolution = resolution
        self.unit_pixel = self.a / self.resolution

        self.sim_time = self.to_meep_time(sim_time)
        self.n_freq = n_freq
        self.f_center = self.to_meep_freq(f_center)
        self.f_width = self.to_meep_freq(f_width)

        self.rr = np.abs(monitor_position) - medium_width/2
        self.rt = np.abs(monitor_position) - medium_width/2
        self.rs = np.abs(source_center) - medium_width/2
        self.d = medium_width

        self.medium = medium
        self.medium_width = self.to_meep_length(medium_width)

        self.sx = self.to_meep_length(field_size[0])
        self.sy = self.to_meep_length(field_size[1])
        self.sz = self.to_meep_length(field_size[2])

        self.field = (self.sx, self.sy, self.sz)

        self.refl_straight = None

        source = [
            mp.Source(
                mp.GaussianSource(
                    self.f_center, fwidth=self.f_width, is_integrated=True
                ),
                component=mp.Ex,
                center=mp.Vector3(0, 0, self.to_meep_length(source_center)),
                size=mp.Vector3(x=self.field[0], y=self.field[0]),
            ),
        ]

        self.simulator = lambda geo: \
            mp.Simulation(
                cell_size=mp.Vector3(*self.field),
                boundary_layers=[
                    mp.PML(self.to_meep_length(pml_width), direction=mp.Z)
                ],
                geometry=geo,
                sources=source,
                resolution=resolution,
                k_point=mp.Vector3()
            )

        self.tran_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, self.to_meep_length(monitor_position)), 
            direction=mp.Y
        )
        self.refl_fr = mp.FluxRegion(
            center=mp.Vector3(0, 0, self.to_meep_length(-monitor_position)), 
            direction=mp.Y,
            weight=-1
        )


    def to_meep_length(self, length):
        return length / self.a

    def to_meep_time(self, time):
        return time * C / self.a

    def to_meep_freq(self, freq):
        return freq * self.a / C

    def to_si_length(self, length):
        return length * self.a

    def to_si_freq(self, freq):
        return freq * C / self.a


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


    def get_refl_incident(self, debug=True):

        geometry = [
            mp.Block(
                size=mp.Vector3(self.sx, self.sy, self.medium_width),
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

            sim.plot2D(
                fields=mp.Ex, 
                output_plane=mp.Volume(
                    center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(self.field[0], 0, self.field[2])
                ),
                frequency=self.f_center
            )
            plt.savefig('Ex_xy{}.png'.format(angle))
            sim.plot2D(
                fields=mp.Ex, 
                output_plane=mp.Volume(
                    center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(0, self.field[1], self.field[2])
                )
            )
            plt.savefig('Ex_yz{}.png'.format(angle))
            sim.plot2D(
                fields=mp.Ex, 
                output_plane=mp.Volume(
                    center=mp.Vector3(0,0,0), 
                    size=mp.Vector3(self.field[0], self.field[1], 0)
                )
            )
            plt.savefig('Ex_xz{}.png'.format(angle))


        return \
            np.array(trans), \
            np.array(refls), \
            np.array(mp.get_flux_freqs(tran_monitor))


    def run(self, tran_inc=None, refl_inc=None, refl_straight=None):

        self.refl_straight = np.load(refl_straight) if refl_straight else None
        tran_incidnet = np.load(tran_inc) if tran_inc and refl_straight else self.get_tran_incident()
        refl_incident = np.load(refl_inc) if refl_inc else self.get_refl_incident()
        tran_signal, refl_signal, freq = self.get_signal()

        self.freq = self.to_si_freq(freq)[80:250]
        self.k = 2*np.pi*self.freq / C
        self.S11 = (-1 * refl_signal / refl_incident)[:, 80:250]
        self.S11_0 = self.S11[0]
        self.S11_90 = self.S11[1]
        self.S21 = (tran_signal / tran_incidnet)[:, 80:250]
        self.S21_0 = self.S21[0]
        self.S21_90 = self.S21[1]

        self.Neff, self.Zeff = self.fit(self.S11, self.S21, self.d, self.freq)

        # p = self.get_p()
        # print(p)

        # self.Neff, self.Zeff, self.mu, self.eps = self.extract_params(
        #     self.S11, self.S21, self.d, self.freq, m=p
        # )
        self.deltaN = self.Neff[0].real - self.Neff[1].real

        self.validation()

        self.incidents = np.array([
            tran_incidnet,
            refl_incident,
            self.refl_straight
        ])

    
    def validation(self):

        N = self.Neff

        tran, refl = fresnel(
            self.freq, N, 
            self.Zeff, self.d
        )

        self.fresnel_tran = tran * np.exp( -1j * self.k * self.d )
        self.fresnel_refl = refl


    def extract_params(self, S11, S21, d, freq, m=0):

        imp_0 = 1 / (8.854e-12 * C)

        S21 = S21 * np.exp( 1j * self.k * d )
        S11 = S11

        imp = imp_0 * np.sqrt(((S11 + 1)**2 - S21**2) / ((1 - S11)**2 - S21**2))
        Z_A = S21 * ( imp + imp_0 ) / ( ( imp + imp_0 ) - S11 * ( imp - imp_0 ) )
        k = 1j / d * ( np.log(np.abs(Z_A)) + 1j*( np.angle(Z_A) + 2*m*np.pi ) )
        mu = k * imp / (2*np.pi*freq)
        eps = k / (2*np.pi*freq * imp)
        n = C*np.sqrt(mu*eps)
        
        # n_imag = (1 / (self.k * d) * np.arccos((1 - S11**2 + S21**2) / (2 * S21))).imag
        # n_real = (1 / (self.k * d) * np.arccos((1 - S11**2 + S21**2) / (2 * S21))).real + 2*m*np.pi / (self.k * d)
        # n = n_real + 1j*n_imag
        # imp = imp_0 * np.sqrt(((S11 + 1)**2 - S21**2) / ((S11 - 1)**2 - S21**2))
        # eps = n / (imp/imp_0)
        # mu = n * (imp/imp_0)

        return n, imp, mu, eps


    def extract_params_torch(self, S11, S21, d, freq, p):

        # p = th.floor(th.abs(p)) * th.sign(p)
        p = th.trunc(th.abs(p)) * th.sign(p)

        imp_0 = 1 / (8.854e-12 * C)
        imp = imp_0 * th.sqrt(((S11 + 1)**2 - S21**2) / ((1 - S11)**2 - S21**2))
        Z_A = S21 * ( imp + imp_0 ) / ( ( imp + imp_0 ) - S11 * ( imp - imp_0 ) )
        k = 1j / d * ( th.log(th.abs(Z_A)) + 1j*( th.angle(Z_A) + 2*p*np.pi ) )
        mu = k * imp / (2*np.pi*freq)
        eps = k / (2*np.pi*freq * imp)
        n = C*th.sqrt(mu*eps)

        return n, imp, mu, eps


    def get_p(self):

        si_thicker_width = self.d+100e-6
        self.medium_width = self.to_meep_length(si_thicker_width)
        self.medium[0][0].size = mp.Vector3(self.sx, self.sy, self.medium_width)
        self.medium[90][0].size = mp.Vector3(self.sx, self.sy, self.medium_width)

        tran_incidnet = self.get_tran_incident()
        refl_incident = self.get_refl_incident()
        tran_signal, refl_signal, freq = self.get_signal()

        S11 = -1 * refl_signal / refl_incident
        S21 = tran_signal / tran_incidnet

        _, _, _, _, Z_1 = self.extract_params(self.S11, self.S21, self.d, self.freq)
        _, _, _, _, Z_2 = self.extract_params(S11, S21, si_thicker_width, self.freq)
        p = ( ( si_thicker_width * np.angle( Z_1 ) - self.d * np.angle( Z_2 ) ) / ( 2 * np.pi * ( self.d - si_thicker_width ) ) )
        self.medium_width = self.to_meep_length(self.d)

        return np.fix(p)
        

    def fit(self, S11_, S21_, d_, freq_, err_th=1, itr=5e4):

        n = th.ones(2, freq_.shape[0], requires_grad=True)
        k = th.ones(2, freq_.shape[0], requires_grad=True)
        imp_real = th.ones(2, freq_.shape[0], requires_grad=True)
        imp_img = th.ones(2, freq_.shape[0], requires_grad=True)

        optim = th.optim.Adam([n, k, imp_real, imp_img], lr=1e-3)

        for _ in range(int(itr)):
            
            optim.zero_grad()

            freq = th.tensor(freq_)
            d = th.tensor(d_)
            k0 = 2*np.pi*freq / C
            S11 = th.tensor(S11_)
            S21 = th.tensor(S21_) * th.exp( 1j * k0 * d )
            imp = th.abs(imp_real) + 1j*imp_img

            n2 = th.abs(n) + 1j*th.abs(k)
            k2 = 2.0 * np.pi * freq * n2 / C
            Z1 = 1
            Z2 = imp
            Z3 = 1
            t12 = 2 * Z2 / ( Z1 + Z2 )
            t23 = 2 * Z3 / ( Z2 + Z3 )
            r12 = ( Z2 + -1* Z1 ) / ( Z1 + Z2 )
            r23 = ( Z3 + -1* Z2 ) / ( Z2 + Z3 )
            P2 = th.exp( 1j * k2 * d )
            t123 = ( t12 * t23 * P2 ) / ( 1 + r12 * r23 * P2**2 )
            r123 = ( r12 + r23 * P2**2 ) / ( 1 + r12 * r23 * P2**2 )

            loss = th.nn.MSELoss()(r123.real, S11.real) \
                    + th.nn.MSELoss()(r123.imag, S11.imag) \
                    + th.nn.MSELoss()(t123.real, S21.real) \
                    + th.nn.MSELoss()(t123.imag, S21.imag) \

            loss.backward()

            optim.step()

        print(loss.item())

        return (th.abs(n) + 1j*th.abs(k)).detach().numpy(), (th.abs(imp_real) + 1j*imp_img).detach().numpy()*1 / (8.854e-12 * C)

    # def fit(self, S11, S21, d, freq, err_th=1, itr=100e3):

    #     n = th.ones(freq.shape, requires_grad=True)
    #     k = th.ones(freq.shape, requires_grad=True)
    #     eta_real = th.ones(freq.shape, requires_grad=True)
    #     eta_imag = th.ones(freq.shape, requires_grad=True)

    #     freq = th.tensor(freq)
    #     d = th.tensor(d)
    #     S11 = th.tensor(S11)
    #     S21 = th.tensor(S21)

    #     optim = th.optim.Adam([n, k, eta_real, eta_imag], lr=1e-3)

    #     for _ in range(int(itr)):
            
    #         optim.zero_grad()

    #         n_hat = th.abs(n) + 1j*th.abs(k)
    #         eta = th.abs(eta_real) + 1j*eta_imag

    #         S11_hat, S21_hat = fresnel_torch(freq, n_hat, eta, d)
    #         loss = th.nn.L1Loss()(S11_hat, S11) 
    #         loss += th.nn.L1Loss()(S21_hat, S21)
    #         loss.backward()

    #         optim.step()

    #     print(loss.item())

    #     return n_hat, eta


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
                (np.angle(self.S21_0)), 'r-', label='tran0')
        plt.plot(self.freq*1e-12,
                (np.angle(self.S11_0)), 'b-', label='refl0')
        plt.plot(self.freq*1e-12,(np.angle(self.S21_90)),
                'r--', label='tran90')
        plt.plot(self.freq*1e-12, (np.angle(self.S11_90)),
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
        plt.plot(self.freq*1e-12, self.Neff[0].real, 'r-', label='n0')
        plt.plot(self.freq*1e-12, self.Neff[0].imag, 'r--', label='k0')
        plt.plot(self.freq*1e-12, self.Neff[1].real, 'b-', label='n90')
        plt.plot(self.freq*1e-12, self.Neff[1].imag, 'b--', label='k90')
        # plt.ylim(0, 2)
        plt.xlim(*freq_range)
        plt.xlabel("Freq (THz)")
        plt.ylabel("$\hat{n}$")
        plt.legend()
        plt.savefig('{}_n.png'.format(prefix))
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
                ( np.angle(self.fresnel_tran[0]) ), 'r-', label='tran0')
        plt.plot(self.freq*1e-12,
                ( np.angle(self.fresnel_refl[0]) ), 'b-', label='refl0')
        plt.plot(self.freq*1e-12, ( np.angle(self.fresnel_tran[1]) ),
                 'r--', label='tran90')
        plt.plot(self.freq*1e-12, ( np.angle(self.fresnel_refl[1]) ),
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
                n_90.imag,\
                \n'
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
                    {},\
                    \n'.format(
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
                        self.Neff[1,i].imag
                    )
                )




