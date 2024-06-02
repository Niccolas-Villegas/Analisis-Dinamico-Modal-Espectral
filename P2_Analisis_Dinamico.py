import numpy as np   
from numpy.linalg import inv, eig
from math import *
import matplotlib.pyplot as plt
import pandas as pd
from units import *

class Analisis_Modal:
    def __init__(self, m, k, h):
        self.m = m
        self.k = k
        self.h = h
        
        self.M = self.Masa()
        self.K = self.K_general()
        self.w, self.T, self.x = self.Modos_formas()
        self.phi, self.I = self.Normalizacion(self.x)
        self.MPF , self.EM, self.PM = self.Factor_Part_Modal(self.phi, self.I)
        self.modes, self.results = self.Tablas(self.x, self.T, self.MPF, self.EM, self.PM)
    
    def Masa(self):
        n_mass = len(self.m)
        M = np.zeros((n_mass,n_mass))
        for i in range(n_mass):
            M[i][i] = round(self.m[i],2)
        return M
    
    def K_general(self):
        n_k = len(self.k)
        K = np.zeros((n_k,n_k))
        K[0][0] = self.k[0]
        for i in range(1,n_k):
            K[i-1:i+1, i-1:i+1] += np.array([[self.k[i], -self.k[i]],
                                             [-self.k[i], self.k[i]]])
        return K

    def Modos_formas(self):
        A = inv(self.M)@self.K
        eigenvalues, eigenvectors = eig(A)
        n_mod = eigenvalues.shape[0]
        w = np.sqrt(eigenvalues)
        for i in range(n_mod-1):
            for j in range(n_mod-1-i):
                if w[j] > w[j+1]:
                    w[j], w[j+1] = w[j+1], w[j]
                    eigenvectors[:,[j,j+1]] = eigenvectors[:,[j+1,j]]
        T = [round(2*pi/f,2) for f in w]
        return w, T, eigenvectors
    
    def Normalizacion(self, eigenvectors):
        n_x = eigenvectors.shape[0]
        phi = np.zeros((n_x,n_x))
        I = n_x*[1]
        for i in range(n_x):
            x = eigenvectors[:,i]
            phi[:,i] = (1/sqrt(x.T@self.M@x))*x
        return phi, np.array(I)
    
    def Factor_Part_Modal(self, phi, I):
        R = []
        Efc_mass = []
        Part_mass = []
        for i in range(phi.shape[0]):
            r = phi[:,i].T@self.M@I.T
            R.append(round(r,2))
            Efc_mass.append(round(r**2,2))       
        for i in range(len(R)):
            Part_mass.append(round(100*Efc_mass[i]/sum(Efc_mass),2))
        return R, Efc_mass, Part_mass
    
    def Graficos(self):
        n_gr = self.x.shape[0]
        basement = n_gr*[0]
        graph_modes = np.vstack((basement,self.x))
        hn = [0] + self.h
        for i in range(1,len(hn)):
            hn[i] += hn[i-1]
        fig, axs = plt.subplots(1, n_gr, figsize = (20,6))
        for i in range(n_gr):
            axs[i].plot(graph_modes[:,i], hn, 'b-o', (n_gr+1)*[0], hn, 'black')
            axs[i].grid('silver')
            axs[i].set_xlim([-1, 1])
            axs[i].set_ylim([0, max(hn)])
            axs[i].set_title(f'Modo {i+1} (T{i+1} = {self.T[i]}s)', size = 14)
        return plt.show()
    
    def Tablas(self, modes, T, MPF, EM, PM):
        # Tabla para los modos de vibracion
        fmode, cmode = modes.shape
        n_mode = [("Modo "+str(i+1)) for i in range(modes.shape[1])]
        story = {i: ("NIVEL "+str(fmode-i)) for i in range(fmode)}
        df_modes = pd.DataFrame(np.around(modes[::-1], 4), columns=n_mode)
        df_modes.rename(index=story, inplace=True)
             
        # Tabla para los resultados de periodo y masas participativas
        mode_col = [i+1 for i in range(cmode)]
        df_results = pd.DataFrame({"Modo": mode_col, "Periodo (s)": T,  "Factor Part. masa": MPF, "Masa Efectiva": EM, "Masa Participativa (%)": PM})
        df_results.index = ['']*len(df_results)
        return df_modes, df_results

class Analisis_Espectral(Analisis_Modal):
    def __init__(self, m, k, h, S_coeff, R, Tp, Tl):
        super().__init__(m, k, h)
        self.S_coeff = S_coeff
        self.R = R
        self.Tp = Tp
        self.Tl = Tl
        
        self.Sa, self.A = self.Aceleraciones(self.T, self.MPF, self.phi)
        self.Sd, self.D, self.Dr = self.Desplazamientos(self.w, self.Sa, self.MPF, self.phi)
        self.F, self.Fr = self.Fuerzas(self.M, self.A)
        self.V, self.Vr = self.Cortantes(self.F)
    
    def Aceleraciones(self, T, MPF, phi):
        Sa = []
        n_acc = len(T)
        A = np.zeros((n_acc,n_acc))
        for i in range(n_acc):
            if T[i] < self.Tp:
                C = 2.5
            elif self.Tp <= T[i] < self.Tl:
                C = 2.5*self.Tp/T[i]
            else:
                C = 2.5*(self.Tp*self.Tl)/(T[i]**2)
            Sa.append(self.S_coeff*C)
            A[:,i] = Sa[i]*MPF[i]*phi[:,i]  
        return Sa, A

    def Desplazamientos(self, w, Sa, MPF, phi):
        Sd = []
        n_disp = len(Sa)
        D = np.zeros((n_disp,n_disp))
        for i in range(n_disp):
            Sd.append(Sa[i]/(w[i]**2))
            D[:,i] = Sd[i]*MPF[i]*phi[:,i]/cm
        Dr = self.Combinacion_Modal(D)
        return Sd, D, Dr
    
    def Fuerzas(self, M, A):
        F = M@A*(1/tnf)
        Fr = self.Combinacion_Modal(F)
        return F, Fr
    
    def Cortantes(self, F):
        n_V = F.shape[0]
        V = np.zeros((n_V,n_V))
        V[0,:] = F[0,:]
        for i in range(1,n_V):
            V[i,:] = V[i-1,:] + F[i,:]
        Vr = self.Combinacion_Modal(V)
        return V, Vr

    def Combinacion_Modal(self, X):
        n_arr = X.shape[0]
        Abs = np.array((n_arr)*[0], dtype=np.float64)
        Sqr = np.array((n_arr)*[0], dtype=np.float64)
        for i in range(n_arr):
            Abs += np.abs(X[:,i])
            Sqr += np.square(X[:,i])
        Sqr = np.sqrt(Sqr)
        r = 0.25*Abs + 0.75*Sqr
        return r
    
    def Graficos(self):
        fig = plt.figure(figsize=(20,6))
        gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 2])
        
        # Grafico para desplazamietos por modo
        n_gr = self.x.shape[0]
        desp_modes = np.vstack((n_gr*[0],self.D))  
        hn = [0] + self.h
        legend_ax1 = []
        for i in range(1,len(hn)):
            hn[i] += hn[i-1]   
        ax1 = fig.add_subplot(gs[0])
        for i in range(n_gr):
            ax1.plot(desp_modes[:,i], hn,'-o')
            modes_max = np.round(np.max(desp_modes[:,i]),2)
            legend_ax1.append('Δ max = '+str(modes_max)+' cm')
        ax1.set_title('Desplazamientos por cada modo (cm)', size = 12)
        ax1.legend(legend_ax1, loc='lower right')
        ax1.grid("silver")
        ax1.set_ylim([0,max(hn)])
        
        # Grafico para desplazamietos reales (combinacion modal)
        # Considerando la estructura regular -> Factor = 0.75*R
        desp_real = (0.75*self.R)*np.hstack(([0],self.Dr))
        desp_max = np.round(np.max(desp_real),2)
        ax2 = fig.add_subplot(gs[1])
        ax2.plot(desp_real, hn, 'b-o', label='Δ max = '+str(desp_max)+' cm')
        ax2.set_title('Desplazamientos reales (cm)', size = 12)
        ax2.legend(loc='lower right')
        ax2.grid("silver")
        ax2.set_ylim([0,max(hn)])
        
        # Grafico para derivas
        drift = [0]
        for i in range(1,len(hn)):
            d = (desp_real[i] - desp_real[i-1])/((hn[i] - hn[i-1])/cm) * 1000
            drift.append(d)
        drift_max = np.round(max(drift),2)
        ax3 = fig.add_subplot(gs[2])
        ax3.plot(drift, hn, 'b-o', label='Drift max = '+str(drift_max)+' (‰)')
        ax3.set_title('Derivas (‰)', size = 12)
        ax3.legend(loc='lower right')
        ax3.grid("silver")
        ax3.set_ylim([0,max(hn)])
        
        # Grafico para cortantes reales (combinacion modal)
        stories = [i+1 for i in range(self.Vr.shape[0])]
        ax4 = fig.add_subplot(gs[3])
        ax4.barh(stories, self.Vr[::-1])
        for index, value in enumerate(self.Vr[::-1]):
            ax4.text(value, index+1, str(np.round(value,2)), va='center', ha='left', fontsize=10)
        ax4.set_title('Cortantes reales (ton)', size = 12)
        ax4.set_xlim([0, max(self.Vr)*1.2])
        
        return plt.show()