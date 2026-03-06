import numpy as np
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import find_peaks
from scipy.integrate import quad
from dataclasses import dataclass, astuple
from typing import NamedTuple, Self
import matplotlib.pyplot as plt

from functions import \
    fidelity_singlet, fidelity_triplet, \
    distribution_singlet, distribution_triplet, \
    distribution_ST

class DistributionSTParams(NamedTuple):
    """ Object immuable représentant le résultat d'un fit.
    """
    Ps: float
    tm_over_T1: float
    mus: float
    mut: float
    sigma: float

    def get_distributions(self, x):
        dis_singlet = distribution_singlet(x, self.mus, self.sigma)
        dis_triplet = distribution_triplet(
            x, self.tm_over_T1, self.mus, self.mut, self.sigma
        )
        return dis_singlet, dis_triplet

    def get_visibility(self, x, threshold):
        fid_sin = fidelity_singlet(threshold, self.mus, self.sigma)
        fid_tri = fidelity_triplet(threshold, self.tm_over_T1, self.mus, self.mut, self.sigma)
        vis = fid_sin + fid_tri - 1
        return fid_sin, fid_tri, vis

    def find_optimal_threshold(self, x):
        func = lambda x: (lambda s, t: s - t)(*self.get_distributions(x))
        threshold = root_scalar_safe(func, bracket=[self.mus-self.sigma, self.mut+self.sigma], method="brentq")
        return threshold

def root_scalar_safe(func, bracket, method="brentq"):
    """
    Test les signes des bornes.
    Calcul et retourne la racine si la fonction passe par zéro.
    Retourne None sinon.
    """
    a = func(bracket[0])
    b = func(bracket[1])
    if np.sign(a) == np.sign(b):
        return None
    return root_scalar(func, bracket=bracket, method="brentq").root


@dataclass
class DistributionSTFitResults:
    """ Objet contenant:
    - le résultat d'un fit
    - des valeurs calculés en fonction du fit

    À instancier avec la fonction .from_bins_hist
    """
    popt: DistributionSTParams = None
    threshold: float = None
    singlet_fidelity: float = None
    triplet_fidelity: float = None
    visibility: float = None

    def as_array(self, *arg_names):
        if len(arg_names) == 0:
            return np.array([*self.popt, *(astuple(self)[1:])])
        
        if len(arg_names) == 1:
            return eval(f"self.{arg_names[0]}")
        
        res = np.array([eval(f"self.{arg}") for arg in arg_names])
        return res


    @classmethod
    def from_bins_hist(
        cls,
        x_values, 
        histogram, 
        p0=None, 
        find_threshold=False,
        compute_visibility=False,
        verbosity=0
    ) -> Self:
        """
        Cette fonction ajuste la fonction `distribution_ST()` à un ensemble de données (normalisées) passé en arguments.

        Args:
            x_values: Points en x associés aux données de l'histogramme
            histogram: Histogramme de données sur lequel sera ajustée la fonction
            p0: Paramètres initiaux passés à la fonction `curve_fit()`. Si `None`, des paramètres initiaux seront estimés
                à partir des pics de l'histogramme identifiés par la fonction `find_peaks()`
            find_threshold: Demande à la fonction de trouver le threshold
            compute_visibility: Demande à la fonction de calculer la visibilité
            verbosity: Niveau de verbosité pour les messages de sortie
        
        Returns:
            Une instance de la classe de données `DistributionSTFitResults` contenant les résultats calculés dans cette fonction

        Niveaux de verbosité:
            0: Pas de message de sortie
            1: Affiche les paramètres trouvés par le fit
            2: Affichage du threshold et de la visibilité (si demandés)
            4: Affichage du fit et du threshold trouvé (si demandé)
            5: Affichage des paramètres initiaux pour aider à trouver le fit
            6: Affichage des distributions singulet et triplet individuelles
        """

        results = DistributionSTFitResults()
        norm_factor = (x_values[1] - x_values[0]) * histogram.sum()  # Facteur de normalisation pour que l'intégrale sur l'ensemble des valeurs soit 1.
        
        if compute_visibility:  # Il faut le threshold pour calculer la visibilité.
            find_threshold = True
        
        if verbosity >= 4:
            plt.figure(figsize=(6, 4))
            plt.plot(x_values, histogram, label="Histogram")

        # Paramètres initiaux de fit
        if p0 is None:
            peaks = find_peaks(histogram, height=0.05*histogram.max(), width=len(histogram)//100)[0][[0, -1]]  # Index des pics
            amps0 = histogram[peaks]  # Amplitudes des pics
            Ps0 = amps0[0] / amps0.sum()  # Probabilité initiale du singulet
            mus0, mut0 = x_values[peaks]  # Position des pics
            sigma0 = (mut0 - mus0) / 6  # Estimation initiale de sigma

            p0 = DistributionSTParams(Ps0, 1, mus0, mut0, sigma0)

            if verbosity >= 5:
                plt.plot(x_values, norm_factor * distribution_ST(x_values, *p0), label="Distribution before fit")
                plt.scatter(x_values[peaks], histogram[peaks], c="red", label="Peaks used as initial guess")
                plt.legend()

        bounds_min = (0, 0, x_values.min(), x_values.min(), 0)
        bounds_max = (1, np.inf, x_values.max(), x_values.max(), np.inf)

        # Fit de la distribution
        try:
            popt_unnamed, cov = curve_fit(distribution_ST, x_values, histogram / norm_factor, p0=p0, bounds=(bounds_min, bounds_max))
            popt = DistributionSTParams(*popt_unnamed)
            results.popt = popt
            if verbosity >= 1:
                print(f"Fit parameters: Ps={popt.Ps}, tm/T1={popt.tm_over_T1}, mus={popt.mus}, mut={popt.mut}, sigma={popt.sigma}")
        except:
            results.popt = DistributionSTParams(*([None]*5))
            if verbosity >= 1:
                print("Fit failed.")
            return results

        if verbosity >= 4:
            plt.plot(x_values, norm_factor * distribution_ST(x_values, *popt), label="Fitted distribution")
        
        if verbosity >= 6:
            plt.plot(x_values, norm_factor * popt.Ps * distribution_singlet(x_values, popt.mus, popt.sigma), label="Singlet distribution", ls=":")
            plt.plot(x_values, norm_factor * (1-popt.Ps) * distribution_triplet(x_values, popt.tm_over_T1, popt.mus, popt.mut, popt.sigma), label="Triplet distribution", ls=":")

        # Trouver le threshold
        if find_threshold:
            results.threshold = results.popt.find_optimal_threshold(x_values)

            if verbosity >= 2:
                print(f"Optimal threshold: {results.threshold}")

            if verbosity >= 4 and results.threshold is not None:
                plt.axvline(results.threshold, ls="--", c="red", label="Threshold")
        
        # Calculer la visibilité
        if compute_visibility and results.threshold is not None:
            fid_sin, fid_tri, vis = results.popt.get_visibility(x_values, results.threshold)
            results.singlet_fidelity = fid_sin
            results.triplet_fidelity = fid_tri
            results.visibility = vis

            if verbosity >= 2:
                print(f"Visibility: {vis}")

        if verbosity >= 4:
            plt.xlabel("X values (a.u.)")
            plt.ylabel("Counts (normalized)")
            plt.legend()
            plt.tight_layout()

        return results

def find_adaptative_thresholds(data, n_slices=1, resolution=1, long_axis=0, bins=401):
    """
    Fonction qui trouve un threshold adaptatif pour tenir compte du drift du signal durant une
    longue mesure.

    Params:
        data: matrice de données
        n_slices: nombre de tranches à utiliser pour calculer chaque threshold
        resolution: calculer le threshold tous les "resolution" tranches. Si la résolution est plus grande que 1, une interpolation linéaire est utilisée pour trouver les thresholds intermédiaires
        long_axis: axe le long duquel calculer le threshold
        bins: paramètre `bins` passé à la fonction `numpy.histogram()` pour calculer les histogrammes des tranches de données

    Returns:
        Array 1d de thresholds pour chaque tranche de données
    """
    # Mettre l'axe long en premier pour faciliter les calculs
    data_transposed = np.moveaxis(data, long_axis, 0)

    # Calculer les histogrammes pour chaque tranche de données
    if type(bins) == int:
        bins = np.linspace(data_transposed.min(), data_transposed.max(), bins+1)

    hist = np.apply_along_axis(lambda arr: np.histogram(arr, bins=bins)[0], 1, data_transposed)
    bins_center = (bins[1:] + bins[:-1]) / 2

    # Indexes où évaluer les thresholds
    eval_idx = np.arange(0, hist.shape[0], resolution)
    if eval_idx[-1] != hist.shape[0]-1:
        eval_idx = np.append(eval_idx, hist.shape[0]-1)
    
    # Indexes des débuts et fins des tranches à utiliser pour calculer chaque threshold
    min_idx = eval_idx - n_slices // 2
    max_idx = eval_idx + n_slices // 2 + n_slices % 2

    # Correction des indexes pour qu'ils soient dans les limites de la matrice de données
    max_idx[min_idx < 0] = n_slices
    min_idx[min_idx < 0] = 0

    min_idx[max_idx > hist.shape[0]] = hist.shape[0] - n_slices
    max_idx[max_idx > hist.shape[0]] = hist.shape[0]

    # Calculer les threholds pour les tranches sélectionnées
    thresholds = np.array([
        DistributionSTFitResults.from_bins_hist(
            bins_center, hist[i_min:i_max].sum(axis=0), find_threshold=True
        ).threshold for i_min, i_max in zip(min_idx, max_idx)
    ], dtype=float)

    # Interpoler les thresholds pour les indexes intermédiaires
    # Ignorer les thresholds NaN (lorsque le fit n'a pas convergé)
    valid_idx = ~np.isnan(thresholds)
    thresholds_interp = np.interp(np.arange(hist.shape[0]), eval_idx[valid_idx], thresholds[valid_idx])

    return thresholds_interp
