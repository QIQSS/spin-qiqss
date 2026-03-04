import numpy as np
from scipy.special import erf
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import find_peaks
from scipy.integrate import quad
from dataclasses import dataclass
from typing import NamedTuple
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

# Fonctions pour les probabilités des distribution des états S et T
def gaussian(x, mu, sigma):
    """
    Distribution gaussienne
    """
    return np.exp(-(x-mu)**2 / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)

def distribution_singlet(x, mus, sigma):
    """
    Distribution des points d'un ensemble de mesure d'états singulets.

    Args:
        x: Valeurs de l'axe sur lequel calculer la distribution.
        mus: Valeur centrale de la distribution.
        sigma: Écart-type de la distribution.
    
    Returns:
        Valeurs de la distribution aux points x.
    """
    return gaussian(x, mus, sigma)

def distribution_triplet(x, tm_over_T1, mus, mut, sigma):
    """
    Distribution des points d'un ensemble de mesure d'états triplets.
    Cette distribution modélise des états triplets (distribution gaussienne)
    qui peuvent relaxer vers l'état singulet avec un temps caractéristique T1.

    Args:
        x: Valeurs de l'axe sur lequel calculer la distribution.
        tm_over_T1: Ratio du temps d'intégration sur le temps de relaxation.
        mus: Moyenne de la distribution de l'état S.
        mut: Moyenne de la distribution de l'état T.
        sigma: Écart-type commun aux deux distributions.
    
    Returns:
        Valeurs de la distribution aux points x.
    """
    delta_mu = mut - mus
    mu_bar = x - sigma**2 * tm_over_T1 / delta_mu

    terme1 = np.exp(-tm_over_T1) * gaussian(x, mut, sigma)
    terme2 = tm_over_T1 / (2 * delta_mu) * \
        np.exp(tm_over_T1 / delta_mu * (mus - x + sigma**2 * tm_over_T1 / (2 * delta_mu))) * \
        (erf((mut - mu_bar) / (np.sqrt(2) * sigma)) - erf((mus - mu_bar) / (np.sqrt(2) * sigma)))
    
    return terme1 + terme2

def distribution_ST(x, Ps, tm_over_T1, mus, mut, sigma):
    """
    Distribution d'un ensemble de mesure contenant initialement des états
    singulet et triplet avec des proportions Ps, 1-Ps.

    Args:
        x: Valeurs de l'axe I sur lesquelles calculer la distribution.
        Ps: Probabilité d'initialiser dans l'état S.
        tm_over_T1: Ratio du temps d'intégration sur le temps de relaxation.
        mus: Moyenne de la distribution de l'état S.
        mut: Moyenne de la distribution de l'état T.
        sigma: Écart-type commun aux deux distributions.
    
    Returns:
        Valeurs de la distribution totale sur les points x.
    """
    return Ps * distribution_singlet(x, mus, sigma) + (1 - Ps) * distribution_triplet(x, tm_over_T1, mus, mut, sigma)


def fidelity_singlet(threshold, mus, sigma):
    """
    Calcule la fidélité liée aux singulets.

    Args:
        threshold: Threshold utilisé pour calculer la fidélité
        mus: Valeur moyenne de la distribution des états singulets
        sigma: Écart-type de la distribution des états singulets

    Returns:
        La fidélité
    """
    return (1 + erf((threshold - mus) / (np.sqrt(2) * sigma))) / 2

def fidelity_triplet(threshold, tm_over_T1, mus, mut, sigma):
    """
    Calcule la fidélité liée aux triplets.

    Args:
        threshold: Threshold utilisé pour calculer la fidélité
        tm_over_T1: Ratio du temps d'intégration sur le temps de relaxation.
        mus: Valeur moyenne de la distribution des états singulets
        mut: Valeur moyenne de la distribution des états triplets
        sigma: Écart-type des distributions des états singulet et triplet

    Returns:
        La fidélité
    """
    return 1 - quad(lambda x: distribution_triplet(x, tm_over_T1, mus, mut, sigma), mut-100*sigma, threshold)[0]


class DistributionSTParams(NamedTuple):
    """ Object immuable représentant le résultat d'un fit
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
    popt: DistributionSTParams = None
    threshold: float = None
    singlet_fidelity: float = None
    triplet_fidelity: float = None
    visibility: float = None


def fit_distribution_ST(x_values, histogram, p0=None, find_threshold=False, compute_visibility=False, debug_plot=False) -> DistributionSTFitResults:
    """
    Cette fonction ajuste la fonction `distribution_ST()` à un ensemble de données (normalisées) passé en arguments.
    La fonction retourne une classe de données qui peut contenir les paramètres trouvés par le fit, le threshold qui
    permet de maximiser la visibilité et la visibilité, selon ce qui est spécifié par l'utilisateur.

    Args:
        x_values: Points en x associés aux données de l'histogramme
        histogram: Histogramme de données sur lequel sera ajustée la fonction
        p0: Paramètres initiaux passés à la fonction `curve_fit()`. Si `None`, des paramètres initiaux seront estimés
            à partir des pics de l'histogramme identifiés par la fonction `find_peaks()`
        find_threshold: Demande à la fonction de trouver le threshold
        compute_visibility: Demande à la fonction de calculer la visibilité
        debug_plot: Demande à la fonction d'afficher des graphiques pour aider à débugguer
    
    Returns:
        Une instance de la classe de données `DistributionSTFitResults` contenant les résultats calculés dans cette fonction
    """
    results = DistributionSTFitResults()
    norm_factor = (x_values[1] - x_values[0]) * histogram.sum()  # Facteur de normalisation pour que l'intégrale sur l'ensemble des valeurs soit 1.
    
    if compute_visibility:  # Il faut le threshold pour calculer la visibilité.
        find_threshold = True
    
    if debug_plot:
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

        if debug_plot:
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
    except:
        results.popt = None
        return results

    if debug_plot:
        plt.plot(x_values, norm_factor * distribution_ST(x_values, *popt), label="Fitted distribution")

    # Trouver le threshold
    if find_threshold:
        results.threshold = results.popt.find_optimal_threshold(x_values)

        if debug_plot and results.threshold is not None:
            plt.axvline(results.threshold, ls="--", c="red", label="Threshold")
    
    # Calculer la visibilité
    if compute_visibility and results.threshold is not None:
        fid_sin, fid_tri, vis = results.popt.get_visibility(x_values, results.threshold)
        results.singlet_fidelity = fid_sin
        results.triplet_fidelity = fid_tri
        results.visibility = vis

    if debug_plot:
        plt.xlabel("X values (a.u.)")
        plt.ylabel("Counts (normalized)")
        plt.legend()
        plt.tight_layout()

    return results


##########
def get_file_variables(file, var_to_exclude=list()):
    """
    Retourne l'ensemble des variables d'un fichier importé à l'exception des variables
    commençant par "__" et des variables inclues dans `var_to_exclude`.

    Args:
        file: Fichier importé
        var_to_exclude (list[str]): Liste contenant le nom des variables à ne pas récolter
    
    Returns:
        dict: Dictionnaire contenant les variables du fichier
    """
    variables = dir(file)
    variables = filter(lambda var: not var.startswith("__"), variables)
    variables = filter(lambda var: var not in var_to_exclude, variables)
   
    return {var: file.__dict__[var] for var in variables}

def get_file_code(file):
    """
    Retourne le code d'un fichier importé.
    Args:
        file: Fichier importé
    
    Returns:
        str: Code du fichier
    """
    path = file.__file__
    with open(path, "r", encoding="utf-8") as f:
        txt = "".join(f.readlines())
    return txt
