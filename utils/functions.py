import numpy as np
from scipy.special import erf
from scipy.integrate import quad

# Répertoire de fonctions mathématiques

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

def distribution_triplet(x, tm_over_T1, mus, mut, sigma, separate_relaxation=False):
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
        separate_relaxation: Si True, retourne les termes 1 et 2, associés à 
            la relaxation de l'état T vers S.
    
    Returns:
        Si separate_relaxation est False: Valeurs de la distribution aux points x.
        Si separate_relaxation est True: (Valeurs de la distribution aux points x, (terme sans relaxation, terme relaxé)).
    """
    delta_mu = mut - mus
    mu_bar = x - sigma**2 * tm_over_T1 / delta_mu

    terme1 = np.exp(-tm_over_T1) * gaussian(x, mut, sigma)
    terme2 = tm_over_T1 / (2 * delta_mu) * \
        np.exp(tm_over_T1 / delta_mu * (mus - x + sigma**2 * tm_over_T1 / (2 * delta_mu))) * \
        (erf((mut - mu_bar) / (np.sqrt(2) * sigma)) - erf((mus - mu_bar) / (np.sqrt(2) * sigma)))
    
    if separate_relaxation:
        return terme1 + terme2, (terme1, terme2)

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

def oscillations_rabi(t, Ps0, T2, f0, ramp_time_correction):
    """
    Fonction d'une oscillation dans le temps, normalisée et doublement amortie

    Args:
        Ps0: hauteur de la première oscillation
        T1: amortissement de la moyenne
        T2: amortissement des oscillations
        f0: fréquence
        ramp_time_correction: décalage du temps pour tenir compte de la rampe
    Returns:
        la fonction...
    """
    t_ = t + ramp_time_correction  # Déplacement du temps

    relax = 1 / 2
    decoh = Ps0/2 * np.exp(-t_ / T2) * np.cos(2*np.pi*f0*t_)

    return relax + decoh

def oscillations_rabi_fourier(f, A, T1, f0):
    num = 1/T1 + 2.j*np.pi*f
    den = num**2 + 4 * np.pi**2 * f0**2
    return A/2 * num / den