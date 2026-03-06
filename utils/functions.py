import numpy as np
from scipy.special import erf
from scipy.integrate import quad

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
