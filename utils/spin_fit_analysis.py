import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import find_peaks
from scipy.integrate import quad
from dataclasses import dataclass, astuple, replace
from typing import NamedTuple, Self
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


from utils.functions import \
    fidelity_singlet, fidelity_triplet, \
    distribution_singlet, distribution_triplet, \
    distribution_ST

class DistributionSTParams(NamedTuple):
    """
    Object immuable représentant le résultat d'un fit.
    Instancié dans la dataclass `DistributionSTFitResult`
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
class DistributionSTFitResult:
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
        """
        Retour:
            self sous forme de np.array
            Sans argument: tous les attributs
            Avec arguments: juste les attributs de arg_names
    
        """
        if len(arg_names) == 0:
            return np.array([*self.popt, *(astuple(self)[1:])])
        
        if len(arg_names) == 1:
            return getattr(self, arg_names[0], eval(f"self.{arg_names[0]}"))
 
        res = np.empty((len(arg_names)))
        for i, arg in enumerate(arg_names):
            res[i] = getattr(self, arg, eval(f"self.{arg}"))
        # res = np.array([getattr(self, arg) for arg in arg_names])
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
            Une instance de la classe de données `DistributionSTFitResult` contenant les résultats calculés dans cette fonction

        Raises:
            ValueError: Si x_values et histogram n'ont pas la même taille.

        Niveaux de verbosité:
            0: Pas de message de sortie
            1: Affiche les paramètres trouvés par le fit
            2: Affichage du threshold (si demandé)
            4: Affichage du fit et du threshold trouvé (si demandé)
               Affichage des fidélités (si demandées)
            5: Affichage des distributions singulet et triplet individuelles
            6: Affichage des paramètres initiaux pour aider à trouver le fit
        """
        if len(x_values) != len(histogram):
            raise ValueError("`x_values` and `histogram` must have the same size.")
            
        results = DistributionSTFitResult()
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

            if verbosity >= 6:
                plt.plot(x_values, norm_factor * distribution_ST(x_values, *p0), label="Distribution before fit")
                plt.scatter(x_values[peaks], histogram[peaks], c="red", label="Peaks used as initial guess")
                plt.legend()

        bounds_min = (0, 0, x_values.min(), x_values.min(), 0)
        bounds_max = (1, np.inf, x_values.max(), x_values.max(), np.inf)

        # Fit de la distribution
        try:
            popt_unnamed, cov = curve_fit(distribution_ST, x_values, histogram / norm_factor, p0=p0, bounds=(bounds_min, bounds_max))
            popt = DistributionSTParams(*popt_unnamed)

            if popt.mus > popt.mut:
                p0_new = DistributionSTParams(mus=popt.mut, mut=popt.mus, sigma=popt.sigma, tm_over_T1=popt.tm_over_T1, Ps=popt.Ps)
                popt_unnamed, cov = curve_fit(distribution_ST, x_values, histogram / norm_factor, p0=p0_new, bounds=(bounds_min, bounds_max))
                popt = DistributionSTParams(*popt_unnamed)

                if popt.mus > popt.mut:
                    raise ValueError("La moyenne du singlet est plus grande que la moyenne du triplet.")

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
        
        if verbosity >= 5:
            distribution_t, (terme1, terme2) = distribution_triplet(x_values, popt.tm_over_T1, popt.mus, popt.mut, popt.sigma, separate_relaxation=True)
            plt.plot(x_values, norm_factor * popt.Ps * distribution_singlet(x_values, popt.mus, popt.sigma), label="Singlet distribution", ls=":")
            plt.plot(x_values, norm_factor * (1-popt.Ps) * distribution_t, label="Triplet distribution", ls=":")

            if verbosity >= 7:
                plt.plot(x_values, norm_factor * (1-popt.Ps) * terme1, label="Triplet: term without relax", ls=(0, (1, 1)))
                plt.plot(x_values, norm_factor * (1-popt.Ps) * terme2, label="Triplet: term with relax", ls=(0, (1, 1)))


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
                print(f"Visibility: {vis*100:.4f} %")
            if verbosity >= 4:
                print(f"Singlet fidelity: {results.singlet_fidelity*100:.4f} %")
                print(f"Triplet fidelity: {results.triplet_fidelity*100:.4f} %")

        if verbosity >= 4:
            plt.xlabel("Signal (a.u.)")
            plt.ylabel("Counts")
            plt.legend()
            plt.tight_layout()

        return results

def fit_slices(
    data2d, n_slices=1, resolution=1, long_axis=0, bins=401, arg_names=["threhold"],
) -> npt.NDArray[DistributionSTFitResult]:
    """
    Fonction qui fit des sous-ensembles de data2d pour tenir compte du drift du signal durant une
    longue mesure.

    Params:
        data2d: matrice de données
        n_slices: nombre de tranches à utiliser pour calculer chaque fit
        resolution: calculer le threshold tous les "resolution" tranches. Si la résolution est plus grande que 1, une interpolation linéaire est utilisée pour trouver les thresholds intermédiaires
        long_axis: axe le long duquel calculer le threshold
        bins: paramètre `bins` passé à la fonction `numpy.histogram()` pour calculer les histogrammes des tranches de données

    Ret:
    # TODO: faux, corriger.
        vecteur 1d de résultat de fit: DistributionSTFitResult
    """

    # Mettre l'axe long en premier pour faciliter les calculs
    data_transposed = np.moveaxis(data2d, long_axis, 0)

    # Calculer les histogrammes pour chaque tranche de données
    if type(bins) is int:
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

    # Calculer les fit pour les tranches sélectionnées
    fits = np.array([
        DistributionSTFitResult.from_bins_hist(
            bins_center, hist[i_min:i_max].sum(axis=0), find_threshold=True
        ) for i_min, i_max in zip(min_idx, max_idx)
    ], dtype=DistributionSTFitResult)

    # Interpoler les arg_names pour les indexes intermédiaires
    # Ignorer les `arg_names` NaN (lorsque le fit n'a pas convergé)
    interp = np.empty((hist.shape[0], len(arg_names)))
    for i, arg in enumerate(arg_names):
        to_interp = np.fromiter((fit.as_array(arg) for fit in fits), dtype=float)
        valid_idx = ~np.isnan(to_interp)
        interp[..., i] = np.interp(np.arange(hist.shape[0]), eval_idx[valid_idx], to_interp[valid_idx])

    return interp


def find_iq_rotation(data_i, data_q, current_iq_phase=0, log_norm=False, ax_aspect="equal", verbosity=0):
    """
    Trouve l'angle du plan IQ, par calcul du vecteur propre à valeur propre minimale de la matrice de covariance.

    Params:
        data_i, data_q: données
        current_iq_phase:
        log_norm: Si True, affiche la colorbar en échelle log
        ax_axpect: Paramètre passé à l'argument `aspect` de imshow()
        verbosity:
            1: print value
            2: print and plot
    Ret:
        Angle
    """

    hist, bins_i, bins_q = np.histogram2d(data_i.flatten(), data_q.flatten(), bins=100)
    hist_i, hist_q = hist.sum(axis=1), hist.sum(axis=0)
    bins_ic, bins_qc = (bins_i[1:] + bins_i[:-1]) / 2, (bins_q[1:] + bins_q[:-1]) / 2

    cov = np.cov(np.stack((data_q.flatten(), data_i.flatten()), axis=0))
    vals, vecs = np.linalg.eig(cov)
    long_vec = vecs[np.argmin(np.abs(vals))]

    pente = long_vec[1] / long_vec[0]
    i_tri, q_tri = data_i[:, 0].mean(), data_q[:, 0].mean()
    i_sin, q_sin = data_i[:, -1].mean(), data_q[:, -1].mean()
    delta_i = i_tri - i_sin
    delta_q = q_tri - q_sin
    correction = np.arctan2(np.sign(delta_q) * pente, np.sign(delta_i))

    theta = current_iq_phase - correction

    if verbosity >= 1:
        print(f"Angle de rotation du plan IQ: {theta} rad (correction de {correction} rad)")

    if verbosity >= 2:

        fig, axs = plt.subplot_mosaic(
            [["hist_i", "."], ["iq_plane", "hist_q"]],
            gridspec_kw={"width_ratios": [4, 1], "height_ratios": [1, 4]},
            figsize=(6,4),
        )

        norm = LogNorm() if log_norm else None
        axs["iq_plane"].imshow(
            hist.T, extent=[*bins_i[[0, -1]], *bins_q[[0, -1]]], aspect=ax_aspect, origin="lower", interpolation="none", norm=norm)
        axs["iq_plane"].set_xlabel("I Quadrature (a.u.)")
        axs["iq_plane"].set_ylabel("Q Quadrature (a.u.)")
        
        x_centre, y_centre = data_i.mean(), data_q.mean()
        B = y_centre - pente * x_centre
        axs["iq_plane"].plot(bins_i, pente*bins_i+B, c="red", label="Droite de correction IQ")

        axs["hist_i"].plot(bins_ic, hist_i)
        axs["hist_i"].set_xticks([])
        axs["hist_i"].margins(x=0)
        axs["hist_i"].set_ylabel("Counts")

        axs["hist_q"].plot(hist_q, bins_qc)
        axs["hist_q"].set_yticks([])
        axs["hist_q"].margins(y=0)
        axs["hist_q"].set_xlabel("Counts")

        plt.tight_layout()


    return theta
